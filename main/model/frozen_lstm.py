"""Frozen LSTM for reservoirs (reservoir_dedup): the input projection runs once per DISTINCT readin input, and the backward
computes only data gradients (the weights are frozen).

The readin input at step t is (observation bits at t, the episode's Z), and Z is constant over an episode, so the pair
(episode, observation pattern) identifies it: at batch 8000 only ~82k of the 240k (episode, step) rows are distinct
(measured 2026-10-13). The forward computes W_ih * relu(readin(x)) once per distinct input and gathers it into each
step; the backward sums the gate gradients of the rows that share an input before the one multiply by W_ih, which is
exact by linearity. The recurrence h_{t-1} W_hh^T (forward) and dgates W_hh (backward) is unchanged; the shared initial
state (STM, LTM) makes step 1's recurrent product a single vector product. Gate order i, f, g, o and the cell are
nn.LSTM's (aten::_thnn_fused_lstm_cell). In fp64 this matches nn.LSTM to ~1e-16 in the outputs and in the gradients of
the readin input, STM and LTM; in fp16 it has the same error against fp64 as cuDNN (session scratchpad tests,
2026-10-13). Ported from the survey prototype reservoir_opt/frozen_lstm_dedup.py.
"""
import torch

_cell = torch.ops.aten._thnn_fused_lstm_cell
_cell_bwd = torch.ops.aten._thnn_fused_lstm_cell_backward_impl


class FrozenLSTMDedup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Au, inv, h0, c0, Wih, Whh, bias):
        """Au (U, N): readin output of each distinct input; inv (T, B) long: the distinct input episode b sees at step t;
        h0, c0 (1, N): shared initial state; Wih, Whh (4N, N) and bias (4N,) frozen, in the compute dtype.
        returns: (B, T, N) hidden states, in the compute dtype"""
        ctx.dtypes = (Au.dtype, h0.dtype, c0.dtype)
        with torch.autocast(device_type = Au.device.type, enabled = False):
            T, B = inv.shape
            U, N = Au.shape
            dt = Whh.dtype
            GXu = torch.addmm(bias, Au.to(dt), Wih.t())                    # (U, 4N): input gates once per distinct input
            WS = torch.empty(T, B, 4 * N, device = Au.device, dtype = dt)  # per step: input gates, then the activated gates
            C = torch.empty(T + 1, B, N, device = Au.device, dtype = dt)   # c_0 .. c_T
            HT = torch.empty(T, B, N, device = Au.device, dtype = dt)
            C[0] = c0.to(dt).expand(B, N)
            hg0 = (h0.to(dt) @ Whh.t()).expand(B, 4 * N).contiguous()       # shared initial state: one vector product
            for t in range(T):
                torch.index_select(GXu, 0, inv[t], out = WS[t])
                hg = hg0 if t == 0 else HT[t - 1] @ Whh.t()
                _cell.out(WS[t], hg, C[t], None, None, out0 = HT[t], out1 = C[t + 1], out2 = WS[t])
            del GXu
        ctx.save_for_backward(WS, C, Wih, Whh, inv)
        ctx.U = U
        return HT.transpose(0, 1)

    @staticmethod
    def backward(ctx, dH):
        WS, C, Wih, Whh, inv = ctx.saved_tensors
        with torch.autocast(device_type = dH.device.type, enabled = False):
            T, B, N4 = WS.shape
            N = N4 // 4
            dt = Whh.dtype
            acc = torch.float64 if dt == torch.float64 else torch.float32
            dH = dH.to(dt)
            Du = torch.zeros(ctx.U, N4, device = dH.device, dtype = acc)       # gate gradients summed per distinct input
            dh = dH[:, T - 1].contiguous()
            dc = torch.zeros(B, N, device = dH.device, dtype = dt)
            for t in reversed(range(T)):
                dg, dc, _ = _cell_bwd(dh, dc, C[t], C[t + 1], WS[t], False)
                Du.index_add_(0, inv[t], dg.to(acc))
                if t > 0:
                    dh = torch.addmm(dH[:, t - 1], dg, Whh)
                else:
                    dh0 = dg.to(acc).sum(0, keepdim = True).to(dt) @ Whh
                    dc0 = dc.to(acc).sum(0, keepdim = True)
            dAu = Du.to(dt) @ Wih                                           # one multiply per distinct input
        a_dt, h_dt, c_dt = ctx.dtypes
        return dAu.to(a_dt), None, dh0.to(h_dt), dc0.to(c_dt), None, None, None


def frozen_lstm_dedup(Au, inv, h0, c0, Wih, Whh, bias):
    return FrozenLSTMDedup.apply(Au, inv, h0, c0, Wih, Whh, bias)
