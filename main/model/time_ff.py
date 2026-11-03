"""Time-injected feedforward net (mode FF, ff_time_inject; John, 2026-10-15). T = step_num layers, one per time step.
Layer k (k = 1 farthest from the readout) gets, through its own trainable readin, the mean of the first T + 1 - k
observations together with Z, added to the frozen random projection of layer k - 1's output; so an average over n
observations passes through n layers before the readout, as in an RNN unrolled over n steps. One readout after layer T,
trained on the final step like every FF net.

The T - 1 frozen N x N layers are nn.Linear's default draws (weight and bias U(-1/sqrt(N), 1/sqrt(N))), drawn from a
per-layer seed, with the weight times ff_frozen_gain. At gain 1 the input-dependent part of the signal shrinks ~0.4x per
layer and different inputs give the same output after ~9 layers, which no readin scale can undo (needs readin weights
~1e9); gain sqrt(6), i.e. Kaiming-uniform, keeps it alive through 29 layers (measured at N = 2000, 2026-10-15). The
frozen layers are stored in the compute dtype, or, with ff_regenerate, rebuilt from their seeds on every use (forward
and backward) so that only one layer is resident: 29 layers at N = 20000 are 23 GB even in fp16.

first_only (ff_inject_first_only; 2026-10-16) is the control without time injection: only layer 1 gets input, the mean
of all T observations with Z, as the standard FF's readin does, so the net is the standard depth-T FF with these frozen
layers (one readin, 36 N + 20 trainable parameters like the shared-readin net).
"""
import math
import torch

_CHUNK = 4096                                               # rows per draw: fixed, so a layer's weights are reproducible


def frozen_layer(seed, k, n, gain, dtype, device):
    """Weight (gain x U(-1/sqrt(n), 1/sqrt(n)), n x n) and bias (U(-1/sqrt(n), 1/sqrt(n))) of frozen layer k, drawn in fp32
    from its own generator in fixed row chunks, then cast: the same tensors every call."""
    g = torch.Generator(device = device)
    g.manual_seed(1_000_003 * seed + k)
    bound = 1 / math.sqrt(n)
    W = torch.empty(n, n, device = device, dtype = dtype)
    for r in range(0, n, _CHUNK):
        rows = min(_CHUNK, n - r)
        W[r:r + rows] = ((torch.rand(rows, n, generator = g, device = device) * 2 - 1) * (gain * bound)).to(dtype)
    b = ((torch.rand(n, generator = g, device = device) * 2 - 1) * bound).to(dtype)
    return W, b


class RegeneratedLinear(torch.autograd.Function):
    """y = h W_k^T + b_k with W_k, b_k rebuilt from the seed in forward and in backward; only the data gradient exists."""
    @staticmethod
    def forward(ctx, h, seed, k, gain):
        ctx.args = (seed, k, gain, h.dtype)
        with torch.autocast(device_type = h.device.type, enabled = False):
            W, b = frozen_layer(seed, k, h.shape[-1], gain, h.dtype, h.device)
            return torch.addmm(b, h, W.t())

    @staticmethod
    def backward(ctx, dy):
        seed, k, gain, dtype = ctx.args
        with torch.autocast(device_type = dy.device.type, enabled = False):
            W, _ = frozen_layer(seed, k, dy.shape[-1], gain, dtype, dy.device)
            return dy.to(dtype) @ W, None, None, None


class ScaleByGain(torch.autograd.Function):
    """z * g for a 0-dim fp32 gain g: the product in z's dtype (exactly z at g = 1); the gain's gradient summed in fp32,
    since an fp16 sum of B x N terms of the loss-scaled gradient overflows."""
    @staticmethod
    def forward(ctx, z, g):
        ctx.save_for_backward(z, g)
        return z * g.to(z.dtype)

    @staticmethod
    def backward(ctx, dy):
        z, g = ctx.saved_tensors
        acc = torch.promote_types(dy.dtype, torch.float32)                  # fp32 at least (fp64 stays fp64)
        return dy * g.to(dy.dtype), (dy.to(acc) * z.to(acc)).sum().to(g.dtype).reshape(g.shape)


class TimeInjectedFF(torch.nn.Module):
    def __init__(self, in_dims, n, steps, gain, regenerate, device, seed = 0, shared = False, first_only = False,
                 layer_gains = False):
        """Frozen layers are drawn on `device` in both modes: CPU and CUDA generators give different numbers for one seed.
        shared: one readin for every layer (each layer still gets its own average), 36 N + 20 trainable parameters as a
        depth-1 FF, instead of one readin per layer (500 N + 20). first_only: input at layer 1 only (one readin).
        layer_gains (ff_layer_gains; John 2026-10-25): one trainable scalar per layer multiplying that layer's input (the
        readin's output, its bias included), initialised at 1, so the layers' inputs can be weighted differently; with
        all gains at 1 the net is exactly the one without them (and draws no random numbers, so its init is the same)."""
        super().__init__()
        self.n, self.steps, self.gain, self.regenerate, self.seed = n, steps, gain, regenerate, seed
        self.shared, self.first_only = shared or first_only, first_only
        self.readins = torch.nn.ModuleList([torch.nn.Linear(in_dims, n) for _ in range(1 if self.shared else steps)])   # trainable
        self.layer_gains = torch.nn.Parameter(torch.ones(steps, device = device)) if layer_gains else None   # trainable
        if not regenerate:                                   # stored, not saved: rebuilt from the seeds on load
            for k in range(1, steps):
                W, b = frozen_layer(seed, k, n, gain, torch.float32, device)
                self.register_buffer(f"W{k}", W, persistent = False)
                self.register_buffer(f"b{k}", b, persistent = False)

    def frozen(self, k, h):
        if self.regenerate:
            return RegeneratedLinear.apply(h, self.seed, k, self.gain)
        return torch.nn.functional.linear(h, getattr(self, f"W{k}").to(h.dtype), getattr(self, f"b{k}").to(h.dtype))

    def forward(self, prefix_means, Z):
        """prefix_means (B, T, obs): mean of the first t observations at index t - 1; Z (B, z). returns: (B, N)"""
        T = self.steps
        h = None
        for k in range(T):                                   # layer k + 1 gets the mean of the first T - k observations
            if self.first_only and k > 0:                    # no time injection: the frozen stack alone above layer 1
                h = torch.relu(self.frozen(k, h))
                continue
            z = self.readins[0 if self.shared else k](torch.cat((prefix_means[:, T - 1 - k], Z), -1))
            if self.layer_gains is not None:
                z = ScaleByGain.apply(z, self.layer_gains[k])
            h = torch.relu(z if h is None else z + self.frozen(k, h))
        return h
