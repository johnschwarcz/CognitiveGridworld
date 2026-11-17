"""readin_shared (2026-11-12, John: can the Classifier escape its missing-channel trap with the Generator's read-in
structure?). The Classifier's input at each step is every channel's bit and every channel's Z. By default one dense layer
reads all of them, so every channel enters through its own random weights and each channel has to start being used on its
own (stalled Classifier-route nets ignore whole channels: claude_plots/fig_channel_census.py). With readin_shared, ONE
function reads each channel's (bit, Z) pair and the results are averaged over channels: the same function for every
channel, as in the Generator's Z input and in exact Bayes (whose evidence is a sum of the same per-channel term).

readin_channels (11-13, John: "training on 2 and testing on 5"): the read-in averages over only the listed channels (the
Generator still learns every channel's code); setting .channels at test changes which channels it reads.

h_t = mean_o relu(W [bit_o,t, Z_o] + b). A bit is 0 or 1, so per episode only the two outputs per channel are computed:
h_t = mean_o a0_o + bits_t . (a1 - a0) / O, exact and no bigger than the dense layer's output."""
import torch
from torch import nn


class SharedChannelReadin(nn.Module):
    def __init__(self, obs_num, Z_num, hid_dim, channels=None):
        super().__init__()
        self.O, self.Zn = obs_num, Z_num
        self.f = nn.Linear(1 + Z_num, hid_dim)
        self.channels = None if channels is None else list(channels)   # readin_channels (11-13): read only these

    def episode(self, bits, Z):
        """bits (B, T, O) in {0, 1}, Z (B, O, Z_num) -> (B, T, hid)"""
        if self.channels is not None:
            bits, Z = bits[:, :, self.channels], Z[:, self.channels]
        z = Z @ self.f.weight[:, 1:].T + self.f.bias                             # (B, O, hid)
        a0, a1 = torch.relu(z), torch.relu(z + self.f.weight[:, 0])
        return (a0.sum(1)[:, None] + torch.einsum("bto,boh->bth", bits.to(z.dtype), a1 - a0)) / Z.shape[1]

    def forward(self, x):
        """the generic input of classifier_to_hid: x (..., O + O * Z_num) = bits, then every channel's Z"""
        bits = x[..., :self.O]
        Z = x[..., self.O:].reshape(*x.shape[:-1], self.O, self.Zn)
        return torch.relu(self.f(torch.cat([bits[..., None], Z], -1))).mean(-2)
