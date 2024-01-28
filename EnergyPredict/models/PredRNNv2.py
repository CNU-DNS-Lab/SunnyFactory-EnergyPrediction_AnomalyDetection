import random
import torch
from torch import nn
from torch.nn import functional as F


# SpatioTemporalLSTMCell
class PredRNNv2Cell(nn.Module):
    def __init__(self, in_channels: int, num_hidden: int, height: int, width: int, kernel_size: int, stride: int):
        super(PredRNNv2Cell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = kernel_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_hidden * 7, kernel_size=kernel_size,
                      stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 7, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden * 4, kernel_size=kernel_size,
                      stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 4, height, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden * 3, kernel_size=kernel_size,
                      stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 3, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden * 2, out_channels=num_hidden, kernel_size=kernel_size,
                      stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(in_channels=num_hidden * 2, out_channels=num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = (f_t * c_t) + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.sigmoid(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = (f_t_prime * m_t) + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new, delta_c, delta_m


class PredRNNv2(nn.Module):
    def __init__(self, num_layers: int, num_hidden: list, shape: tuple, kernel_size: int, stride: int,
                 seq_len: int = 24):
        super(PredRNNv2, self).__init__()

        T, C, H, W = shape
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.seq_len = seq_len
        self.total_len = seq_len * 2
        cell_list = []

        for i in range(num_layers):
            in_channels = 1 if i == 0 else num_hidden[i - 1]
            cell_list.append(
                PredRNNv2Cell(in_channels=in_channels, num_hidden=num_hidden[i], height=H, width=W,
                              kernel_size=kernel_size, stride=stride)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(in_channels=num_hidden[num_layers - 1], out_channels=1, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.adapter = nn.Conv2d(in_channels=num_hidden[0], out_channels=num_hidden[0], kernel_size=1,
                                 stride=1, padding=1, bias=False)

    def forward(self, x, ratio: float = 0.0):
        device = x.device
        B, T, C, H, W = x.shape
        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.num_layers):
            zeros = torch.zeros([B, self.num_hidden[i], H, W]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([B, self.num_hidden[0], H, W]).to(device)

        for t in range(self.seq_len):
            if t == 0:
                net = x[:, t]
            else:
                # reverse_schedule_sampling
                net = x_gen if random.random() < ratio else x[:, t]

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
            )
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2
            )

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
                )
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2
                )

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        return torch.stack(next_frames, dim=1)
