import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, num_hidden: int, height: int, width: int, kernel_size: int, stride: int):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = kernel_size // 2
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_hidden * 4, kernel_size=kernel_size,
                      stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 4, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden, out_channels=num_hidden * 4, kernel_size=kernel_size,
                      stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 4, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden * 2, out_channels=num_hidden, kernel_size=kernel_size,
                      stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden, height, width])
        )

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = (f_t * c_t) + (i_t * g_t)
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, num_layers: int, num_hidden: list, shape: tuple, kernel_size: int, stride: int,
                 seq_len: int = 24):
        super(ConvLSTM, self).__init__()

        T, C, H, W = shape
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.seq_len = seq_len
        self.total_len = seq_len * 2
        cell_list = []

        for i in range(num_layers):
            in_channels = 1 if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channels=in_channels, num_hidden=num_hidden[i], height=H, width=W,
                             kernel_size=kernel_size, stride=stride)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], out_channels=1, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x):
        device = x.device
        B, T, C, H, W = x.shape
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([B, self.num_hidden[i], H, W]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.seq_len):
            h_t[0], c_t[0] = self.cell_list[0](x[:, t], h_t[0], c_t[0])
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        return torch.stack(next_frames, dim=1)
