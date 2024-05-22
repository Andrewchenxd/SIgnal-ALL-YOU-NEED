"""
part of codes is adopted from:
https://github.com/mxbastidasr/DAWN_WACV2020
"""

import torch
import torch.nn as nn

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

        self.conv_even = lambda x: x[:, :, ::2]
        self.conv_odd = lambda x: x[:, :, 1::2]

    def forward(self, x):
        """
        returns the odd and even part
        :param x:
        :return: x_even, x_odd
        """
        return self.conv_even(x), self.conv_odd(x)


class Operator(nn.Module):
    def __init__(self, in_planes, kernel_size=3, dropout=0.):
        super(Operator, self).__init__()

        pad = (kernel_size - 1) // 2 + 1

        self.operator = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Operator as Predictor() or Updator()
        :param x:
        :return: P(x) or U(x)
        """
        x = self.operator(x)
        return x


class LiftingScheme(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(LiftingScheme, self).__init__()

        self.split = Splitting()

        self.P = Operator(in_planes, kernel_size)
        self.U = Operator(in_planes, kernel_size)

    def forward(self, x):
        """
        Implement Lifting Scheme
        :param x:
        :return: c: approximation coefficient
                 d: details coefficient
        """
        (x_even, x_odd) = self.split(x)
        c = x_even + self.U(x_odd)
        d = x_odd - self.P(c)
        return c, d


class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, kernel_size, regu_details, regu_approx):
        super(LevelTWaveNet, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.wavelet = LiftingScheme(in_planes, kernel_size=kernel_size)

    def forward(self, x):
        """
        Conduct decomposition and calculate regularization terms
        :param x:
        :return: approx component, details component, regularization terms
        """
        global regu_d, regu_c
        (L, H) = self.wavelet(x)  # 10 9 128
        approx = L
        details = H
        if self.regu_approx + self.regu_details != 0.0:
            if self.regu_details:
                regu_d = self.regu_details * H.abs().mean()
            # Constrain on the approximation
            if self.regu_approx:
                regu_c = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                # Only the details
                regu = regu_d
            elif self.regu_details == 0.0:
                # Only the approximation
                regu = regu_c
            else:
                # Both
                regu = regu_d + regu_c

            return approx, details, regu


class AWN(nn.Module):
    def __init__(self,
                 num_classes,
                 num_levels=1,
                 in_channels=64,
                 kernel_size=3,
                 latent_dim=256,
                 regu_details=0.01,
                 regu_approx=0.01):
        super(AWN, self).__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = self.in_channels * (self.num_levels + 1)
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.regu_details = regu_details
        self.regu_approx = regu_approx

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            # Call a 2d Conv to integrate I, Q channels
            nn.Conv2d(1, self.in_channels,
                      kernel_size=(2, 7), stride=(1,), bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels,
                      kernel_size=(5,), stride=(1,), padding=(2,), bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.levels = nn.ModuleList()

        for i in range(self.num_levels):
            self.levels.add_module(
                'level_' + str(i),
                LevelTWaveNet(self.in_channels,
                              self.kernel_size,
                              self.regu_details,
                              self.regu_approx)
            )

        self.SE_attention_score = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 4, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels // 4, self.out_channels, bias=False),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.latent_dim, num_classes)
        )

    def forward(self, x):
        # x = x.unsqueeze(1)  # x:[N, 2, T] -> [N, 1, 2, T]
        x = self.conv1(x)
        x = x.squeeze(2)  # x:[N, C, 1, T] -> [N, C, T]
        x = self.conv2(x)
        regu_sum = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        for l in self.levels:
            x, details, regu = l(x)
            regu_sum += [regu]
            det += [self.avgpool(details)]
        aprox = self.avgpool(x)
        det += [aprox]

        x = torch.cat(det, 1)
        x = x.view(-1, x.size()[1])
        x = torch.mul(self.SE_attention_score(x), x)
        # logit =x
        logit = self.fc(x)

        return logit, regu_sum

if __name__ == '__main__':
    import time
    net1= AWN(num_classes=10,
              num_levels=1,
                 in_channels=64,
                 kernel_size=3,
                 latent_dim=256,
                 regu_details=0.01,
                 regu_approx=0.01).cuda()
    # print(net1)
    total = 32
    a = torch.randn((total, 1, 128, 128)).cuda()
    b = torch.randn((total, 2, 128)).cuda()
    c = torch.randn((total, 1, 17)).cuda()
    # net1(a, b, c)
    for _ in range(100):
        net1(b)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        net1(b)
    torch.cuda.synchronize()
    end = time.perf_counter()

    print('{} ms'.format((end - begin) / (100 * total) * 1000))


    def count_parameters_in_MB(model):
        return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)


    num_params = count_parameters_in_MB(net1)
    print(f'Number of parameters: {num_params}')