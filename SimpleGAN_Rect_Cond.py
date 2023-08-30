import torch
import torch.nn as nn


def get_generator(ngpu, nz, conditions):
    ngf = 64
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    nc = 1

    class Generator(nn.Module):
        def __init__(self, ngpu, conditions):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.dim_z = nz
            self.shift_in_w_space = False

            self.n_conditions = len(conditions)
            self.n_classes = 5

            self.label_conditioned_generator = nn.Sequential(
                nn.Linear(1, 100),
                nn.Linear(100, 4*4)
            )

            self.latent = nn.Sequential(
                nn.Linear(self.dim_z - 1, 4*4 * ngf * 8),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.main = nn.Sequential(
                # input is Z, going into a convolution
                #nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                #nn.BatchNorm2d(ngf * 8),
                #nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8 + self.n_conditions, ngf * 8, (3, 4), (1, 2), 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 16
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 32
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 64
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 128
            )

        def forward(self, z, shift=None):
            if shift is None:
                out = z.clone()
            else:
                out = z + shift

            if out.ndim == 2:
                out = out.unsqueeze_(2)
                out = out.unsqueeze_(3)

            cond = out[:, :1, :, :]
            noise = out[:, 1:, :, :]

            label_output = self.label_conditioned_generator(cond.view(cond.shape[0],
                                                            self.n_conditions)).view(-1, 1, 4, 4)
            latent_output = self.latent(noise.view(noise.shape[0],
                                                   self.dim_z - self.n_conditions)).view(-1, ngf * 8, 4, 4)
            out = torch.cat((latent_output, label_output), dim=1)

            #print("G")
            #print(out.shape)
            #out = self.main(out)
            #print(out.shape)
            #return out
            return self.main(out)

    return Generator(ngpu, conditions).to(device)


def get_discriminator(ngpu, conditions):
    ndf = 64
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    nc = 1

    class Discriminator(nn.Module):
        def __init__(self, ngpu, conditions):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu

            self.n_conditions = len(conditions)
            self.n_classes = 5

            self.label_encoding = nn.Sequential(  # Dense layers for encoding the labels
                nn.Linear(1, 100),
                nn.Linear(100, 1 * 64 * 128)
            )

            self.main = nn.Sequential(
                # input is (nc + amount of conditions) x 64 x 128
                nn.Conv2d(nc + self.n_conditions, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 64
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 32
                nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 8
                nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),  # 3 1 1 makes it keep the size
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 8
                nn.Conv2d(ndf * 4, ndf * 8, (3, 4), (1, 2), 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input, labels):
            #print("D")
            #print(input.shape)
            #print(labels.shape)
            label_encoding = self.label_encoding(labels.view(labels.shape[0],
                                                             self.n_conditions)).view(-1, self.n_conditions, 64, 128)
            #print(label_encoding.shape)
            out = torch.cat((input, label_encoding), dim=1)
            #print(catted.shape)
            #out = self.main(input)
            #print(out.shape)
            #return out
            return self.main(out)

    return Discriminator(ngpu, conditions).to(device)
