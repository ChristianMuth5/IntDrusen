from __future__ import print_function

import io
import json
import logging
import os
import sys
import shutil
import glob
from enum import Enum
from time import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import truncnorm
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet18
from torchvision.transforms import Resize
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm, tqdm_notebook
import subprocess
from visualization import generate_gifs_warp, generate_gifs_linear, generate_gifs_warp_int, generate_gifs_linear_int


def map_for_stylegan(input):
    out = torch.index_select(input, 1, torch.tensor([0]).cuda())
    return out.squeeze(dim=1)

# Taken and adapted from WarpedGANSpace
class StyleGAN2Wrapper(nn.Module):
    def __init__(self, G, shift_in_w_space):
        super(StyleGAN2Wrapper, self).__init__()
        self.G = G
        self.shift_in_w_space = shift_in_w_space
        self.dim_z = G.z_dim
        self.dim_w = G.w_dim if self.shift_in_w_space else self.dim_z

    def get_w(self, z):
        """Return batch of w latent codes given a batch of z latent codes.

        Args:
            z (torch.Tensor) : Z-space latent code of size [batch_size, 512]

        Returns:
            w (torch.Tensor) : W-space latent code of size [batch_size, 512]

        """
        return map_for_stylegan(self.G.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8))

    def forward(self, z, shift=None):
        """StyleGAN2 generator forward function.

        Args:
            z (torch.Tensor)     : Batch of latent codes in Z-space
            shift (torch.Tensor) : Batch of shift vectors in Z- or W-space (based on self.shift_in_w_space)

        Returns:
            I (torch.Tensor)     : Output images of size [batch_size, 3, resolution, resolution]
        """
        # The given latent codes lie on Z- or W-space, while the given shifts lie on the W-space
        if shift is not None and len(shift.shape) == 4:
            shift = shift.squeeze(3).squeeze(2)
        if self.shift_in_w_space:
            #if latent_is_w:
                # Input latent code is in W-space
            # Input latent code is in Z-space -- get w code first
            #print(z.shape)
            w = map_for_stylegan(self.G.mapping(z, None, truncation_psi=0.5, truncation_cutoff=8))
            w = w if shift is None else w + shift
            #print(w.shape)
            w = w.unsqueeze(dim=1).repeat(1, self.G.num_ws, 1)
            #if w.shape[0] > 8:
            #    out = torch.cat([self.G.synthesis(w[i:min(i+8, w.shape[0])], noise_mode='const', force_fp32=True) for i in range(0,w.shape[0], 8)])
            #else:
            #    out = self.G.synthesis(w, noise_mode='const', force_fp32=True)
            out = self.G.synthesis(w, noise_mode='const', force_fp32=True)
            #print(self.G.training)
            #print(self.G.synthesis.training)
            out = torch.index_select(out, 1, torch.tensor([0]).cuda())
            #else:
                # Input latent code is in Z-space -- get w code first
                #w = self.G.get_latent(z)
                #return self.G([w if shift is None else w + shift], input_is_latent=True)[0]
        # The given latent codes and shift vectors lie on the Z-space
        else:
            out = self.G(z if shift is None else z + shift, None)
        return out


# Big chunks of this code are taken from GANLatentDiscovery
def find_latent_linear(run, G, out_dir, model_folder, logger: logging.Logger):
    latent_method = run["latent_method"]
    steps = latent_method["latent_steps"]
    deformator_type = latent_method["deformator_type"]
    model = run["model"]["train_method"]
    is_style_gan = model in ["StyleGAN2", "StyleGAN3"]
    directions = latent_method["directions"]
    shift_in_w_space = latent_method["shift_in_w_space"] if "shift_in_w_space" in latent_method else False

    if is_style_gan:
        G = StyleGAN2Wrapper(G,shift_in_w_space=shift_in_w_space)

    # Number of channels in the training images. For color images this is 3
    nc = 1
    # Size of z latent vector (i.e. size of generator input)
    nz = G.dim_z
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    G.eval()

    def feed_G(G, input, shift=None):
        return G(torch.squeeze(input, (2, 3)), shift) if is_style_gan else G(input)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    ## GAN Latent Space Code

    # Functions from ortho_utils.py from GANLatentDiscovery

    def torch_expm(A):
        n_A = A.shape[0]
        A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1, 2), keepdim=True))

        # Scaling step
        maxnorm = torch.tensor([5.371920351148152], dtype=A.dtype, device=A.device)
        zero = torch.tensor([0.0], dtype=A.dtype, device=A.device)
        n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
        A_scaled = A / 2.0 ** n_squarings
        n_squarings = n_squarings.flatten().type(torch.int64)

        # Pade 13 approximation
        U, V = torch_pade13(A_scaled)
        P = U + V
        Q = -U + V
        R = torch.linalg.solve(Q, P)

        # Unsquaring step
        res = [R]
        for i in range(int(n_squarings.max())):
            res.append(res[-1].matmul(res[-1]))
        R = torch.stack(res)
        expmA = R[n_squarings, torch.arange(n_A)]
        return expmA[0]

    def torch_log2(x):
        return torch.log(x) / np.log(2.0)

    def torch_pade13(A):
        b = torch.tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                          1187353796428800., 129060195264000., 10559470521600.,
                          670442572800., 33522128640., 1323241920., 40840800.,
                          960960., 16380., 182., 1.], dtype=A.dtype, device=A.device)

        ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
        A2 = torch.matmul(A, A)
        A4 = torch.matmul(A2, A2)
        A6 = torch.matmul(A4, A2)
        U = torch.matmul(A,
                         torch.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 +
                         b[3] * A2 + b[1] * ident)
        V = torch.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + \
            b[0] * ident
        return U, V

    # From latent_deformator.py from GANLatentDiscovery

    class DeformatorType(Enum):
        FC = 1
        LINEAR = 2
        ID = 3
        ORTHO = 4
        PROJECTIVE = 5
        RANDOM = 6

    class LatentDeformatorOrtho(nn.Module):
        def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024, random_init=True, bias=True):
            super(LatentDeformatorOrtho, self).__init__()
            self.shift_dim = shift_dim
            self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
            self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

            self.log_mat_half = nn.Parameter(
                (1.0 if random_init else 0.001) * torch.randn([self.input_dim, self.input_dim], device='cuda'), True)

        def forward(self, input):
            input = input.view([-1, self.input_dim])

            mat = torch_expm((self.log_mat_half - self.log_mat_half.transpose(0, 1)).unsqueeze(0))
            out = F.linear(input, mat)

            flat_shift_dim = np.product(self.shift_dim)
            if out.shape[1] < flat_shift_dim:
                padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]], device=out.device)
                out = torch.cat([out, padding], dim=1)
            elif out.shape[1] > flat_shift_dim:
                out = out[:, :flat_shift_dim]

            # handle spatial shifts
            try:
                out = out.view([-1] + self.shift_dim)
            except Exception:
                pass

            return out

    class LatentDeformatorFC(nn.Module):
        def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024, random_init=True, bias=True):
            super(LatentDeformatorFC, self).__init__()
            self.shift_dim = shift_dim
            self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
            self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

            self.fc1 = nn.Linear(self.input_dim, inner_dim)
            self.bn1 = nn.BatchNorm1d(inner_dim)
            self.act1 = nn.ELU()

            self.fc2 = nn.Linear(inner_dim, inner_dim)
            self.bn2 = nn.BatchNorm1d(inner_dim)
            self.act2 = nn.ELU()

            self.fc3 = nn.Linear(inner_dim, inner_dim)
            self.bn3 = nn.BatchNorm1d(inner_dim)
            self.act3 = nn.ELU()

            self.fc4 = nn.Linear(inner_dim, self.out_dim)

        def forward(self, input):
            input = input.view([-1, self.input_dim])
            x1 = self.fc1(input)
            x = self.act1(self.bn1(x1))
            x2 = self.fc2(x)
            x = self.act2(self.bn2(x2 + x1))
            x3 = self.fc3(x)
            x = self.act3(self.bn3(x3 + x2 + x1))
            out = self.fc4(x) + input

            flat_shift_dim = np.product(self.shift_dim)
            if out.shape[1] < flat_shift_dim:
                padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]], device=out.device)
                out = torch.cat([out, padding], dim=1)
            elif out.shape[1] > flat_shift_dim:
                out = out[:, :flat_shift_dim]

            # handle spatial shifts
            try:
                out = out.view([-1] + self.shift_dim)
            except Exception:
                pass

            return out

    class LatentDeformatorLinear(nn.Module):
        def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024, random_init=True, bias=True):
            super(LatentDeformatorLinear, self).__init__()
            self.shift_dim = shift_dim
            self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
            self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

            self.linear = nn.Linear(self.input_dim, self.out_dim, bias=bias)
            self.linear.weight.data = torch.zeros_like(self.linear.weight.data)

            min_dim = int(min(self.input_dim, self.out_dim))
            self.linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            if random_init:
                self.linear.weight.data = 0.1 * torch.randn_like(self.linear.weight.data)

        def forward(self, input):
            input = input.view([-1, self.input_dim])

            out = self.linear(input)

            flat_shift_dim = np.product(self.shift_dim)
            if out.shape[1] < flat_shift_dim:
                padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]], device=out.device)
                out = torch.cat([out, padding], dim=1)
            elif out.shape[1] > flat_shift_dim:
                out = out[:, :flat_shift_dim]

            # handle spatial shifts
            try:
                out = out.view([-1] + self.shift_dim)
            except Exception:
                pass

            return out

    class LatentDeformatorID(nn.Module):
        def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024, random_init=True, bias=True):
            super(LatentDeformatorID, self).__init__()
            self.shift_dim = shift_dim
            self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
            self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

        def forward(self, input):
            return input

    class LatentDeformatorProjective(nn.Module):
        def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024, random_init=True, bias=True):
            super(LatentDeformatorProjective, self).__init__()
            self.shift_dim = shift_dim
            self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
            self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

            self.linear = nn.Linear(self.input_dim, self.out_dim, bias=bias)
            self.linear.weight.data = torch.zeros_like(self.linear.weight.data)

            min_dim = int(min(self.input_dim, self.out_dim))
            self.linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            if random_init:
                self.linear.weight.data = 0.1 * torch.randn_like(self.linear.weight.data)

        def forward(self, input):
            input = input.view([-1, self.input_dim])

            input_norm = torch.norm(input, dim=1, keepdim=True)
            out = self.linear(input)
            out = (input_norm / torch.norm(out, dim=1, keepdim=True)) * out

            flat_shift_dim = np.product(self.shift_dim)
            if out.shape[1] < flat_shift_dim:
                padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]], device=out.device)
                out = torch.cat([out, padding], dim=1)
            elif out.shape[1] > flat_shift_dim:
                out = out[:, :flat_shift_dim]

            # handle spatial shifts
            try:
                out = out.view([-1] + self.shift_dim)
            except Exception:
                pass

            return out

    class LatentDeformatorRandom(nn.Module):
        def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024, random_init=True, bias=True):
            super(LatentDeformatorRandom, self).__init__()
            self.shift_dim = shift_dim
            self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
            self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

            self.linear = torch.empty([self.out_dim, self.input_dim])
            nn.init.orthogonal_(self.linear)

        def forward(self, input):
            input = input.view([-1, self.input_dim])

            self.linear = self.linear.to(input.device)
            out = F.linear(input, self.linear)

            flat_shift_dim = np.product(self.shift_dim)
            if out.shape[1] < flat_shift_dim:
                padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]], device=out.device)
                out = torch.cat([out, padding], dim=1)
            elif out.shape[1] > flat_shift_dim:
                out = out[:, :flat_shift_dim]

            # handle spatial shifts
            try:
                out = out.view([-1] + self.shift_dim)
            except Exception:
                pass

            return out

    def get_deformator(deformator_type, shift_dim, input_dim):
        if deformator_type == 'FC':
            return LatentDeformatorFC(shift_dim=shift_dim, input_dim=input_dim).cuda()
        if deformator_type == 'LINEAR':
            return LatentDeformatorLinear(shift_dim=shift_dim, input_dim=input_dim).cuda()
        if deformator_type == 'ID':
            return LatentDeformatorID(shift_dim=shift_dim, input_dim=input_dim).cuda()
        if deformator_type == 'ORTHO':
            return LatentDeformatorOrtho(shift_dim=shift_dim, input_dim=input_dim).cuda()
        if deformator_type == 'PROJECTIVE':
            return LatentDeformatorProjective(shift_dim=shift_dim, input_dim=input_dim).cuda()
        if deformator_type == 'RANDOM':
            return LatentDeformatorRandom(shift_dim=shift_dim, input_dim=input_dim).cuda()
        return False

    # From latent_shift_predictor.py from GANLatentDiscovery
    # Note: there is also a class based on LeNet and not ResNet

    def save_hook(module, input, output):
        setattr(module, 'output', output)

    class LatentShiftPredictor(nn.Module):
        def __init__(self, dim, downsample=None):
            super(LatentShiftPredictor, self).__init__()
            self.features_extractor = resnet18(weights=None)
            # number of color channels times 2, since two images are concatenated
            self.features_extractor.conv1 = nn.Conv2d(nc * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                      bias=False)
            nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')

            self.features = self.features_extractor.avgpool
            self.features.register_forward_hook(save_hook)
            self.downsample = downsample

            # half dimension as we expect the model to be symmetric
            self.type_estimator = nn.Linear(512, np.product(dim))
            self.shift_estimator = nn.Linear(512, 1)

        def forward(self, x1, x2):
            batch_size = x1.shape[0]
            if self.downsample is not None:
                x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)
            self.features_extractor(torch.cat([x1, x2], dim=1))
            features = self.features.output.view([batch_size, -1])

            logits = self.type_estimator(features)
            shift = self.shift_estimator(features)

            return logits, shift.squeeze()

    # From utils.py from GANLatentDiscovery

    def make_noise(batch, dim, truncation=None):
        return torch.randn(batch, dim, 1, 1, device=device)
        # if isinstance(dim, int):
        #    dim = [dim]
        # if truncation is None or truncation == 1.0:
        #    return torch.randn([batch] + dim)
        # else:
        #    return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)

    def one_hot(dims, value, indx):
        vec = torch.zeros(dims)
        vec[indx] = value
        return vec

    # From different files from GANLatentDiscovery for visualization.py

    class VerbosityLevel(Enum):
        SILENT = 0,
        JUPYTER = 1,
        CONSOLE = 2

    def numerical_order(files):
        return sorted(files, key=lambda x: int(x.split('.')[0]))

    def make_verbose():
        return VerbosityLevel.CONSOLE

    def wrap_with_tqdm(it, verbosity=make_verbose(), **kwargs):
        if verbosity == VerbosityLevel.SILENT or verbosity == False:
            return it
        elif verbosity == VerbosityLevel.JUPYTER:
            return tqdm_notebook(it, **kwargs)
        elif verbosity == VerbosityLevel.CONSOLE:
            return tqdm(it, **kwargs)

    def _filename(path):
        return os.path.basename(path).split('.')[0]

    class UnannotatedDataset(Dataset):
        def __init__(self, root_dir, sorted=False,
                     transform=transforms.Compose(
                         [
                             transforms.ToTensor(),
                             lambda x: 2 * x - 1
                         ])):
            self.img_files = []
            for root, _, files in os.walk(root_dir):
                for file in numerical_order(files) if sorted else files:
                    if UnannotatedDataset.file_is_img(file):
                        self.img_files.append(os.path.join(root, file))
            self.transform = transform

        @staticmethod
        def file_is_img(name):
            extension = os.path.basename(name).split('.')[-1]
            return extension in ['jpg', 'jpeg', 'png']

        def align_names(self, target_names):
            new_img_files = []
            img_files_names_dict = {_filename(f): f for f in self.img_files}
            for name in target_names:
                try:
                    new_img_files.append(img_files_names_dict[_filename(name)])
                except KeyError:
                    logger.error('names mismatch: absent {}'.format(_filename(name)))
            self.img_files = new_img_files

        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, item):
            img = Image.open(self.img_files[item])
            if self.transform is not None:
                return self.transform(img)
            else:
                return img

    class RGBDataset(Dataset):
        def __init__(self, source_dataset):
            super(RGBDataset, self).__init__()
            self.source = source_dataset

        def __len__(self):
            return len(self.source)

        def __getitem__(self, index):
            out = self.source
            if out.shape[0] == 1:
                out = out.repeat([3, 1, 1])
            return out

    # From visualization.py from GANLatentDiscovery

    def fig_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return Image.open(buf)

    @torch.no_grad()
    def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False):
        shifted_images = []
        for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
            if deformator is not None:
                latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).cuda())
            else:
                latent_shift = one_hot(nz, shift, dim).cuda()
            latent_shift = latent_shift.unsqueeze(2).unsqueeze(3)
            shifted_image = feed_G(G, z + latent_shift).cpu()[0]

            # print(shifted_image.shape)

            if shift == 0.0 and with_central_border:
                shifted_image = add_border(shifted_image)

            shifted_images.append(shifted_image)
        return shifted_images

    def add_border(tensor):
        border = 3
        for ch in range(tensor.shape[0]):
            color = 1.0 if ch == 0 else -1
            tensor[ch, :border, :] = color
            tensor[ch, -border:, ] = color
            tensor[ch, :, :border] = color
            tensor[ch, :, -border:] = color
        return tensor

    @torch.no_grad()
    def make_interpolation_chart(G, deformator=None, z=None,
                                 shifts_r=10.0, shifts_count=5,
                                 dims=None, dims_count=10, texts=None, **kwargs):
        with_deformation = deformator is not None
        if with_deformation:
            deformator_is_training = deformator.training
            deformator.eval()
        z = z if z is not None else make_noise(1, nz)

        if with_deformation:
            original_img = feed_G(G, z).cpu()
        else:
            original_img = feed_G(G, z).cpu()
        imgs = []
        if dims is None:
            dims = range(dims_count)
        for i in dims:
            imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator))

        rows_count = len(imgs) + 1
        fig, axs = plt.subplots(rows_count, **kwargs)

        axs[0].axis('off')
        axs[0].imshow(to_image(original_img, True))

        if texts is None:
            texts = dims
        for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
            ax.axis('off')
            plt.subplots_adjust(left=0.5)
            ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
            ax.text(-20, 21, str(text), fontsize=10)

        if deformator is not None and deformator_is_training:
            deformator.train()

        return fig

    @torch.no_grad()
    def inspect_all_directions(G, deformator, out_dir, zs=None, num_z=3, shifts_r=8.0):
        os.makedirs(out_dir, exist_ok=True)

        step = 20
        max_dim = nz
        zs = zs if zs is not None else make_noise(num_z, nz)
        shifts_count = zs.shape[0]

        for start in range(0, max_dim - 1, step):
            imgs = []
            dims = range(start, min(start + step, max_dim))
            for z in zs:
                z = z.unsqueeze(0)
                fig = make_interpolation_chart(
                    G, deformator=deformator, z=z,
                    shifts_count=shifts_count, dims=dims, shifts_r=shifts_r,
                    dpi=250, figsize=(int(shifts_count * 4.0), int(0.5 * step) + 2))
                fig.canvas.draw()
                plt.close(fig)
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # crop borders
                nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
                img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
                imgs.append(img)

            out_file = os.path.join(out_dir, '{}_{}.jpg'.format(dims[0], dims[-1]))
            logger.debug('saving chart to {}'.format(out_file))
            Image.fromarray(np.hstack(imgs)).save(out_file)

    def to_image(tensor, adaptive=False):
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        if adaptive:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
        else:
            tensor = (tensor + 1) / 2
            tensor.clamp(0, 1)
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))

    # From trainer.py from GANLatentDiscovery

    class DataParallelPassthrough(nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super(DataParallelPassthrough, self).__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    class MeanTracker(object):
        def __init__(self, name):
            self.values = []
            self.name = name

        def add(self, val):
            self.values.append(float(val))

        def mean(self):
            return np.mean(self.values)

        def flush(self):
            mean = self.mean()
            self.values = []
            return self.name, mean

    class ShiftDistribution(Enum):
        NORMAL = 0,
        UNIFORM = 1,

    class Params(object):
        def __init__(self, **kwargs):
            self.shift_scale = 6.0
            self.min_shift = 0.5
            self.shift_distribution = ShiftDistribution.UNIFORM

            self.deformator_lr = 0.0001
            self.shift_predictor_lr = 0.0001
            self.n_steps = 20000  # 20_000 seems good for fast test runs, 200_000 for real runs
            self.batch_size = 32

            self.directions_count = None
            self.max_latent_dim = None

            self.label_weight = 1.0
            self.shift_weight = 0.25

            self.steps_per_log = self.n_steps / 100
            self.steps_per_save = self.n_steps / 10
            self.steps_per_img_log = self.n_steps / 10
            self.steps_per_backup = self.n_steps / 5

            self.truncation = None

        def add_param(self, key, val):
            self.__dict__[key] = val

        def setup(self, n_steps):
            self.n_steps = n_steps
            self.steps_per_log = self.n_steps / 100
            self.steps_per_save = self.n_steps / 10
            self.steps_per_img_log = self.n_steps / 10
            self.steps_per_backup = self.n_steps / 5

    def setup_params(deformator, n_steps):
        # Init params
        params = Params()
        # update dims with respect to the deformator if some of params are None
        params.directions_count = int(deformator.input_dim)
        # params.directions_count = run["latent_directions"]
        params.max_latent_dim = int(deformator.out_dim)
        params.setup(n_steps=n_steps)
        return params

    class Trainer(object):
        def __init__(self, params=Params(), out_dir='', verbose=False):
            if verbose:
                logger.debug('Trainer inited with:\n{}'.format(str(params.__dict__)))
            self.p = params
            self.log_dir = os.path.join(out_dir, 'logs')
            os.makedirs(self.log_dir, exist_ok=True)
            self.cross_entropy = nn.CrossEntropyLoss()

            tb_dir = os.path.join(out_dir, 'tensorboard')
            self.models_dir = os.path.join(out_dir, 'models')
            self.images_dir = os.path.join(self.log_dir, 'images')
            os.makedirs(tb_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.images_dir, exist_ok=True)

            self.checkpoint = os.path.join(out_dir, 'checkpoint.pt')
            self.writer = SummaryWriter(tb_dir)
            self.out_json = os.path.join(self.log_dir, 'stat.json')
            self.fixed_test_noise = None

        def make_shifts(self, latent_dim):
            target_indices = torch.randint(0, self.p.directions_count, [self.p.batch_size, 1, 1], device='cuda')
            if self.p.shift_distribution == ShiftDistribution.NORMAL:
                shifts = torch.randn(target_indices.shape, device='cuda')
            elif self.p.shift_distribution == ShiftDistribution.UNIFORM:
                shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

            shifts = self.p.shift_scale * shifts
            shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
            shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift

            try:
                latent_dim[0]
                latent_dim = list(latent_dim)
            except Exception:
                latent_dim = [latent_dim]

            z_shift = torch.zeros([self.p.batch_size] + latent_dim, device='cuda')
            for i, (index, val) in enumerate(zip(target_indices, shifts)):
                z_shift[i][index] += val

            z_shift = z_shift.unsqueeze(2).unsqueeze(3)

            return target_indices, shifts, z_shift

        def log_train(self, step, should_print=True, stats=()):
            if should_print:
                out_text = '{}% [step {}]'.format(int(100 * step / self.p.n_steps), step)
                for named_value in stats:
                    out_text += (' | {}: {:.2f}'.format(*named_value))
                logger.debug(out_text)
            for named_value in stats:
                self.writer.add_scalar(named_value[0], named_value[1], step)

            with open(self.out_json, 'w') as out:
                stat_dict = {named_value[0]: named_value[1] for named_value in stats}
                json.dump(stat_dict, out)

        def log_interpolation(self, G, deformator, step):
            noise = make_noise(1, nz, self.p.truncation)
            if self.fixed_test_noise is None:
                self.fixed_test_noise = noise.clone()
            for z, prefix in zip([noise, self.fixed_test_noise], ['rand', 'fixed']):
                fig = make_interpolation_chart(G, deformator, z=z, shifts_r=3 * self.p.shift_scale, shifts_count=3,
                                               dims_count=deformator.input_dim, dpi=500)

                self.writer.add_figure('{}_deformed_interpolation'.format(prefix), fig, step)
                fig_to_image(fig).convert("RGB").save(os.path.join(self.images_dir, '{}_{}.jpg'.format(prefix, step)))

        def start_from_checkpoint(self, deformator, shift_predictor):
            step = 0
            if os.path.isfile(self.checkpoint):
                state_dict = torch.load(self.checkpoint)
                step = state_dict['step']
                deformator.load_state_dict(state_dict['deformator'])
                shift_predictor.load_state_dict(state_dict['shift_predictor'])
                logger.debug('starting from step {}'.format(step))
            return step

        def save_checkpoint(self, deformator, shift_predictor, step):
            state_dict = {
                'step': step,
                'deformator': deformator.state_dict(),
                'shift_predictor': shift_predictor.state_dict(),
            }
            torch.save(state_dict, self.checkpoint)

        def save_models(self, deformator, shift_predictor, step):
            torch.save(deformator.state_dict(),
                       os.path.join(self.models_dir, 'deformator_{}.pt'.format(step)))
            torch.save(shift_predictor.state_dict(),
                       os.path.join(self.models_dir, 'shift_predictor_{}.pt'.format(step)))

        def log_accuracy(self, G, deformator, shift_predictor, step):
            deformator.eval()
            shift_predictor.eval()

            accuracy = validate_classifier(G, deformator, shift_predictor, trainer=self)
            self.writer.add_scalar('accuracy', accuracy.item(), step)

            deformator.train()
            shift_predictor.train()
            return accuracy

        def log(self, G, deformator, shift_predictor, step, avgs):
            if step % self.p.steps_per_log == 0:
                self.log_train(step, True, [avg.flush() for avg in avgs])

            if step % self.p.steps_per_img_log == 0:
                self.log_interpolation(G, deformator, step)

            if step % self.p.steps_per_backup == 0 and step > 0:
                self.save_checkpoint(deformator, shift_predictor, step)
                accuracy = self.log_accuracy(G, deformator, shift_predictor, step)
                logger.debug('Step {} accuracy: {:.3}'.format(step, accuracy.item()))

            if step % self.p.steps_per_save == 0 and step > 0:
                self.save_models(deformator, shift_predictor, step)

        def train(self, G, deformator, shift_predictor, multi_gpu=False):
            G.cuda().eval()
            deformator.cuda()
            deformator.train()
            deformator.apply(weights_init)
            shift_predictor.cuda().train()

            # should_gen_classes = is_conditional(G)
            if multi_gpu:
                G = DataParallelPassthrough(G)

            deformator_opt = torch.optim.Adam(deformator.parameters(), lr=self.p.deformator_lr) \
                if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] else None
            shift_predictor_opt = torch.optim.Adam(
                shift_predictor.parameters(), lr=self.p.shift_predictor_lr)

            avgs = MeanTracker('percent'), MeanTracker('loss'), MeanTracker('direction_loss'), \
                   MeanTracker('shift_loss')
            avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss = avgs

            recovered_step = self.start_from_checkpoint(deformator, shift_predictor)
            for step in range(recovered_step, self.p.n_steps, 1):
                G.zero_grad()
                deformator.zero_grad()
                shift_predictor.zero_grad()

                z = make_noise(self.p.batch_size, nz, self.p.truncation)
                target_indices, shifts, basis_shift = self.make_shifts(deformator.input_dim)

                # if should_gen_classes:
                #    classes = G.mixed_classes(z.shape[0])

                # Deformation
                shift = deformator(basis_shift)
                shift = shift.unsqueeze(2).unsqueeze(3)
                # if should_gen_classes:
                #    imgs = G(z, classes)
                #    imgs_shifted = G.gen_shifted(z, shift, classes)
                # else:
                imgs = feed_G(G, z)
                # imgs_shifted = G.gen_shifted(z, shift)
                imgs_shifted = feed_G(G, z + shift)

                # print("Prints incoming:")
                # print(imgs.shape)
                # print(imgs_shifted.shape)

                logits, shift_prediction = shift_predictor(imgs, imgs_shifted)

                logits = logits.squeeze()
                target_indices = target_indices.squeeze()

                logit_loss = self.p.label_weight * self.cross_entropy(logits, target_indices)
                shift_loss = self.p.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

                # total loss
                loss = logit_loss + shift_loss
                loss.backward()

                if deformator_opt is not None:
                    deformator_opt.step()
                shift_predictor_opt.step()

                # update statistics trackers
                avg_correct_percent.add(torch.mean(
                    (torch.argmax(logits, dim=1) == target_indices).to(torch.float32)).detach())
                avg_loss.add(loss.item())
                avg_label_loss.add(logit_loss.item())
                avg_shift_loss.add(shift_loss)

                self.log(G, deformator, shift_predictor, step, avgs)

    @torch.no_grad()
    def validate_classifier(G, deformator, shift_predictor, params_dict=None, trainer=None):
        n_steps = 100
        # if trainer is None:
        #    trainer = Trainer(params=Params(**params_dict), verbose=False)

        percents = torch.empty([n_steps])
        for step in range(n_steps):
            z = make_noise(trainer.p.batch_size, nz, trainer.p.truncation)
            target_indices, shifts, basis_shift = trainer.make_shifts(deformator.input_dim)

            imgs = feed_G(G, z)
            shift = deformator(basis_shift)
            shift = shift.unsqueeze(2).unsqueeze(3)
            imgs_shifted = feed_G(G, z + shift)

            logits, _ = shift_predictor(imgs, imgs_shifted)
            percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

        return percents.mean()

    # From run_train.py from GANLatentDiscovery

    def save_results_charts(G, deformator, params, out_dir):
        deformator.eval()
        G.eval()
        z = make_noise(3, nz, params.truncation)
        inspect_all_directions(G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(params.shift_scale))), zs=z,
                               shifts_r=params.shift_scale)
        inspect_all_directions(G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(3 * params.shift_scale))),
                               zs=z,
                               shifts_r=3 * params.shift_scale)

    def train_gan_latent_space(G, deformator_type, out_dir, n_steps):
        # Choose which Latent Deformator to use:
        deformator = get_deformator(deformator_type=deformator_type, shift_dim=nz, input_dim=directions)
        if not deformator:
            logger.error('Deformator type does not exist')
            return

        shift_predictor = LatentShiftPredictor(deformator.input_dim).cuda()

        params = setup_params(deformator, n_steps)

        trainer = Trainer(params, out_dir=out_dir)
        trainer.train(G, deformator, shift_predictor, multi_gpu=False)

        #save_results_charts(G, deformator, params, trainer.log_dir)
        return deformator

    def save_drusen_image(t, i, path_dir, path_i, step_i, inter_name=""):
        if inter_name:
            folder = os.path.join(path_dir, f"{inter_name}")
        else:
            folder = os.path.join(path_dir, f"Druse_{i:03d}")
        if path_i < 0:
            fname = os.path.join(folder, "original_image.jpg")
        else:
            fname = os.path.join(folder, "paths_images", f"path_{path_i:03d}", f"{step_i:06d}.jpg")
        im = to_image(t[i], True)
        im.save(fname)

    def generate_path_images(G, folder, deformator):
        path_dir = os.path.join(folder, "paths")
        if os.path.exists(path_dir):  # skip if this already exists
            return
        os.makedirs(path_dir)

        # Sample G for drusen and generate all relevant paths and save the drusen
        num_drusen = latent_method["num_samples"]
        num_paths = directions
        drusen_folders = [os.path.join(path_dir, f"Druse_{i:03d}") for i in range(num_drusen)]
        for df in drusen_folders:
            os.makedirs(df)
            paths_dir = os.path.join(df, "paths_images")
            os.makedirs(paths_dir)
            for i in range(num_paths):
                os.makedirs(os.path.join(paths_dir, f"path_{i:03d}"))

        z = make_noise(num_drusen, nz)
        drusen = feed_G(G, z)
        for i in range(num_drusen):
            save_drusen_image(drusen, i, path_dir, -1, -1)
            torch.save(z[i].cpu(), os.path.join(path_dir, f"Druse_{i:03d}", 'latent_code_image.pt'))

        # Load deformator
        deformator.eval()

        # Setup params, n_steps does not matter here
        params = setup_params(deformator, 100000)
        shifts_r = 2 * params.shift_scale
        shifts_count = 16

        latent_codes = torch.zeros((num_drusen, num_paths, 33, nz))

        # Create images:
        for dim in range(num_paths):
            i = 0
            for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
                latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).cuda())
                latent_shift = latent_shift.unsqueeze(2).unsqueeze(3)
                z_shifted = z + latent_shift

                result = feed_G(G, z, latent_shift).cpu()
                for j in range(num_drusen):
                    latent_codes[j, dim, i] = z_shifted[j].squeeze()
                    save_drusen_image(result, j, path_dir, dim, i)
                i += 1
        for i in range(num_drusen):
            torch.save(latent_codes[i], os.path.join(path_dir, f"Druse_{i:03d}", 'paths_latent_codes.pt'))

        return

    def generate_interesting_path_images(G, folder, model_dir):
        path_dir = os.path.join(folder, "interesting_paths")
        if os.path.exists(path_dir):  # delete if this already exists
            shutil.rmtree(path_dir, ignore_errors=True)
        os.makedirs(path_dir, exist_ok=True)

        folders = glob.glob(os.path.join(model_dir, "interesting_drusen", "*"))
        drusen = []
        latent_codes = []
        for f in folders:
            df = glob.glob(os.path.join(f, "*.jpg"))
            nums = [int(os.path.split(x)[1].split('.')[0].split('_')[-1]) for x in df]
            for n in nums:
                drusen.append((n, os.path.split(f)[1]))
                latent_codes.append(torch.load(os.path.join(f, f"latent_code_{n}.pt")).unsqueeze_(0))
        z = torch.cat(latent_codes).cuda()
        z = z.unsqueeze(2).unsqueeze(3)

        num_drusen = len(drusen)
        num_paths = directions
        drusen_folders = [os.path.join(path_dir, f"{drusen[i][1]}_{drusen[i][0]}") for i in range(num_drusen)]
        for df in drusen_folders:
            os.makedirs(df)
            paths_dir = os.path.join(df, "paths_images")
            os.makedirs(paths_dir)
            for i in range(num_paths):
                os.makedirs(os.path.join(paths_dir, f"path_{i:03d}"))

        for i in range(num_drusen):
            shutil.copy(os.path.join(model_dir, "interesting_drusen", drusen[i][1], f"image_{drusen[i][0]}.jpg"),
                        os.path.join(path_dir, f"{drusen[i][1]}_{drusen[i][0]}", "original_image.jpg"))

        # Load deformator
        deformator = get_deformator(deformator_type=deformator_type, shift_dim=nz, input_dim=directions)
        deformator.cuda()
        deformator_paths = glob.glob(os.path.join(folder, "models", "deformator_*.pt"))
        nums = np.asarray([int(os.path.split(x)[-1].split("_")[1].split(".")[0]) for x in deformator_paths],
                          dtype=int)
        deformator_path = deformator_paths[np.argmax(nums)]
        deformator.load_state_dict(torch.load(deformator_path))
        deformator.eval()

        # Setup params, n_steps does not matter here
        params = setup_params(deformator, 100000)
        shifts_r = 2 * params.shift_scale
        shifts_count = 16

        latent_codes = torch.zeros((num_drusen, num_paths, 33, nz))

        # Create images:
        for dim in range(num_paths):
            i = 0
            for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
                latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).cuda())
                latent_shift = latent_shift.unsqueeze(2).unsqueeze(3)
                z_shifted = z + latent_shift

                result = feed_G(G, z, latent_shift).cpu()
                for j in range(num_drusen):
                    latent_codes[j, dim, i] = z_shifted[j].squeeze()
                    save_drusen_image(result, j, path_dir, dim, i, inter_name=f"{drusen[j][1]}_{drusen[j][0]}")
                i += 1
        for i in range(num_drusen):
            torch.save(latent_codes[i], os.path.join(path_dir, f"{drusen[i][1]}_{drusen[i][0]}", 'paths_latent_codes.pt'))

        return

    if "paths" in run:
        generate_interesting_path_images(G, out_dir, model_folder)
        generate_gifs_linear_int(out_dir, run["paths"])
    else:
        d = train_gan_latent_space(G, deformator_type, out_dir=out_dir, n_steps=steps)
        generate_path_images(G, out_dir, d)
        generate_gifs_linear(out_dir)

    return


def find_latent_warp(run, out_dir, model_folder, logger: logging.Logger):
    if os.path.exists(os.path.join(out_dir, "experiments")):
        logger.info("exists")
    warp_folder = "WarpedGANSpace"
    latent_method = run["latent_method"]
    latent_steps = latent_method["latent_steps"]
    model = run["model"]["train_method"]
    is_stylegan = model in ["StyleGAN2", "StyleGAN3"]
    directions = latent_method["directions"]
    shift_in_w_space = latent_method["shift_in_w_space"] if "shift_in_w_space" in latent_method else False
    show_int_paths = "paths" in run

    # Delete old data
    shutil.rmtree("experiments", ignore_errors=True)

    # Copy model code and generator
    warp_model_folder = os.path.join(warp_folder, "models", "pretrained", "generators")
    if is_stylegan:
        shutil.copy(run["model"]["path_to_g"], os.path.join(warp_model_folder, f"{model}.pkl"))
    else:
        shutil.copy(os.path.join(model_folder, "generator.pt"), os.path.join(warp_model_folder, "generator.pt"))
        shutil.copy(os.path.join(model_folder, "model.py"), os.path.join(warp_model_folder, "model.py"))

    # Execute discovery of directions
    num_support_sets = directions
    num_support_dipoles = 5
    eps_min = 0.15
    eps_max = 0.25
    gan_type = "GAN128"
    if is_stylegan:
        gan_type = model
    cmd = ["python",
           os.path.join(warp_folder, "train.py"),
           f"--gan-type={gan_type}",
           "--reconstructor-type=LeNet",
           "--learn-gammas",
           f"--num-support-sets={num_support_sets}",
           f"--num-support-dipoles={num_support_dipoles}",
           f"--min-shift-magnitude={eps_min}",
           f"--max-shift-magnitude={eps_max}",
           "--batch-size=8",
           f"--max-iter={latent_steps}"]
    if is_stylegan:
        if shift_in_w_space:
            cmd += ["--shift-in-w-space"]
        cmd += ["--stylegan2-resolution=128"]

    if not show_int_paths:
        subprocess.run(cmd)
        logger.info("Finished discovering directions")

    # Create sample images
    num_samples = latent_method["num_samples"]
    cmd = ["python",
           os.path.join(warp_folder, "sample_gan.py"),
           f"--gan-type={gan_type}",
           f"--num-samples={num_samples}"]
    if is_stylegan:
        if shift_in_w_space:
            cmd += ["--shift-in-w-space"]
        cmd += ["--stylegan2-resolution=128"]
    if not show_int_paths:
        subprocess.run(cmd)
        logger.info("Finished creating samples")

    # Show paths
    exp_folder = f"{gan_type}"
    if is_stylegan:
        exp_folder += "-128"
        exp_folder += "-W" if shift_in_w_space else "-Z"
    exp_folder += f"-LeNet-K{num_support_sets}-D{num_support_dipoles}-LearnGammas-eps{eps_min}_{eps_max}"
    exp_folder = os.path.join("experiments", "complete", exp_folder)

    pool_folder = f"{gan_type}_{num_samples}"
    if show_int_paths:
        path_to_pool = os.path.join(out_dir, "experiments", "latent_codes", gan_type, pool_folder)
        shutil.rmtree(path_to_pool, ignore_errors=True)
        os.makedirs(path_to_pool, exist_ok=True)

        # do the generation of how the folder should look like: folder for each druse, with image.jpg, with latent_code.pt
        model_int_folder = os.path.join(model_folder, "interesting_drusen")
        for folder in glob.glob(os.path.join(model_int_folder, "*")):
            name = os.path.split(folder)[1]
            for druse in glob.glob(os.path.join(folder, "*.jpg")):
                drusen_id = int(os.path.split(druse)[1].split('_')[-1].split('.')[0])
                path_to_druse = os.path.join(path_to_pool, f"{name}_{drusen_id}")
                os.makedirs(path_to_druse, exist_ok=True)
                shutil.copy(druse, os.path.join(path_to_druse, "image.jpg"))
                d = torch.load(os.path.join(folder, f"latent_code_{drusen_id}.pt")).unsqueeze(0)
                torch.save(d, os.path.join(path_to_druse, "latent_code.pt"))

        shutil.move(os.path.join(out_dir, "experiments"), os.getcwd())
    cmd = ["python",
           os.path.join(warp_folder, "traverse_latent_space.py"),
           f"--exp={exp_folder}",
           f"--pool={pool_folder}"]
    subprocess.run(cmd)
    logger.info("Finished showing paths")

    # Move data
    shutil.move("experiments", out_dir)
    logger.info("Finished moving data")

    # Create better plots
    if show_int_paths:
        generate_gifs_warp_int(out_dir, run["paths"])
    else:
        generate_gifs_warp(out_dir)
    logger.info("Finished creating plots")


def find_latent_pde(run, out_dir, model_folder, logger: logging.Logger):
    if os.path.exists(os.path.join(out_dir, "experiments")):
        logger.info("exists")
    pde_folder = "PDETraversal"
    latent_method = run["latent_method"]
    latent_steps = latent_method["latent_steps"]
    model = run["model"]["train_method"]
    is_stylegan = model in ["StyleGAN2", "StyleGAN3"]
    directions = latent_method["directions"]
    shift_in_w_space = latent_method["shift_in_w_space"] if "shift_in_w_space" in latent_method else False
    show_int_paths = "paths" in run

    # Delete old data
    shutil.rmtree("experiments", ignore_errors=True)

    # Copy model code and generator
    pde_model_folder = os.path.join(pde_folder, "models", "pretrained", "generators")
    if is_stylegan:
        shutil.copy(run["model"]["path_to_g"], os.path.join(pde_model_folder, f"{model}.pkl"))
    else:
        shutil.copy(os.path.join(model_folder, "generator.pt"), os.path.join(pde_model_folder, "generator.pt"))
        shutil.copy(os.path.join(model_folder, "model.py"), os.path.join(pde_model_folder, "model.py"))

    # Execute discovery of directions
    num_support_sets = directions
    num_support_timesteps = 10
    gan_type = "GAN128"
    if is_stylegan:
        gan_type = model
    cmd = ["python",
           os.path.join(pde_folder, "train.py"),
           f"--gan-type={gan_type}",
           "--reconstructor-type=LeNet",
           f"--num-support-sets={num_support_sets}",
           f"--num-support-timesteps={num_support_timesteps}",
           "--batch-size=8",
           f"--max-iter={latent_steps}",
           f"--ckp-freq={latent_steps//10}"]
    if is_stylegan:
        if shift_in_w_space:
            cmd += ["--shift-in-w-space"]
        cmd += ["--stylegan2-resolution=128"]

    if not show_int_paths:
        subprocess.run(cmd)
        logger.info("Finished discovering directions")

    # Create sample images
    num_samples = latent_method["num_samples"]
    cmd = ["python",
           os.path.join(pde_folder, "sample_gan.py"),
           f"--gan-type={gan_type}",
           f"--num-samples={num_samples}"]
    if is_stylegan:
        if shift_in_w_space:
            cmd += ["--shift-in-w-space"]
        cmd += ["--stylegan2-resolution=128"]
    if not show_int_paths:
        subprocess.run(cmd)
        logger.info("Finished creating samples")

    # Show paths
    exp_folder = f"{gan_type}"
    if is_stylegan:
        exp_folder += "-128"
        exp_folder += "-W" if shift_in_w_space else "-Z"
    exp_folder += f"-LeNet-K{num_support_sets}-D{num_support_timesteps}"
    exp_folder = os.path.join("experiments", "complete", exp_folder)

    pool_folder = f"{gan_type}_{num_samples}"
    if show_int_paths:
        path_to_pool = os.path.join(out_dir, "experiments", "latent_codes", gan_type, pool_folder)
        shutil.rmtree(path_to_pool, ignore_errors=True)
        os.makedirs(path_to_pool, exist_ok=True)

        # do the generation of how the folder should look like: folder for each druse, with image.jpg, with latent_code.pt
        model_int_folder = os.path.join(model_folder, "interesting_drusen")
        for folder in glob.glob(os.path.join(model_int_folder, "*")):
            name = os.path.split(folder)[1]
            for druse in glob.glob(os.path.join(folder, "*.jpg")):
                drusen_id = int(os.path.split(druse)[1].split('_')[-1].split('.')[0])
                path_to_druse = os.path.join(path_to_pool, f"{name}_{drusen_id}")
                os.makedirs(path_to_druse, exist_ok=True)
                shutil.copy(druse, os.path.join(path_to_druse, "image.jpg"))
                d = torch.load(os.path.join(folder, f"latent_code_{drusen_id}.pt")).unsqueeze(0)
                torch.save(d, os.path.join(path_to_druse, "latent_code.pt"))

        shutil.move(os.path.join(out_dir, "experiments"), os.getcwd())
    cmd = ["python",
           os.path.join(pde_folder, "traverse_latent_space_twosides.py"),
           f"--exp={exp_folder}",
           f"--pool={pool_folder}",
           f"--shift-steps=32"]
    subprocess.run(cmd)
    logger.info("Finished showing paths")

    # Move data
    shutil.move("experiments", out_dir)
    logger.info("Finished moving data")

    # Create better plots
    if show_int_paths:
        generate_gifs_warp_int(out_dir, run["paths"])
    else:
        generate_gifs_warp(out_dir)
    logger.info("Finished creating plots")


# Find latent directions in G
# G the Generator, already loaded
# out_dir the path where to save everything to
def find_latent(run, G, out_dir, logger: logging.Logger):
    if "latent_method" not in run or run["latent_method"] is None:  # don't search for directions
        return
    latent_method = run["latent_method"]
    method = latent_method["method"]
    steps = latent_method["latent_steps"]
    deformator_type = latent_method["deformator_type"] if method == "linear" else None
    shift_in_w_space = latent_method["shift_in_w_space"] if "shift_in_w_space" in latent_method else False

    model_folder = out_dir
    out_dir = f"{out_dir}_{method}{steps}"
    if shift_in_w_space:
        out_dir += "_w"
    if method == "linear":
        out_dir += f"_{deformator_type.lower()}"
    elif method == "warp" or method == "pde":
        out_dir += f"_{latent_method['directions']}"

    already_trained = os.path.exists(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    # Init the logger
    file_handler = logging.FileHandler(os.path.join(out_dir, 'mainlog.txt'))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def close_logger():
        file_handler.close()
        logger.removeHandler(file_handler)

    logger.info(f"Start discovering directions for {out_dir} with method {method}, for {steps} steps.")
    if method == "linear":
        logger.info(f"With {deformator_type} deformator type.")

    if already_trained:
        logger.info("Already discovered directions")
    if not already_trained or "paths" in run:
        if method == "linear":
            find_latent_linear(run, G, out_dir, model_folder, logger)
        elif method == "warp":
            find_latent_warp(run, out_dir, model_folder, logger)
        elif method == "pde":
            find_latent_pde(run, out_dir, model_folder, logger)
    logger.info("Finished discovering directions")
    close_logger()
    return
