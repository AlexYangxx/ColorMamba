import cv2
from torch import nn
import numpy as np
import functools
import torch
from models.trans import CrissCrossAttention
from models.commom import *
import torch.nn.functional as F
from models.gen_net import UNetGenerator
from models.mambair_arch import MambaIR

class dsuGenerator_hsv(nn.Module):
    """Create a Dense scale Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(dsuGenerator_hsv, self).__init__()
        # construct unet structure
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # add the outermost layer
        self.coder_1 = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        )

        # gradually reduce the number of filters from ngf * 8 to ngf
        self.coder_2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
        )
        self.coder_3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
        )
        self.coder_4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        # add intermediate layers with ngf * 8 filters
        self.coder_5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )
        self.coder_6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )
        self.coder_7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        # add the innermost layer
        self.innermost_8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        self.decoder_7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
        )
        self.decoder_6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
        )
        self.decoder_5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
        )

        self.decoder_4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            # nn.Dropout(0.5),
        )
        self.decoder_3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            # nn.Dropout(0.5),
        )
        self.decoder_2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 12, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            # nn.Dropout(0.5),
        )

        self.decoder_1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 14, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """Standard forward"""
        # return self.model(input)

        x1 = self.coder_1(input)
        x2 = self.coder_2(x1)
        x3 = self.coder_3(x2)
        x4 = self.coder_4(x3)
        x5 = self.coder_5(x4)
        x6 = self.coder_6(x5)
        x7 = self.coder_7(x6)
        y7 = self.innermost_8(x7)

        # add skip connections
        y6 = self.decoder_7(torch.cat([x7, y7], 1))

        y5 = self.decoder_6(torch.cat([x6, y6], 1))

        y4 = self.decoder_5(torch.cat([x5, y5], 1))
        y4to2 = F.interpolate(y4, scale_factor=4, mode='bilinear', align_corners=True)
        y4to1 = F.interpolate(y4, scale_factor=8, mode='bilinear', align_corners=True)

        y3 = self.decoder_4(torch.cat([x4, y4], 1))
        y3to1 = F.interpolate(y3, scale_factor=4, mode='bilinear', align_corners=True)

        y2 = self.decoder_3(torch.cat([x3, y3], 1))

        y1 = self.decoder_2(torch.cat([x2, y4to2, y2], 1))

        output = self.decoder_1(torch.cat([x1, y4to1, y3to1, y1], 1))

        return output, y1, y2, y3, y4


class dsuGeneratorRGB2NIR(nn.Module):
    """Create a Dense scale Unet-based generator"""

    def __init__(self, input_nc, output_nc, norm_G, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(dsuGeneratorRGB2NIR, self).__init__()
        # construct unet structure
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.image_size = [128, 64, 32, 16]
        # add the outermost layer
        self.coder_1 = nn.Sequential(
            # nn.Conv2d(input_nc,1,kernel_size=1, stride=1, padding=0, bias= use_bias),
            nn.Conv2d(1, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            MambaIR(img_size=self.image_size[0], embed_dim=ngf),
        )

        # gradually reduce the number of filters from ngf * 8 to ngf
        self.coder_2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            MambaIR(img_size=self.image_size[1], embed_dim=ngf * 2),
            norm_layer(ngf * 2),
        )
        self.coder_3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            MambaIR(img_size=self.image_size[2], embed_dim=ngf * 4),
            norm_layer(ngf * 4),
        )
        self.coder_4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        # add intermediate layers with ngf * 8 filters
        self.coder_5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )
        self.coder_6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )
        self.coder_7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        # add the innermost layer
        self.innermost_8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        self.decoder_7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
        )
        self.decoder_6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
        )

        self.decoder_5 = decoder(ngf * 16, ngf * 8, norm_G)

        self.decoder_4 = decoder(ngf * 16, ngf * 4, norm_G)

        # self.decoder_3 = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
        #     norm_layer(ngf * 2),
        #     # nn.Dropout(0.5),
        # )
        self.decoder_3 = decoder(ngf * 8, ngf * 2, norm_G, self.image_size[2], Mamba_use=False)

        self.decoder_2 = decoder(ngf * 12, ngf, norm_G, self.image_size[1], Mamba_use=False)

        self.block = nn.Sequential(
            nn.ReLU(True),
            #MambaIR(img_size=self.image_size[0], embed_dim=ngf * 14),
            nn.ConvTranspose2d(ngf * 14, ngf, kernel_size=4, stride=2, padding=1),
        )
        # self.fusion = SPADE(norm_G.replace('spectral', ''), 1, 1)
        self.fusion = SPADEResnetBlock(3, 3, 1, norm_G)
        self.cross_block = CrissCrossAttention(ngf, 3)
        self.fin_out = nn.Sequential(nn.Conv2d(ngf, 3, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, input, hsv, f1, f2, f3, f4):
        """Standard forward"""
        x1 = self.coder_1(input)
        x2 = self.coder_2(x1)
        x3 = self.coder_3(x2)
        x4 = self.coder_4(x3)
        x5 = self.coder_5(x4)
        x6 = self.coder_6(x5)
        x7 = self.coder_7(x6)
        y7 = self.innermost_8(x7)

        # add skip connections
        y6 = self.decoder_7(torch.cat([x7, y7], 1))

        y5 = self.decoder_6(torch.cat([x6, y6], 1))

        y4 = self.decoder_5(torch.cat([x5, y5], 1), f4)
        y4to2 = F.interpolate(y4, scale_factor=4, mode='bilinear', align_corners=True)
        y4to1 = F.interpolate(y4, scale_factor=8, mode='bilinear', align_corners=True)

        y3 = self.decoder_4(torch.cat([x4, y4], 1), f3)
        y3to1 = F.interpolate(y3, scale_factor=4, mode='bilinear', align_corners=True)

        y2 = self.decoder_3(torch.cat([x3, y3], 1), f2)

        y1 = self.decoder_2(torch.cat([x2, y4to2, y2], 1), f1)

        # 使用SPADEResnetBlock
        segmap = self.apply_laplacian(input)[:, None]
        feature = self.block(torch.cat([x1, y4to1, y3to1, y1], 1))
        seg_hsv = self.fusion(hsv, segmap)
        # print(feature.shape, seg_hsv.shape, hsv.shape, segmap.shape, input.shape)
        seg_hsv = self.cross_block(feature, seg_hsv)
        output = self.fin_out(seg_hsv)
        return output

    def apply_laplacian(self, input):
        # 将归一化的图像转换为 [0, 255] 范围内的数据
        nir_image = (input.detach().cpu().numpy()[:, 0] * 255).astype(np.uint8)

        laplacian = cv2.Laplacian(nir_image, cv2.CV_64F)

        laplacian = (laplacian - np.min(laplacian)) / (np.max(laplacian) - np.min(laplacian))
        # print(laplacian)
        # 将处理后的图像转换回PyTorch张量
        laplacian_tensor = torch.from_numpy(laplacian).to(input.device)
        #     out.append(laplacian_tensor)
        # out = torch.stack(out, dim=0).to(input.device)
        return laplacian_tensor


class dsuGenerator(nn.Module):
    """Create a Dense scale Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(dsuGenerator, self).__init__()
        # construct unet structure
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # add the outermost layer
        self.coder_1 = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)
        )

        # gradually reduce the number of filters from ngf * 8 to ngf
        self.coder_2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
        )
        self.coder_3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
        )
        self.coder_4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        # add intermediate layers with ngf * 8 filters
        self.coder_5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )
        self.coder_6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )
        self.coder_7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        # add the innermost layer
        self.innermost_8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
        )

        self.decoder_7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
        )
        self.decoder_6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
        )
        self.decoder_5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
        )

        self.decoder_4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            # nn.Dropout(0.5),
        )
        self.decoder_3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            # nn.Dropout(0.5),
        )
        self.decoder_2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 12, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            # nn.Dropout(0.5),
        )

        self.decoder_1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 14, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """Standard forward"""
        # return self.model(input)

        x1 = self.coder_1(input)
        x2 = self.coder_2(x1)
        x3 = self.coder_3(x2)
        x4 = self.coder_4(x3)
        x5 = self.coder_5(x4)
        x6 = self.coder_6(x5)
        x7 = self.coder_7(x6)
        y7 = self.innermost_8(x7)

        # add skip connections
        y6 = self.decoder_7(torch.cat([x7, y7], 1))

        y5 = self.decoder_6(torch.cat([x6, y6], 1))

        y4 = self.decoder_5(torch.cat([x5, y5], 1))
        y4to2 = F.interpolate(y4, scale_factor=4, mode='bilinear', align_corners=True)
        y4to1 = F.interpolate(y4, scale_factor=8, mode='bilinear', align_corners=True)

        y3 = self.decoder_4(torch.cat([x4, y4], 1))
        y3to1 = F.interpolate(y3, scale_factor=4, mode='bilinear', align_corners=True)

        y2 = self.decoder_3(torch.cat([x3, y3], 1))

        y1 = self.decoder_2(torch.cat([x2, y4to2, y2], 1))

        output = self.decoder_1(torch.cat([x1, y4to1, y3to1, y1], 1))

        return output


class all_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        self.netG_A = dsuGeneratorRGB2NIR(input_nc, output_nc, 'spectralinstance', ngf, norm_layer=norm_layer,
                                          use_dropout=use_dropout)
        # self.netG_C = dsuGenerator_hsv(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.netG_C = UNetGenerator(ngf // 2)

    def forward(self, nir_img, nir_hsv_img):
        fake_B_hsv, f1, f2, f3, f4 = self.netG_C(nir_img, nir_hsv_img)
        # print(fake_B_hsv.shape, f1.shape, f2.shape, f3.shape, f4.shape)
        fake_B = self.netG_A(nir_img, fake_B_hsv, f1, f2, f3, f4)
        return fake_B_hsv, fake_B


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        norm_layer = SpectralNorm
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw,
                                     bias=use_bias)),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            norm_layer(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


if __name__ == '__main__':
    model = all_Generator(3, 3)
    x_gray = torch.randn(size=(3, 1, 256, 256))
    x = torch.randn(size=(3, 3, 256, 256))
    model(x_gray, x)
