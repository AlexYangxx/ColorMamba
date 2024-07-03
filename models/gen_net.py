import copy
import torch.nn.functional as F
import torch
from torch import nn
from models.mambair_arch import MambaIR

class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.bn2 = nn.InstanceNorm2d(out_channels)

        self.act1 = nn.LeakyReLU(.2, True)
        self.act2 = nn.LeakyReLU(.2, True)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return out

class double_conv2d_bn_mamba(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn_mamba, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.bn2 = nn.InstanceNorm2d(out_channels)

        self.act1 = nn.LeakyReLU(.2, True)
        self.act2 = nn.LeakyReLU(.2, True)
        self.mamba = MambaIR(img_size=image_size, embed_dim=out_channels)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.mamba(self.conv2(out))))
        return out

class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(.2, True)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        return out


class Edge_conv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).to(x.device)
        x = F.conv2d(x, laplacian_kernel_target, padding=1)

        return x


class UNetGenerator(nn.Module):
    def __init__(self, nfg=64):
        super(UNetGenerator, self).__init__()
        self.image_size = [128, 64, 32, 16]
        self.egde_conv = Edge_conv()
        self.layer1_conv = double_conv2d_bn(5, nfg)
        self.layer1_conv_edge = double_conv2d_bn(1, nfg)
        self.layer2_conv = double_conv2d_bn_mamba(nfg, nfg * 2, self.image_size[0])
        self.layer2_conv_edge = copy.deepcopy(self.layer2_conv)
        self.layer3_conv = double_conv2d_bn_mamba(nfg * 2, nfg * 4, self.image_size[1])
        self.layer3_conv_edge = copy.deepcopy(self.layer3_conv)
        self.layer4_conv = double_conv2d_bn_mamba(nfg * 4, nfg * 8, self.image_size[2])
        self.layer4_conv_edge = copy.deepcopy(self.layer4_conv)
        self.layer5_conv = double_conv2d_bn(nfg * 8, nfg * 16)
        self.layer5_conv_edge = copy.deepcopy(self.layer5_conv)
        self.layer6_conv = double_conv2d_bn(nfg * 16 * 2, nfg * 8 * 2)
        self.layer7_conv = double_conv2d_bn(nfg * 8 * 2, nfg * 4 * 2)
        self.layer8_conv = double_conv2d_bn(nfg * 4 * 2, nfg * 2 * 2)
        self.layer9_conv = double_conv2d_bn(nfg * 2 * 2, nfg * 2)
        self.layer10_conv = nn.Conv2d(nfg * 2, 3, kernel_size=3,
                                      stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(nfg * 16 * 2, nfg * 8 * 2)
        self.deconv2 = deconv2d_bn(nfg * 8 * 2, nfg * 4 * 2)
        self.deconv3 = deconv2d_bn(nfg * 4 * 2, nfg * 2 * 2)
        self.deconv4 = deconv2d_bn(nfg * 2 * 2, nfg * 2)

        self.sigmoid = nn.Sigmoid()

    def get_feature(self, x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        return conv1, conv2, conv3, conv4, conv5

    def get_feature_edge(self, x):
        conv1 = self.layer1_conv_edge(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv_edge(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv_edge(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv_edge(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv_edge(pool4)

        return conv1, conv2, conv3, conv4, conv5

    def forward(self, x, style):
        edge_x = self.egde_conv(x)
        x = torch.cat([x, edge_x, style], dim=1)
        conv1, conv2, conv3, conv4, conv5 = self.get_feature(x)
        conv1_edge, conv2_edge, conv3_edge, conv4_edge, conv5_edge = self.get_feature_edge(edge_x)
        convt5 = torch.cat([conv5, conv5_edge], dim=1)
        convt1 = self.deconv1(convt5)
        concat1 = torch.cat([convt1, conv4, conv4_edge], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3, conv3_edge], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2, conv2_edge], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1, conv1_edge], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        outp = nn.Tanh()(outp)
        return outp, conv2, conv3, conv4, conv5


if __name__ == '__main__':
    model = UNetGenerator()
    x_gray = torch.randn(size=(3, 1, 256, 256))
    x = torch.randn(size=(3, 3, 256, 256))
    model(x_gray, x)
