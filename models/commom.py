from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import Parameter
import functools
from torch.nn import Parameter
import enum
from models.mambair_arch import MambaIR

NUM_BANDS = 6


class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, sampling=None):
        layers = []

        if sampling == Sampling.DownSampling:
            layers.append(Conv3X3WithPadding(in_channels, out_channels, 2))
        else:
            if sampling == Sampling.UpSampling:
                layers.append(Upsample(2))
            layers.append(Conv3X3WithPadding(in_channels, out_channels))

        layers.append(nn.LeakyReLU(inplace=True))
        super(ConvBlock, self).__init__(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        super(AutoEncoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1], Sampling.DownSampling)
        self.conv3 = ConvBlock(channels[1], channels[2], Sampling.DownSampling)
        self.conv4 = ConvBlock(channels[2], channels[3], Sampling.DownSampling)
        self.conv5 = ConvBlock(channels[3], channels[2], Sampling.UpSampling)
        self.conv6 = ConvBlock(channels[2] * 2, channels[1], Sampling.UpSampling)
        self.conv7 = ConvBlock(channels[1] * 2, channels[0], Sampling.UpSampling)
        self.conv8 = nn.Conv2d(channels[0] * 2, out_channels, 1)

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        l6 = self.conv6(torch.cat((l3, l5), 1))
        l7 = self.conv7(torch.cat((l2, l6), 1))
        out = self.conv8(torch.cat((l1, l7), 1))
        return out


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def SynchronizedBatchNorm2d(norm_nc, affine):
    pass


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        # ks = int(parsed.group(2))
        param_free_norm_type = config_text
        ks = 3

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # 中间嵌入空间的维度。是的，硬编码。
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # 确保segmap是四维的
        if segmap.dim() == 3:
            segmap = segmap.unsqueeze(1)  # 假设segmap的形状为[N, H, W]，在第1个维度添加一个维度
        elif segmap.dim() == 2:
            segmap = segmap.unsqueeze(0).unsqueeze(0)  # 假设segmap的形状为[H, W]，在第0和第1个维度添加维度

        # 确保x和segmap是float类型
        x = x.float()
        segmap = segmap.float()

        # Part 1. generate parameter-free normalized activations
        # 第 1 步，生成无参数的规范化激活
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc, norm_G):  # 接受输入通道数 fin、输出通道数 fout 和配置选项 opt
        super().__init__()
        # Attributes # 属性
        self.learned_shortcut = (fin != fout)  # 设置一个属性 learned_shortcut，用于指示是否存在学习的快捷连接。
        fmiddle = min(fin, fout)  # 计算输入和输出通道数的最小值，作为中间通道数。

        # create conv layers 创建卷积层
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified 如果指定了谱归一化，则应用谱归一化
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers 定义规范化层
        spade_config_str = norm_G.replace('spectral', '')  # 提取 SPADE 的配置字符串，去除 'spectral'
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc)  # 创建 SPADE 规范化层 norm_0，用于对输入进行规范化
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc)  # 创建 SPADE 规范化层 norm_1，用于对中间结果进行规范化
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,  注意，带有 SPADE 的 ResNet 块还接受 |seg|，
    # the semantic segmentation map as input  即语义分割图，作为输入
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):  # 定义计算快捷连接的方法，接受输入 x 和语义分割图 seg
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)  # 定义激活函数


class decoder(nn.Module):
    def __init__(self, fin, fout, norm_G, image_size = None, Mamba_use =False):
        super().__init__()
        if Mamba_use:
            self.block = nn.Sequential(
                nn.ReLU(True),
                MambaIR(img_size=image_size, embed_dim=fin),
                nn.ConvTranspose2d(fin, fout, kernel_size=4, stride=2, padding=1),
            )
        else:
            self.block = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(fin, fout, kernel_size=4, stride=2, padding=1),
            )
        self.SPADE_color = SPADEResnetBlock(fout, fout, fout, norm_G)
        self.norm = nn.InstanceNorm2d(fout)

    def forward(self, input, hsv):
        x = self.block(input)
        x = self.SPADE_color(x, hsv)
        output = self.norm(x)

        return output
