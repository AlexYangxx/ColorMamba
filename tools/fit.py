import torch
import torch.nn as nn
from models import CycleGanNIR_net
from tools.utils import CategoryLoss, BinaryLoss
from torch.optim.lr_scheduler import StepLR
from tools.losses import ReconstructionLoss
from models.commom import get_norm_layer, AutoEncoder


def load_pretrained(model, pretrained, requires_grad=False):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained)['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False


class CycleGAN:
    def __init__(self, Lconst_penalty=15, L1_penalty=100,
                 schedule=10, lr=0.0001, gpu_ids=None, is_training=True,
                 image_size=256):

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.Lconst_penalty = Lconst_penalty
        self.L1_penalty = L1_penalty

        self.schedule = schedule

        self.gpu_ids = gpu_ids

        self.lr = lr
        self.is_training = is_training
        self.image_size = image_size

    def setup(self):
        self.netG = CycleGanNIR_net.all_Generator(3, 3).to(self.gpu_ids[0])
        try:
            self.netG.load_state_dict(torch.load('./models/.G_pre.pt'), strict=False)
        except:pass
        norm_layer = get_norm_layer(norm_type='instance')
        self.netD_rgb = CycleGanNIR_net.NLayerDiscriminator(3, 64, n_layers=3, norm_layer=norm_layer).to(
            self.gpu_ids[0])
        self.netD_hsv = CycleGanNIR_net.NLayerDiscriminator(3, 64, n_layers=3, norm_layer=norm_layer).to(
            self.gpu_ids[0])

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_rgb = torch.optim.Adam(self.netD_rgb.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_hsv = torch.optim.Adam(self.netD_hsv.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.scheduler_G = StepLR(self.optimizer_G, step_size=1, gamma=0.5)
        self.scheduler_D_rgb = StepLR(self.optimizer_D_rgb, step_size=1, gamma=0.5)
        self.scheduler_D_hsv = StepLR(self.optimizer_D_hsv, step_size=1, gamma=0.5)

        self.criterionGAN = CycleGanNIR_net.GANLoss('lsgan').to(self.gpu_ids[0])
        self.pretrained = AutoEncoder().to(self.gpu_ids[0])
        load_pretrained(self.pretrained, 'assets/autoencoder.pth')
        self.criterion = ReconstructionLoss(self.pretrained)
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()

        if self.gpu_ids:
            self.l1_loss.cuda()
            self.mse.cuda()

        if self.is_training:
            self.netD_rgb.train()
            self.netD_hsv.train()
            self.netG.train()
        else:
            self.netD_rgb.eval()
            self.netD_hsv.eval()
            self.netG.eval()

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in ['G', 'D_rgb', 'D_hsv']:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_input(self, input):
        self.nir_gray = torch.Tensor(input['nir_gray']).float().to(self.gpu_ids[0])
        self.nir_rgb = torch.Tensor(input['nir_rgb']).float().to(self.gpu_ids[0])
        self.nir_hsv = torch.Tensor(input['nir_hsv']).float().to(self.gpu_ids[0])
        self.rgb_gray = torch.Tensor(input['rgb_gray']).float().to(self.gpu_ids[0])
        self.rgb_rgb = torch.Tensor(input['rgb_rgb']).float().to(self.gpu_ids[0])
        self.rgb_hsv = torch.Tensor(input['rgb_hsv']).float().to(self.gpu_ids[0])

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self):

        real_rgb_logits = self.netD_rgb(self.rgb_rgb)
        real_hsv_logits = self.netD_hsv(self.rgb_hsv)
        fake_rgb_logits = self.netD_rgb(self.fake_B.detach())
        fake_hsv_logits = self.netD_hsv(self.fake_B_hsv.detach())

        self.d_loss_rgb = self.criterionGAN(real_rgb_logits, True) + self.criterionGAN(fake_rgb_logits, False)

        self.d_loss_hsv = self.criterionGAN(real_hsv_logits, True) + self.criterionGAN(fake_hsv_logits, False)
        self.d_loss_rgb.backward()
        self.d_loss_hsv.backward()
        return self.d_loss_rgb.item() + self.d_loss_hsv.item()

    def forward(self):
        self.fake_B_hsv, self.fake_B = self.netG(self.nir_gray, self.nir_hsv)

    def backward_G(self):

        fake_B_hsv_logits = self.netD_hsv(self.fake_B_hsv)
        fake_B_rgb_logits = self.netD_rgb(self.fake_B)
        loss_B_hsv_fake = self.criterionGAN(fake_B_hsv_logits, True)
        loss_B_rgb_fake = self.criterionGAN(fake_B_rgb_logits, True)
        l1_loss = self.L1_penalty * (self.mse(self.fake_B, self.rgb_rgb) * 20 +
                                     self.criterion(self.fake_B, self.rgb_rgb) * 20 +
                                     self.criterion(self.fake_B_hsv, self.rgb_hsv) +
                                     self.mse(self.fake_B_hsv, self.rgb_hsv)
                                     )

        self.g_loss = loss_B_hsv_fake + loss_B_rgb_fake + l1_loss
        self.g_loss.backward()
        return self.g_loss.item()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD_hsv, True)  # enable backprop for D
        self.set_requires_grad(self.netD_rgb, True)
        self.optimizer_D_hsv.zero_grad()
        self.optimizer_D_rgb.zero_grad()

        D_loss = self.backward_D()
        self.optimizer_D_hsv.step()
        self.optimizer_D_rgb.step()

        self.set_requires_grad(self.netD_hsv, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_rgb, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights

        self.forward()

        self.optimizer_G.zero_grad()

        G_loss = self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights
        return D_loss, G_loss

    def update_lr(self):
        self.scheduler_D_hsv.step()
        self.scheduler_D_rgb.step()
        self.scheduler_G.step()
