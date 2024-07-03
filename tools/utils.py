import torch
import random
import torch.nn as nn


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class BinaryLoss(nn.Module):
    def __init__(self, real):
        super(BinaryLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.real = real

    def forward(self, logits):
        if self.real:
            labels = torch.ones(logits.shape[0], 1)
        else:
            labels = torch.zeros(logits.shape[0], 1)
        if logits.is_cuda:
            labels = labels.cuda()
        return self.bce(logits, labels)


class CategoryLoss(nn.Module):
    def __init__(self, category_num):
        super(CategoryLoss, self).__init__()
        emb = nn.Embedding(category_num, category_num)
        emb.weight.data = torch.eye(category_num)
        self.emb = emb
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, category_logits, labels):
        target = self.emb(labels)
        return self.loss(category_logits, target)


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  ## 放入一张图像，再从buffer里取一张出来
        to_return = []  ## 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:  ## 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  ## 满了就1/2的概率从buffer里取，或者就用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def backward_D(real_A, real_B, fake_B, netD, real_binary_loss, fake_binary_loss):
    real_AB = torch.cat([real_A, real_B], 1)
    fake_AB = torch.cat([real_A, fake_B], 1)

    real_D_logits = netD(real_AB)
    fake_D_logits = netD(fake_AB.detach())

    d_loss_real = real_binary_loss(real_D_logits)
    d_loss_fake = fake_binary_loss(fake_D_logits)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    return d_loss.item()
