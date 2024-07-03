import os

import matplotlib.pyplot as plt
import torch
from models import CycleGanNIR_net
import data_loader


if __name__ == '__main__':
    device = 'cuda'
    weights_path = './save_weights_V2'
    weights_path = weights_path + '/best.pth'
    assert weights_path.split('/')[-1].split('.')[-1] == 'pth'
    input_shape = (1, 256, 256)
    model = CycleGanNIR_net.all_Generator(3, 3).to(device)
    key = model.load_state_dict(torch.load(weights_path, map_location=device))
    print(key)
    model.eval()

    path_in = './datasets/Testing/visual/vis_testing_0003_nir.png'

    nir_gray, nir_rgb, nir_hsv, *_ = data_loader.Dataset(None).read_data(path_in)

    real_A_gray = nir_gray.to(device)
    real_A_rgb = nir_rgb.to(device)
    real_A_hsv = nir_hsv.to(device)

    with torch.no_grad():
        fake_B_hsv, fake_B = model(real_A_gray[None], real_A_hsv[None])
    fake_B = fake_B.cpu().numpy()[0].transpose(1, 2, 0)
    real_A = real_A_rgb.cpu().numpy().transpose(1, 2, 0)
    fake_B = data_loader.nor(fake_B)

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.title('gray')
    plt.imshow(real_A)

    plt.subplot(122)
    plt.title('rgb')
    plt.imshow(fake_B)
    plt.show()
