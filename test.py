import os
from PIL import Image
from tools.tools import calculate_rmse, calculate_psnr, calculate_ssim, calculate_ae, calculate_lpips
import numpy as np
import tqdm
import torch
from models import CycleGanNIR_net
from torch.utils import data
import data_loader


if __name__ == '__main__':
    device = 'cuda'
    weights_path = './save_weights_V2'
    path_result_out = './result'
    weights_path = weights_path + '/best.pth'
    assert weights_path.split('/')[-1].split('.')[-1] == 'pth'
    os.makedirs(path_result_out, exist_ok=True)
    input_shape = (1, 256, 256)
    model = CycleGanNIR_net.all_Generator(3, 3).to(device)
    key = model.load_state_dict(torch.load(weights_path, map_location=device))
    print(key)
    model.eval()

    dataes = data_loader.get_test_paths()

    test_loader = data.DataLoader(data_loader.Dataset_test(dataes), batch_size=1,
                                  shuffle=False, num_workers=4)
    ssim_, psnr_, fsim_, ae_ = [], [], [], []
    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        real_A_gray = batch['nir_gray'].to(device)
        real_A_rgb = batch['nir_rgb'].to(device)
        real_A_hsv = batch['nir_hsv'].to(device)
        real_B = batch['rgb_rgb'].to(device)
        path_name = batch['rgb_path'][0]
        name = path_name.split('_')[-3]

        with torch.no_grad():
            fake_B_hsv, fake_B = model(real_A_gray, real_A_hsv)
        real_B = real_B.cpu().numpy()[0].transpose(1, 2, 0)
        fake_B = fake_B.cpu().numpy()[0].transpose(1, 2, 0)
        real_A = real_A_rgb.cpu().numpy()[0].transpose(1, 2, 0)
        out = fake_B * 255
        Image.fromarray(out.astype(np.uint8)).save(path_result_out + '/' + name + '.png')
        ae_.append(calculate_ae(real_B, fake_B))
        ssim_.append(calculate_ssim(real_B, fake_B))
        psnr_.append(calculate_psnr(real_B, fake_B))

    print('psnr: ', np.mean(psnr_))
    print('ssim: ', np.mean(ssim_))
    print('ae: ', np.mean(ae_))
    with open('best_test.txt', 'w') as f:
        f.write("Average PSNR %f\n" % np.mean(psnr_))
        f.write("Average SSIM %f\n" % np.mean(ssim_))
        f.write("Average AE %f \n" % np.mean(ae_))

        f.write("PSNR_list:\n")
        for index, item in enumerate(psnr_):
            f.write("{}. {}\n".format(index + 1, item))

        f.write("SSIM_list:\n")
        for index, item in enumerate(ssim_):
            f.write("{}. {}\n".format(index + 1, item))

        f.write("AE_list:\n")
        for index, item in enumerate(ae_):
            f.write("{}. {}\n".format(index + 1, item))


