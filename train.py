import os

import numpy as np

from tools.tools import calculate_rmse, calculate_psnr
import torch
import tqdm
from tools.fit import CycleGAN
import data_loader
from torch.utils import data

if __name__ == '__main__':
    checkpoint_dir = './save_weights_V3'
    os.makedirs(checkpoint_dir, exist_ok=True)
    Lconst_penalty = 15
    batch_size = 8
    n_epochs = 1000
    schedule = 50
    best_psnr = 0
    gpu_ids = ['cuda:0']
    model = CycleGAN(
        Lconst_penalty=Lconst_penalty,
        gpu_ids=gpu_ids
    )

    model.setup()
    model.print_networks(True)

    dataes, dataes_test = data_loader.get_data_paths()

    train_loader = data.DataLoader(data_loader.Dataset(dataes), batch_size=batch_size,
                                   shuffle=True, num_workers=4)
    test_loader = data.DataLoader(data_loader.Dataset_test(dataes_test), batch_size=1,
                                  shuffle=True, num_workers=4)
    for epoch in range(n_epochs):
        dt_size = len(train_loader.dataset)
        pbar = tqdm.tqdm(
            total=dt_size // batch_size,
            desc=f'Epoch {epoch + 1} / {n_epochs}',
            postfix=dict,
            miniters=.3
        )
        model.netG.train()
        for i, batch in enumerate(train_loader):
            model.set_input(batch)

            d_loss, g_loss = model.optimize_parameters()

            pbar.set_postfix(**{
                'G_loss': g_loss,
                'D_loss': d_loss,
            })
            pbar.update(1)
        pbar.close()
        dt_size_test = len(test_loader.dataset)
        pbar_test = tqdm.tqdm(
            total=dt_size_test,
            desc=f'Epoch {epoch + 1} / {n_epochs}',
            postfix=dict,
            miniters=.3
        )
        model.netG.eval()
        test_psnr = []
        for i, batch in enumerate(test_loader):
            model.set_input(batch)
            real_B = batch['rgb_rgb']
            # d_loss, g_loss = model.optimize_parameters()
            with torch.no_grad():
                model.forward()
            pred = model.fake_B
            real_B = real_B.cpu().numpy()[0].transpose(1, 2, 0)
            fake_B = pred.cpu().numpy()[0].transpose(1, 2, 0)
            p = calculate_psnr(real_B, fake_B)
            test_psnr.append(p)
            pbar_test.set_postfix(**{
                # 'G_loss': g_loss,
                # 'D_loss': d_loss,
                'psnr': np.mean(test_psnr)
            })
            pbar_test.update(1)
        pbar_test.close()
        if (epoch + 1) % schedule == 0:
            model.update_lr()
        if np.mean(test_psnr) > best_psnr:
            best_psnr = np.mean(test_psnr)
            torch.save(model.netG.state_dict(), checkpoint_dir + '/best.pth')
        if epoch % 50 == 0:
            torch.save(model.netG.state_dict(), checkpoint_dir + '/weights_{}.pth'.format(epoch))
