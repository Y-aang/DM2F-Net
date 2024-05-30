# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, HazeRD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy, DM2FNet_woPhy_loss_weight_x0, DM2FNet_woPhy_inference_weight_x0
from model import DM2FNet_woPhy_inference_weight_x0, DM2FNet_woPhy_loss_weight_x0

from datasets import SotsDataset, OHazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS'
# exp_name = 'ohaze_x1'
# exp_name = 'ohaze_x0_infer'
# exp_name = 'ohaze_x0_loss_infer'
exp_name = 'HazeRD'

args = {
    # 'snapshot': 'iter_20000_loss_0.05122_lr_0.000000',
    # 'snapshot': 'iter_16000_loss_0.05120_lr_0.000047',
    # 'snapshot': 'iter_18000_loss_0.04626_lr_0.000025',
    'snapshot': 'iter_18000_loss_0.05337_lr_0.000025',
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    # 'O-Haze': OHAZE_ROOT,
    'HazeRD': HazeRD_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name or 'HazeRD' in name:
                # net = DM2FNet_woPhy_loss_weight_x0().cuda()
                # net = DM2FNet_woPhy_inference_weight_x0().cuda()
                # net = DM2FNet_woPhy_loss_weight_x0().cuda()
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'val')
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                print('path: ', os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'))
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:0'))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims = [], []
            loss_record = AvgMeter()

            for idx, data in tqdm(enumerate(dataloader)):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name or 'HazeRD' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    print('ssim image shape: ', r.shape)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False
                                                 , win_size=3
                                                 )
                    ssims.append(ssim)
                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}")


if __name__ == '__main__':
    main()
