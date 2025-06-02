import argparse
import os
import time
from functools import partial

import SimpleITK as sitk
import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.transforms import allow_missing_keys_mode
from tqdm import tqdm

from src.modules.segmentation.multitalent_module import MultiTalentModule
from src.modules.segmentation.supervised_module import SupervisedSegmentationModule
from src.utils.dataset import make_multi_dataloader, make_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True, type=str)
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--ckpt_file', required=True, type=str)
    parser.add_argument('--type', required=False, type=str, default='multitalent')
    parser.add_argument('--save_dir', required=True, type=str)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        config = config.get('config', config)

    config['dataset']['data_dir'] = args.data_root
    config['dataset']['cache_num'] = 0

    if args.type == 'multitalent':
        module = MultiTalentModule.load_from_checkpoint(args.ckpt_file, strict=False).cuda()
        tst_loader, transforms = make_multi_dataloader(config, subset='test', return_transforms=True)
    else:
        module = SupervisedSegmentationModule.load_from_checkpoint(args.ckpt_file, strict=False).cuda()
        tst_loader, transforms = make_dataloader(config['dataset'], subset='test', return_transforms=True)

    count_skips = 0

    with (torch.amp.autocast('cuda') and torch.no_grad()):
        for batch in tqdm(tst_loader):
            time.sleep(1)

            if args.type == 'multitalent':
                x = batch['image']
                d_name = batch['dataset'][0]
                x = x.to("cuda") if torch.cuda.is_available() else x

                y_pred = sliding_window_inference(x,
                                                  roi_size=config['dataset']['img_size'],
                                                  sw_batch_size=8,
                                                  predictor=partial(module.model, keys=d_name),
                                                  overlap=0.25,
                                                  mode='gaussian',
                                                  )
            else:
                x = batch['image']
                x = x.to("cuda") if torch.cuda.is_available() else x

                y_pred = sliding_window_inference(x,
                                                  roi_size=config['dataset']['img_size'],
                                                  sw_batch_size=8,
                                                  predictor=module.model,
                                                  overlap=0.25,
                                                  mode='gaussian',
                                                  )

            with allow_missing_keys_mode(transforms):
                inverted_y_hat = transforms.inverse({"label": y_pred[0]})

            y_hat = (torch.sigmoid(inverted_y_hat['label']) > 0.5)[1:]

            for c in range(y_hat.shape[0]):
                y_hat_c = y_hat[c].permute(2, 1, 0).cpu().numpy().astype(np.uint8)

                orig_img = sitk.ReadImage(f"{args.data_root}/test/img/{batch['fname'][0]}")
                pred_img = sitk.GetImageFromArray(y_hat_c)
                pred_img.CopyInformation(orig_img)

                # Save the predicted label image
                os.makedirs(f"{args.save_dir}", exist_ok=True)
                sitk.WriteImage(pred_img, f"{args.save_dir}/{os.path.basename(batch['fname'][0]).replace('.nii.gz', f'_c{c+1}.nii.gz')}")