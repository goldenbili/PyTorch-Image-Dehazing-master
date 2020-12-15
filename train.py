import argparse
import os

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim
import torchvision
import numpy as np

import dataloader
import net

import cv2
from cv2 import imwrite

'''
Willy: 
作法：
最後一層直接兩維輸出
Q: loss function 的 Y 永遠是對的 這樣會不會影響訓練效果
*try: 只做 UV 的 loss function ??

進度：
20101126
1. net修改（o）
2. 輸入:img->rgb->yuv444(ori-img)->yuv420(dehaze-img)（ing）
'''


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    use_gpu = config.use_gpu

    if use_gpu:
        dehaze_net = net.dehaze_net().cuda()
    else:
        dehaze_net = net.dehaze_net()

    dehaze_net.apply(weights_init)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path,mode="val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    if use_gpu:
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    dehaze_net.train()

    for epoch in range(config.num_epochs):
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            '''
            list_image = []
            # 300
            for i in range(len(img_orig)):
                unit_img_orig = img_orig[i].cpu().detach().numpy()
                list_size = [unit_img_orig.size()]

                #
                batch_list_img_orig = []
                # 8
                for j in range(list_size[0]):
                    unit_img_orig_one = unit_img_orig[j]
                    batch_list_img_orig.append(unit_img_orig_one)
                    # list_image.append(img_orig[i].cpu().detach().numpy())
                    #for train_index in range(config.train_batch_size):
            '''



            '''
            img_full = np.concatenate(list_image[0:20], 1)
            #imwrite("result_3x3.jpg", img_full)
            for i in range(1, len(list_image), 20):
                img_row = np.concatenate(list_image[i:i + 20], 1)
                img_full = np.concatenate([img_full, img_row], 0)
            #imwrite("result_3x3.jpg", img_full)
            '''

            for index in range(len(img_orig)):
                unit_img_orig = img_orig[index]
                unit_img_haze = img_haze[index]
                # train stage
                if use_gpu:
                    unit_img_orig = unit_img_orig.cuda()
                    unit_img_haze = unit_img_haze.cuda()

                clean_image = dehaze_net(unit_img_haze)

                loss = criterion(clean_image, unit_img_orig)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(),config.grad_clip_norm)
                optimizer.step()

            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1,"_",str(index), ":", loss.item())
            if ((iteration+1) % config.snapshot_iter) == 0:
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + "_" + str(index) + '.pth')

        # Validation Stage
        sub_image_list = []
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):
            for index in range(len(img_orig)):
                unit_img_orig = img_orig[index]
                unit_img_haze = img_haze[index]
                # train stage
                if use_gpu:
                    unit_img_orig = unit_img_orig.cuda()
                    unit_img_haze = unit_img_haze.cuda()

                clean_image = dehaze_net(unit_img_haze)
                sub_image_list.append(clean_image)

            image_all = np.concatenate(sub_image_list[:32],1)
            for i in range(1,15):
                index = i*32
                image_row = np.concatenate(sub_image_list[index:index+32],0)
                image_all = np.concatenate(image_all,image_row)
            #torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")

        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")










if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="test_images/")
    #parser.add_argument('--hazy_images_path', type=str, default="data/data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    parser.add_argument('--use_gpu',type=int, default=0)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)









