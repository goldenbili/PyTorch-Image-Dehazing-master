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
from google.colab.patches import cv2_imshow
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
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    use_gpu = config.use_gpu
    bk_width = config.block_width
    bk_height = config.block_height
    resize = config.resize

    if use_gpu:
        dehaze_net = net.dehaze_net().cuda()
    else:
        dehaze_net = net.dehaze_net()

    if config.snap_train_data:
        dehaze_net.load_state_dict(torch.load(config.snapshots_folder + config.snap_train_data))
    else:
        dehaze_net.apply(weights_init)
    print(dehaze_net)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path, 'train', resize, bk_width, bk_height)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path, "val", resize, bk_width, bk_height)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)

    if use_gpu:
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    dehaze_net.train()

    # 同一組訓練資料跑 epoch 次
    save_counter = 0
    for epoch in range(config.num_epochs):
        # 有 iteration 張一起訓練.
        # img_orig , img_haze 是包含 iteration 個圖片的 tensor 資料集 , 訓練時會一口氣訓練 iteration 個圖片.
        # 有點像將圖片橫向拼起來 實際上是不同維度.
        for iteration, (img_orig, img_haze, rgb,  bl_num_width, bl_num_height, data_path) in enumerate(train_loader):

            if save_counter == 0:
                print("img_orig.size:")
                print(len(img_orig))
                print("bl_num_width.type:")
                print(bl_num_width.type)
                print("shape:")
                print(bl_num_width.shape)

            # train stage
            num_width = int(bl_num_width[0].item())
            num_height = int(bl_num_height[0].item())
            full_bk_num = num_width * num_height
            display_block_iter = full_bk_num / config.display_block_iter
            for index in range(len(img_orig)):
                unit_img_orig = img_orig[index]
                unit_img_haze = img_haze[index]
                if save_counter == 0:
                    print("unit_img_orig type:")
                    print(unit_img_orig.type())
                    print("size:")
                    print(unit_img_orig.size())
                    print("shape:")
                    print(unit_img_orig.shape)

                if use_gpu:
                    unit_img_orig = unit_img_orig.cuda()
                    unit_img_haze = unit_img_haze.cuda()

                clean_image = dehaze_net(unit_img_haze)

                loss = criterion(clean_image, unit_img_orig)

                if torch.isnan(unit_img_haze).any() or torch.isinf(clean_image).any():
                    print("unit_img_haze:")
                    print(unit_img_haze.shape)
                    print(unit_img_haze)

                    print("clean_image:")
                    print(clean_image.shape)
                    print(clean_image)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
                optimizer.step()

                # show loss every config.display_block_iter
                if ((index + 1) % display_block_iter) == 0:
                    print("Loss at Epoch:" + str(epoch) + "_index:" + str(index + 1) + "/" + str(len(img_orig)) +
                          "_iter:" + str(iteration + 1) + "_Loss value:" + str(loss.item()))
                # save snapshot every save_counter times
                if ((save_counter + 1) % config.snapshot_iter) == 0:
                    save_name = "Epoch:" + str(epoch) + "_TrainTimes:" + str(save_counter + 1) + ".pth"
                    torch.save(dehaze_net.state_dict(), config.snapshots_folder + save_name)
                    # torch.save(dehaze_net.state_dict(),
                    #           config.snapshots_folder , "Epoch:", str(epoch), "
                    #           _TrainTimes:", str(save_counter+1), ".pth")

                save_counter = save_counter + 1

        # Validation Stage

        for iter_val, (img_orig, img_haze, rgb, bl_num_width,  bl_num_height, data_path) in enumerate(val_loader):
            sub_image_list = []
            ori_sub_image_list = []
            rgb_image_list = []
            for index in range(len(img_orig)):
                unit_img_orig = img_orig[index]
                unit_img_haze = img_haze[index]
                unit_img_rgb = rgb[index]

                if use_gpu:
                    unit_img_orig = unit_img_orig.cuda()
                    unit_img_haze = unit_img_haze.cuda()
                    unit_img_rgb = unit_img_rgb.cuda()

                clean_image = dehaze_net(unit_img_haze)
                if index == 0:
                    print("sub image index:" + str(index))
                    print("yuv444 tensor:")
                    print(unit_img_orig.shape)
                    print(unit_img_orig)
                    print("yuv420 tensor:")
                    print(clean_image.shape)
                    print(clean_image)
                    print("rgb tensor:")
                    print(unit_img_rgb.shape)
                    print(unit_img_rgb)

                # 先把 yuv 轉回 rgb
                '''
                R = Y + 1.4075 * (V - 128)
                G = Y - 0.3455 * (U - 128) - (0.7169 * (V - 128))
                B = Y + 1.7790 * (U - 128)
                
                R = Y + 1.402 (V-128)
                G = Y - 0.34414 (U-128) - 0.71414 (V-128)
                B = Y + 1.772 (U-128)
                '''

                sub_image_list.append(clean_image)
                ori_sub_image_list.append(unit_img_orig)
                rgb_image_list.append(unit_img_rgb)

            '''
            print("iter_val:"+str(iter_val))
            print("num_width-tensor:")
            print(bl_num_width)
            '''

            print(data_path)
            temp_data_path = data_path[0]
            print('temp_data_path:')
            print(temp_data_path)
            orimage_name = temp_data_path.split("/")[-1]
            print(orimage_name)
            orimage_name = orimage_name.split(".")[0]
            print(orimage_name)

            num_width = int(bl_num_width[0].item())
            # num_width = int(bl_num_width[iter_val].item())
            # print("num_width:" + str(num_width))

            num_height = int(bl_num_height[0].item())
            # num_height = int(bl_num_height[iter_val].item())
            # print("num_height:" + str(num_height))
            full_bk_num = num_width * num_height

            # ------------------------------------------------------------------#
            image_all = torch.cat((sub_image_list[:num_width]), 3)
            # print("Merge image1.index" + str(iter_val))
            # print("image_all.shape")

            for i in range(num_width, full_bk_num, num_width):
                image_row = torch.cat(sub_image_list[i:i + num_width], 3)
                image_all = torch.cat([image_all, image_row], 2)
            '''
            print("image_all_shape:")
            print(image_all.shape)
            '''
            torchvision.utils.save_image(image_all, config.sample_output_folder + "Epoch:" + str(epoch) +
                                         "_Index:" + str(iter_val + 1) + "_" + orimage_name + "_cal.jpg")
            # ------------------------------------------------------------------#

            # ------------------------------------------------------------------#
            image_all_ori = torch.cat(ori_sub_image_list[:num_width], 3)
            '''
            image_all_ori = torch.cat((ori_sub_image_list[0], ori_sub_image_list[1]), 1)
            for j in range(2, num_width):
                image_all_ori = torch.cat((image_all_ori, ori_sub_image_list[j]), 1)
            '''
            for i in range(num_width, full_bk_num, num_width):
                image_row = torch.cat(ori_sub_image_list[i:i + num_width], 3)
                '''
                image_row = torch.cat((ori_sub_image_list[i],ori_sub_image_list[i +1]), 1)
                for j in range(i+2, num_width):
                    image_row = torch.cat((image_row, ori_sub_image_list[j]), 1)
                '''
                image_all_ori = torch.cat([image_all_ori, image_row], 2)
            image_name = config.sample_output_folder + str(iter_val + 1) + "_ori.jpg"
            print(image_name)
            # torchvision.utils.save_image(image_all_ori, image_name)
            torchvision.utils.save_image(image_all, config.sample_output_folder + "Epoch:" + str(epoch) +
                                         "_Index:" + str(iter_val + 1) + "_" + orimage_name + "_ori.jpg")
            # ------------------------------------------------------------------#

            # ------------------------------------------------------------------#
            rgb_image_all = torch.cat(rgb_image_list[:num_width], 3)
            '''
            image_all_ori = torch.cat((ori_sub_image_list[0], ori_sub_image_list[1]), 1)
            for j in range(2, num_width):
                image_all_ori = torch.cat((image_all_ori, ori_sub_image_list[j]), 1)
            '''
            for i in range(num_width, full_bk_num, num_width):
                image_row = torch.cat(rgb_image_list[i:i + num_width], 3)
                '''
                image_row = torch.cat((ori_sub_image_list[i],ori_sub_image_list[i +1]), 1)
                for j in range(i+2, num_width):
                    image_row = torch.cat((image_row, ori_sub_image_list[j]), 1)
                '''
                rgb_image_all = torch.cat([rgb_image_all, image_row], 2)
            image_name = config.sample_output_folder + str(iter_val + 1) + "_rgb.jpg"
            print(image_name)
            torchvision.utils.save_image(rgb_image_all, config.sample_output_folder + "Epoch:" + str(epoch) +
                                         "_Index:" + str(iter_val + 1) + "_" + orimage_name + "_rgb.jpg")
            # ------------------------------------------------------------------#

        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    # parser.add_argument('--orig_images_path', type=str, default="test_images/")
    # parser.add_argument('--hzy_images_path', type=str, default="data/data/")
    parser.add_argument('--orig_images_path', type=str, default="test_images/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--display_block_iter', type=int, default=6)
    parser.add_argument('--snapshot_iter', type=int, default=10000)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    parser.add_argument('--snap_train_data', type=str, default="")
    parser.add_argument('--use_gpu', type=int, default=0)
    parser.add_argument('--resize', type=bool, default=False)
    parser.add_argument('--block_width', type=int, default=32)
    parser.add_argument('--block_height', type=int, default=32)

    conf = parser.parse_args()
    print("snapshots_folder:" + conf.snapshots_folder)
    print("snap_train_data:" + conf.snap_train_data)
    print("sample_output_folder:" + conf.sample_output_folder)

    if not os.path.exists(conf.snapshots_folder):
        os.mkdir(conf.snapshots_folder)
    if not os.path.exists(conf.sample_output_folder):
        os.mkdir(conf.sample_output_folder)

    train(conf)

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
