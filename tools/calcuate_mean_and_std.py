import numpy as np
import cv2
import os


def get_mean_and_std(imgs_path,img_h,img_w):
    # img_h, img_w = 32, 32
    means, stdevs = [], []
    img_list = []
    imgs_path_list = os.listdir(imgs_path)

    len_ = len(imgs_path_list)
    i = 0
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(imgs_path, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    means = [i*255 for i in means]
    stdevs = [i*255 for i in stdevs]
    
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))


if __name__ == '__main__':
    imgs_path = r'D:\SourcetreeSpace\mmseg-pectoralis\data\pectoralis_dataset_cropbg_unified\train\images'  # 图片目录
    img_h = 512
    img_w = 320  # 根据自己数据集适当调整，别太大了，最开始头铁4000、6000速度特别慢
    get_mean_and_std(imgs_path, img_h, img_w)