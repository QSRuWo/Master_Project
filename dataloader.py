# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DepthDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            # 加载和预处理训练数据
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))

            # 划分数据集
            total_size = len(self.training_samples)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size) + 1
            test_size = total_size - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(self.training_samples, [train_size, val_size, test_size])

            print(f"Length of train_dataset: {len(train_dataset)} \n"
                  f"Length of val_dataset: {len(val_dataset)} \n"
                  f"Length of test_dataset: {len(test_dataset)} \n"
                  f"Total Length of NYU Depth V2 dataset: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

            # 如果是分布式训练
            if args.distributed:
                # 使用分布式采样器，确保数据在多个进程中均匀分布
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                # 如果不是分布式训练，则不使用采样器
                self.train_sampler = None

            # 创建 DataLoader 实例来加载训练数据
            self.train_loader = DataLoader(train_dataset, args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   sampler=self.train_sampler)
            self.val_loader = DataLoader(val_dataset, args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           sampler=self.train_sampler)
            self.test_loader = DataLoader(test_dataset, args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           sampler=self.train_sampler)

        elif mode == 'online_eval':
            # 加载和预处理测试数据
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            # 无论是否分布式，都不使用采样器（可能是为了确保评估在完整的数据集上进行）
            if args.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            # 创建 DataLoader 实例来加载在线评估数据
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            # 加载和预处理测试数据
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            # 创建 DataLoader 实例来加载测试数据
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

    def get_dataloader(self):
        return self.train_loader, self.val_loader, self.test_loader


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        # 临时改变
        self.filenames = os.listdir(self.args.RGB_Path)

    def __getitem__(self, idx):
        # 从文件名列表中获取对应索引的路径
        sample_path = self.filenames[idx]
        # 临时注释下面一行
        # 从路径中获取焦距值
        # focal = float(sample_path.split()[2])

        if self.mode == 'train':
            # 临时注释下面六行
            # 如果使用的数据集是kitti，且配置允许使用右侧图像，并且随机概率大于0.5
            # if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
            #     # 获取右侧图像和深度图的路径
            #     image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[3]))
            #     depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[4]))
            # else:
            #     # 否则获取常规图像和深度图的路径
            #     image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[0]))
            #     depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[1]))
            # 临时加入下面两行
            image_path = os.path.join(self.args.RGB_Path, self.filenames[idx])
            depth_path = os.path.join(self.args.Depth_Path, self.filenames[idx])
            # 临时修改下面两行
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            # 如果配置中指定了需要进行KB裁剪
            if self.args.do_kb_crop is True:
                # 计算图像的高和宽
                height = image.height
                width = image.width
                # 计算上边界和左边界的裁剪值
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                # 对深度图和图像进行裁剪
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # To avoid blank boundaries due to pixel registration
            # 对于nyu数据集，为了避免像素对齐造成的空白边界

            if self.args.dataset == 'nyu':
                # 对深度图和图像进行指定的裁剪
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            # 如果配置中指定了需要进行随机旋转
            if self.args.do_random_rotate is True:
                # 计算随机的旋转角度，范围为 [-degree, degree]
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                # 旋转图像
                image = self.rotate_image(image, random_angle)
                # 旋转深度图，使用NEAREST模式避免引入不必要的插值
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            # 将图像转换为numpy数组，并将像素值归一化到[0,1]
            image = np.asarray(image, dtype=np.float32) / 255.0
            # 将深度图转换为numpy数组
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            # 在深度图的最后一个维度上增加一个维度，将其变成(H, W, 1)的形状
            depth_gt = np.expand_dims(depth_gt, axis=2)

            # 如果数据集是nyu，对深度值进行归一化处理
            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                # 对于其他数据集，使用另一种归一化方法
                depth_gt = depth_gt / 256.0

            # 根据指定的输入高度和宽度随机裁剪图像和深度图
            image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            # 调用train_preprocess方法对图像和深度图进行进一步预处理
            image, depth_gt = self.train_preprocess(image, depth_gt)
            # sample = {'image': image, 'depth': depth_gt, 'focal': focal}
            # 临时
            sample = {'image': image, 'depth': depth_gt}

        # 如果模式不是'train'
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            # 获取图像路径
            image_path = os.path.join(data_path, remove_leading_slash(sample_path.split()[0]))
            # 加载并归一化图像
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                # 使用评估的深度图路径
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]))
                has_valid_depth = False
                # 尝试加载深度图
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True # 如果成功加载，标记为True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                # 如果有有效的深度图
                if has_valid_depth:
                    # 转换深度图为numpy数组
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    # 在深度图的最后一个维度上增加一个维度，使其变为(H, W, 1)形状
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    # 如果数据集是nyu，对深度值进行归一化处理
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        # 对于其他数据集，使用另一种归一化方法
                        depth_gt = depth_gt / 256.0

            # 如果配置中指定了需要进行kb裁剪
            if self.args.do_kb_crop is True:
                # 获取图像的高度和宽度
                height = image.shape[0]
                width = image.shape[1]
                # 计算上边距和左边距
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                # 裁剪图像
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                # 如果是在线评估模式并且有有效的深度图，则也对深度图进行裁剪
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            # 如果模式是'online_eval'
            if self.mode == 'online_eval':
                # 将处理后的图像、深度图、焦距等数据封装成字典，用于后续的在线评估
                # sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                #           'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}
                # 临时
                sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}
            else:
                # 对于其他非训练模式，只返回图像和焦距
                # sample = {'image': image, 'focal': focal}
                # 临时
                sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        # 使用PIL库的rotate方法进行旋转，angle参数指定旋转角度，resample参数决定重采样的方法
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        # 以下断言确保图像的尺寸大于或等于裁剪的尺寸，以及图像和深度图尺寸匹配
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        # 在图像的宽度和高度范围内随机选择裁剪的起始坐标
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        # 根据随机选择的坐标裁剪图像
        img = img[y:y + height, x:x + width, :]
        # 使用相同的坐标裁剪深度图
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        # 随机水平翻转图像和深度图
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy() # 沿水平方向翻转图像。
            depth_gt = (depth_gt[:, ::-1, :]).copy() # 沿水平方向翻转图像。

        # Random gamma, brightness, color augmentation
        # 随机进行gamma、亮度和颜色增强
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        # gamma增强
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        # 亮度增强
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        # 颜色增强。对图像的每个通道（R,G,B）进行随机缩放。
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image

        # 限制图像的像素值范围在[0,1]之间。
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        # 使用固定的均值和标准差来对图像进行标准化。
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        # 临时
        # 提取图像和焦距信息。
        # image, focal = sample['image'], sample['focal']
        image = sample['image']
        # 转换图像为张量格式。
        image = self.to_tensor(image)
        # 使用预定义的均值和标准差对图像进行标准化。
        image = self.normalize(image)

        # 如果当前模式为'test'，则只返回图像和焦距。
        if self.mode == 'test':
            # 临时
            # return {'image': image, 'focal': focal}
            return {'image': image}

        # 提取深度图信息。
        depth = sample['depth']
        # 如果当前模式为'train'，则转换深度图为张量并返回图像、深度图和焦距。
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            # return {'image': image, 'depth': depth, 'focal': focal}
            # 临时
            return {'image': image, 'depth': depth}
        # 对于其他模式，返回图像、深度图、焦距以及其他相关信息。
        else:
            has_valid_depth = sample['has_valid_depth']
            # return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
            #         'image_path': sample['image_path'], 'depth_path': sample['depth_path']}
            # 临时
            return {'image': image, 'depth': depth, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        # 检查输入的pic是否是PIL图像或numpy数组。
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        # 如果pic是numpy数组，直接转换其维度并返回对应的torch张量。
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        # 处理PIL图像。
        # 如果图像模式是'I'，将其转换为int32类型的torch张量。
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            # 如果图像模式是'I;16'，将其转换为int16类型的torch张量。
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            # 对于其他模式，使用pic的字节数据创建ByteTensor。
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        # 根据PIL图像的模式设置通道数。
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        # 调整张量的形状以匹配图像的尺寸和通道数。
        img = img.view(pic.size[1], pic.size[0], nchannel)

        # 调整张量的维度，使其变为[C, H, W]格式。
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        # 如果img是ByteTensor，将其转换为float张量；否则直接返回。
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
