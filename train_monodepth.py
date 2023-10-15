import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import cv2
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt.models_with_adabins import DPTDepthModelWithAdaBins
from torchsummary import summary
from loss import SILogLoss, BinsChamferLoss
import numpy as np
import logging
from dataloader import DepthDataLoader

# class NYUDepthV2Dataset(Dataset):
#     def __init__(self, rgb_dir, depth_dir, transforms=None):
#         super(NYUDepthV2Dataset, self).__init__()
#         self.rgb_dir = rgb_dir
#         self.depth_dir = depth_dir
#         self.transforms = transforms
#         self.filenames = os.listdir(rgb_dir)
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         rgb_path = os.path.join(self.rgb_dir, self.filenames[idx])
#         depth_path = os.path.join(self.depth_dir, self.filenames[idx])
#
#         # try:
#         #     print(rgb_path)
#         #     print(os.path.exists(rgb_path))
#         # except:
#         #     raise Exception()
#
#         rgb_image = cv2.imread(rgb_path)
#         depth_image = cv2.imread(depth_path)
#
#         if rgb_image.ndim == 2:
#             rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
#         if depth_image.ndim == 2:
#             depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
#
#         rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) / 255.0
#         # 临时改成变成灰度图
#         depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY) / 255.0
#         depth_image = depth_image.reshape(depth_image.shape[0], depth_image.shape[1], 1)
#
#
#         if self.transforms:
#             result = self.transforms({"image": rgb_image, "depth": depth_image})
#             rgb_image, depth_image = result["image"], result["depth"]
#
#         # 临时增加因为没采用数据增强
#         rgb_image = torch.from_numpy(rgb_image)
#         depth_image = torch.from_numpy(depth_image)
#         # print(depth_image.shape)
#
#         # 暂时不采用数据增强，所以临时改变一下图像和深度图的张量形状
#         rgb_image = rgb_image.permute(2, 0, 1)
#         depth_image = depth_image.permute(2, 0, 1)
#
#         # print(f"train_monodepth.py NYUDepthV2Dataset image path: {rgb_path}\n"
#         #       f"train_monodepth.py NYUDepthV2Dataset depth path: {depth_path}\n"
#         #       f"train_monodepth.py NYUDepthV2Dataset image shape: {rgb_image.shape}\n"
#         #       f"train_monodepth.py NYUDepthV2Dataset depth shape: {depth_image.shape}\n")
#         # raise Exception
#         return rgb_image, depth_image


def run(args, path_to_rgb, path_to_depth, batch_size, num_workers, lr, epochs, model_path, model_type='dpt_hybrid', optimize=True, min_val=1e-3, max_val=10):
    # 读取参数
    print("Start reading data...")
    rgb_dir = path_to_rgb
    depth_dir = path_to_depth
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    batch_size = batch_size
    num_workers = num_workers
    lr = lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_output = args.weight_output
    validate_every = args.validate_every

    print("device: %s" % device)

    # 定义日志
    # logging.basicConfig(filename=f"./logs/dpt_hybird_adabins/{args.Model_Type}_log.log",
    #                     level=logging.INFO,
    #                     format='%(asctime)s [%(levelname)s]: %(message)s',
    #                     filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个handler，写入日志文件
    fh = logging.FileHandler(f"./logs/dpt_hybird_adabins/{args.Model_Type}_log.log", mode='w')
    fh.setLevel(logging.INFO)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加handler到logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    if model_type == 'dpt_hybrid_adabins':
        net_w = net_h = 384
        model = DPTDepthModelWithAdaBins(
            path=None,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
            n_bins=256,
            min_val=1e-3,
            max_val=10
        )
    # 加载预训练模型
    pretrained_weights = torch.load(r"./weights/nyu_merged_weights.pt")

    model.load_state_dict(pretrained_weights, strict=False)
    model.to(device)
    # 使得只用decoder部分和conv1x1参与反向传播
    for name, param in model.named_parameters():
        # 暂时先不fine tune scratch部分
        if name.startswith("conv_out") or name.startswith("conv1x1"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    # print(f"Printing the model architecture...")
    # summary(model, input_size=(3, 384, 512))
    # print(model)

    # 定义数据增强
    # transforms = Compose(
    #     [
    #         Resize(
    #             net_w,
    #             net_h,
    #             resize_target=None,
    #             keep_aspect_ratio=True,
    #             ensure_multiple_of=32,
    #             resize_method="minimal",
    #             image_interpolation_method=cv2.INTER_CUBIC,
    #         ),
    #         normalization,
    #         PrepareForNet(),
    #     ]
    # )

    # 创建dataloader
    # nyu_dataset = NYUDepthV2Dataset(rgb_dir, depth_dir, transforms=None)
    # total_size = len(nyu_dataset)
    # train_size = int(0.8 * total_size)
    # val_size = int(0.1 * total_size) + 1
    # test_size = total_size - train_size - val_size
    #
    # train_dataset, val_dataset, test_dataset = random_split(nyu_dataset, [train_size, val_size, test_size])
    # print(f"Length of train_dataset: {len(train_dataset)} \n"
    #       f"Length of val_dataset: {len(val_dataset)} \n"
    #       f"Length of test_dataset: {len(test_dataset)} \n"
    #       f"Total Length of NYU Depth V2 dataset: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # test_loader =DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loaders = DepthDataLoader(args, "train")
    train_loader, val_loader, test_loader = loaders.get_dataloader()

    # if optimize == True and device == torch.device("cuda"):
    #     model = model.to(memory_format=torch.channels_last)
    #     model = model.half()

    # 定义优化器和损失函数
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # criterion = nn.MSELoss() # 暂时使用MSELoss
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss()
    w_chamfer = 0.1

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
                                              div_factor=25, final_div_factor=100)

    best_loss = np.inf
    
    print("Start training...")

    for epoch in range(epochs):
        model.train()

        step = 0

        loss_per_epoch = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}, Training", unit="batch"):
            # inputs, depths = inputs.to(device).half(), depths.to(device).half()
            # inputs, depths = inputs.to(device).float(), depths.to(device).float()
            inputs = batch["image"].to(device)
            depths = batch["depth"].to(device)
            # print(inputs.shape, depths.shape)
            # print(f"inputs.shape: {inputs.shape}")
            # print(f"depths.shape: {depths.shape}")

            optimizer.zero_grad()
            # 在conv_out加了双线性插值使得输出为输入尺寸
            bin_edges, preds = model(inputs)

            mask = depths > args.min_depth

            # print(f"preds.shape before criterion_ueff: {preds.shape}")
            # print(f"depths.shape before criterion_ueff: {depths.shape}")
            # print(f"mask.shape before criterion_ueff: {mask.shape}")

            l_dense = criterion_ueff(preds, depths, mask=mask.to(torch.bool), interpolate=True)

            l_chamfer = criterion_bins(bin_edges, depths)

            loss = l_dense + w_chamfer * l_chamfer

            loss_per_epoch += loss

            loss.backward()

            # 裁剪梯度，防止出现过大的梯度值（这是一个可选步骤）
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional

            optimizer.step()

            scheduler.step()

            logger.info(f"Epoch: {epoch+1}/{epochs}, Step: {step+1}/{len(train_loader)}, Loss: {loss.item()}")

            if step % validate_every == 0:
                model.eval()

                with torch.no_grad():
                    loss_val = 0
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}, Validation", unit="batch"):
                        inputs_val, depths_val = batch["image"].to(device), batch["depth"].to(device)

                        bin_edges_val, preds_val = model(inputs_val)

                        mask_val = depths_val > args.min_depth

                        l_dense_val = criterion_ueff(preds_val, depths_val, mask=mask_val.to(torch.bool), interpolate=True)

                        l_chamfer_val = criterion_bins(bin_edges_val, depths_val)

                        loss_val += l_dense_val + w_chamfer * l_chamfer_val

                    loss_val /= len(val_loader)

                    logger.info(f"Epoch: {epoch+1}/{epochs}, Step: {step+1}/{len(train_loader)}, Validation loss: {loss_val}")

                    torch.save(model.state_dict(), f"./weight_store/{args.Model_Type}/latest_model.pth")
                    logger.info(f"New latest model saved with validation loss: {loss_val}")

                    if loss_val < best_loss:
                        best_loss = loss_val
                        torch.save(model.state_dict(), f"./weight_store/{args.Model_Type}/best_model.pth")
                        logger.info(f"New best model saved with validation loss: {loss_val}")

                model.train()
            step += 1
        loss_per_epoch /= len(train_loader)
        logger.info(f"Average loss in epoch {epoch+1}: {loss_per_epoch}")


if __name__ == "__main__":
    current_path = os.environ.get('PATH', '')
    cl_path = r"E:\Microsoft Visual Studio 2022\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"
    new_path = f"{cl_path};{current_path}"
    os.environ['PATH'] = new_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nyu", type=str)
    parser.add_argument("--RGB_Path", default="downloadDataset/NYU/nyu_images")
    parser.add_argument("--Depth_Path", default="downloadDataset/NYU/nyu_depths")
    parser.add_argument("--Model_Type", default="dpt_hybrid_adabins")
    parser.add_argument("--Model_Path", default=r"weights/dpt_hybrid-midas-501f0c75.pt")
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--lr", default=0.000357, help="initial lr")
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--validate_every", default=100)
    parser.add_argument("--weight_output", default=r"./weight_store/dpt_hybrid_adabins", type=str)
    parser.add_argument("--min_depth", default=1e-3, type=float)
    parser.add_argument("--max_depth", default=10, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float, help="For optimizer AdamW")
    parser.add_argument("--do_kb_crop", action="store_true", help="For kitti dataset")
    parser.add_argument('--do_random_rotate', default=True,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument("--degree", type=float, default=2.5, help="For random rotate")
    parser.add_argument("--input_height", type=int, default=416, help="crop height of image to input height, for data preprocess")
    parser.add_argument("--input_width", type=int, default=544, help="crop width of image to input width, for data preprocess")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    run(args, args.RGB_Path, args.Depth_Path, args.batch_size, args.num_workers, args.lr, args.epochs, args.Model_Path, args.Model_Type)