import torch
import torch.nn as nn
from torchvision import transforms

from dpt.models_with_adabins import DPTDepthModelWithAdaBins

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        # 检查pic是否为PIL图像或numpy格式的图像
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        # 如果pic是numpy数组（即ndarray），将其转换为torch.Tensor
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        # 如果pic是PIL图像，以下代码处理不同的图像模式并转换为torch.Tensor
        # 这一部分主要是处理不同的PIL图像数据类型和模式
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        # 获取图像通道数
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        # 重新排列张量的维度以匹配[C, H, W]格式
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # 如果图像是ByteTensor类型，则转换为float类型
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img



@torch.no_grad()
def predict(image, model):
    min_depth = 1e-3
    max_depth = 10
    print(f"infer.py predict() image.shape: {image.shape}")
    bins, pred = model(image)
    print(f"infer.py predict() pred.shape: {pred.shape}")
    print(f"infer.py predict() pred requires grad: {pred.requires_grad}")
    # 使用numpy将预测值剪切到指定的深度范围内
    pred = np.clip(pred.cpu().numpy(), min_depth, max_depth)

    print(f"infer.py predict() pred.shape after clip: {pred.shape}")

    # Flip
    # 将图像沿水平轴翻转（即左右翻转），这是数据增强的一种常见手段
    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to("cuda:0")
    # 使用模型对翻转的图像进行预测
    pred_lr = model(image)[-1]
    # 将翻转后的预测值剪切到指定的深度范围，并再次进行左右翻转以对齐原始图像
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], min_depth, max_depth)

    # Take average of original and mirror
    final = 0.5 * (pred + pred_lr)
    # 将预测值的大小调整为与原始图像相同
    final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
                                      mode='bilinear', align_corners=True).cpu().numpy()
    print(f"infer.py predict() final.shape: {final.shape}")
    # 为预测值设置上下限
    final[final < min_depth] = min_depth
    final[final > max_depth] = max_depth
    # 为无穷大和NaN的值设置默认值
    final[np.isinf(final)] = max_depth
    final[np.isnan(final)] = min_depth
    # 计算分箱中心
    centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
    centers = centers.cpu().squeeze().numpy()
    centers = centers[centers > min_depth]
    centers = centers[centers < max_depth]

    print(f"infer.py predict() final.shape: {final.shape}")

    return centers, final

if __name__ == "__main__":
    model_path = None
    model = DPTDepthModelWithAdaBins(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
        n_bins=256,
        min_val=1e-3,
        max_val=10
    )

    pretrained_weights = torch.load(r"./weights/nyu_merged_weights.pt")
    model_dict = model.state_dict()

    print(len(model_dict), len(pretrained_weights))

    for key, value in model_dict.items():
        if key not in pretrained_weights.keys():
            print(key)

    model.load_state_dict(pretrained_weights, strict=False)
    model.to("cuda:0")
    # input = torch.ones(1, 3, 480, 640)
    # bin_edges, pred = model(input)
    # print(pred.shape)

    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    img = Image.open(r"E:\MasterProject\AdaBins-main\test_imgs\frame-000338.color.jpg")
    img = np.asarray(img) / 255.

    toTensor = ToTensor()

    img = toTensor(img).unsqueeze(0).float().to("cuda:0")

    bin_centers, pred = predict(img, model)

    plt.imshow(pred.squeeze(), cmap='magma_r')
    plt.show()
