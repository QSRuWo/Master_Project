"""Utils for monoDepth.
"""
import sys
import re
import numpy as np
import cv2
import torch

from PIL import Image


from .pallete import get_mask_pallete

def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)

# def write_pfm(path, image, scale=1):
#     """Write pfm file.
#
#     Args:
#         path (str): pathto file
#         image (array): data
#         scale (int, optional): Scale. Defaults to 1.
#     """
#
#     # 使用 'wb' 模式打开文件，'wb' 意味着以二进制写模式打开文件。
#     with open(path, "wb") as file:
#         color = None
#
#         # 检查图像数据的数据类型是否是float32，如果不是，抛出异常。
#         if image.dtype.name != "float32":
#             raise Exception("Image dtype must be float32.")
#
#         # 翻转图像数组，因为PFM格式要求从左下角开始读取数据。
#         image = np.flipud(image)
#
#         # 如果图像是三通道的彩色图像，设置color为True
#         if len(image.shape) == 3 and image.shape[2] == 3:
#             color = True
#         # 如果图像是灰度图像，设置color为False
#         elif (
#             len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
#         ):
#             color = False
#         # 如果图像不是彩色也不是灰度图像，抛出异常。
#         else:
#             raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")
#
#         # 根据图像类型（彩色或灰度）写入PFM头信息。
#         file.write("PF\n" if color else "Pf\n".encode())
#         # 写入图像的宽和高。
#         file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))
#
#         # 获取图像数据的字节序信息。
#         endian = image.dtype.byteorder
#
#         # 检查系统和图像数据的字节序，如果需要，改变scale的符号。
#         if endian == "<" or endian == "=" and sys.byteorder == "little":
#             scale = -scale
#
#         # 写入scale信息。
#         file.write("%f\n".encode() % scale)
#
#         # 将图像数据写入文件。
#         image.tofile(file)



def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def resize_image(img):
    """Resize image and make it fit for network.

    Args:
        img (array): image

    Returns:
        tensor: data ready for network
    """
    height_orig = img.shape[0]
    width_orig = img.shape[1]

    if width_orig > height_orig:
        scale = width_orig / 384
    else:
        scale = height_orig / 384

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )
    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """Resize depth map and bring to CPU (numpy).

    Args:
        depth (tensor): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")

    depth_resized = cv2.resize(
        depth.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )

    return depth_resized


def write_depth(path, depth, bits=1, absolute_depth=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    write_pfm(path + ".pfm", depth.astype(np.float32))

    if absolute_depth:
        out = depth
    else:
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2 ** (8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return
# def write_depth(path, depth, bits=1, absolute_depth=False):
#     """Write depth map to pfm and png file.
#
#     Args:
#         path (str): filepath without extension
#         depth (array): depth
#     """
#
#     # 使用write_pfm函数将深度数据保存为PFM文件。深度数据转为float32类型。
#     write_pfm(path + ".pfm", depth.astype(np.float32))
#
#     # 如果absolute_depth为True，直接使用原始深度数据。
#     if absolute_depth:
#         out = depth
#     else:
#         # 计算深度数据的最小和最大值。
#         depth_min = depth.min()
#         depth_max = depth.max()
#
#         # 计算要使用的最大值，根据给定的位数（例如8位或16位）。
#         max_val = (2 ** (8 * bits)) - 1
#
#         # 如果深度范围大于浮点数的最小值，则对深度数据进行归一化。
#         # 否则，创建一个与深度数据同形状的零数组。
#         if depth_max - depth_min > np.finfo("float").eps:
#             out = max_val * (depth - depth_min) / (depth_max - depth_min)
#         else:
#             out = np.zeros(depth.shape, dtype=depth.dtype)
#
#     # 如果指定的位数是1（即8位），将处理后的深度数据保存为8位PNG。
#     if bits == 1:
#         cv2.imwrite(path + ".png", out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
#     # 如果指定的位数是2（即16位），将处理后的深度数据保存为16位PNG。
#     elif bits == 2:
#         cv2.imwrite(path + ".png", out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
#
#     return



def write_segm_img(path, image, labels, palette="detail", alpha=0.5):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        image (array): input image
        labels (array): labeling of the image
    """

    mask = get_mask_pallete(labels, "ade20k")

    img = Image.fromarray(np.uint8(255*image)).convert("RGBA")
    seg = mask.convert("RGBA")

    out = Image.blend(img, seg, alpha)

    out.save(path + ".png")

    return
