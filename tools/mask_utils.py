"""光晕/高光 mask 的公共工具: gamma、亮度、阈值化、可视化。

所有上游仓库(Flare7K / BracketFlare)的 gamma 工具函数都写死了 .cuda(),
这里给出等价的、设备无关的实现, 便于在 CPU 上调试。
"""
import numpy as np
import torch
from PIL import Image


def gamma_fwd(x, gamma=2.2):
    """gamma 域 -> 线性域。"""
    return torch.clamp(x, 1e-7, 1.0) ** gamma


def gamma_inv(x, gamma=2.2):
    """线性域 -> gamma 域。"""
    return torch.clamp(x, 1e-7, 1.0) ** (1.0 / gamma)


def luminance(t):
    """[B,3,H,W] -> [B,H,W] 的 Rec.709 亮度。"""
    return 0.2126 * t[:, 0] + 0.7152 * t[:, 1] + 0.0722 * t[:, 2]


def split_6ch(out, gamma=2.2):
    """等价于上游 predict_flare_from_6_channel, 但不依赖 CUDA。"""
    deflare, flare = out[:, :3], out[:, 3:]
    merge_lin = gamma_fwd(deflare, gamma) + gamma_fwd(flare, gamma)
    return deflare, flare, gamma_inv(torch.clamp(merge_lin, 1e-7, 1.0), gamma)


def flare_mask_from_layer(flare, merge, mode="otsu", thresh=0.15, floor=0.02):
    """由预测出的光晕层得到显式 mask。

    光晕是加性半透明层 (merge_linear = scene_linear + flare_linear),
    所以阈值化要在**线性域的光晕亮度**上做, 而不是 gamma 域的像素值上。

    mode:
      otsu  自适应阈值, 无超参, 跨曝光/跨场景最稳 (默认)
      rel   阈值 = thresh * max(flare_lin)
      ratio 光晕能量占比 flare/merge > thresh, 暗背景上容易误检
      abs   固定亮度阈值
    返回 (binary [H,W] float, soft [H,W] float, tau)
    """
    flare_lin = luminance(gamma_fwd(flare))[0]
    merge_lin = luminance(gamma_fwd(merge))[0]
    flare_np = flare_lin.cpu().numpy()

    if mode == "otsu":
        from skimage.filters import threshold_otsu
        tau = float(threshold_otsu(flare_np))
    elif mode == "rel":
        tau = thresh * float(flare_np.max())
    elif mode == "abs":
        tau = thresh
    elif mode == "ratio":
        ratio = flare_lin / merge_lin.clamp_min(1e-6)
        binary = ((ratio > thresh) & (flare_lin > floor)).float()
        return binary, ratio.clamp(0, 1), None
    else:
        raise ValueError(mode)

    binary = (flare_lin > tau).float()
    soft = (flare_lin / max(float(flare_np.max()), 1e-6)).clamp(0, 1)
    return binary, soft, tau


def save_gray(path, arr01):
    Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(path)


def save_overlay(path, rgb01, red_mask=None, green_mask=None, alpha=0.5):
    ov = rgb01.copy()
    if red_mask is not None:
        ov[..., 0] = np.where(red_mask > 0.5, (1 - alpha) * ov[..., 0] + alpha, ov[..., 0])
    if green_mask is not None:
        ov[..., 1] = np.where(green_mask > 0.5, (1 - alpha) * ov[..., 1] + alpha, ov[..., 1])
    Image.fromarray((np.clip(ov, 0, 1) * 255).astype(np.uint8)).save(path)
