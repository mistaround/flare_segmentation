"""Flare7K++ (Uformer) 推理 -> 显式光晕 mask.

上游仓库 third_party/Flare7K 的 test.py 只输出去光晕图, 且写死 .cuda().
本脚本在 CPU 上跑通同一个预训练模型, 并把"光晕层"转成可保存的显式 mask:

  flare_soft.png   预测光晕层的亮度 (0-255 连续图, 推荐做监督/后处理的中间量)
  flare_mask.png   上面按阈值二值化后的光晕分割 mask
  light_mask.png   输入图上的光源/过曝区 mask (get_highlight_mask + 形态学细化)
  overlay.png      输入图 + 红色光晕 mask + 绿色光源 mask 的可视化

用法:
  python tools/flare7kpp_mask_demo.py --input <img_or_dir> --output outputs/flare7kpp
"""
import argparse
import os
import sys
import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "Flare7K")
sys.path.insert(0, os.path.abspath(REPO))

from basicsr.archs.uformer_arch import Uformer  # noqa: E402
from basicsr.archs.unet_arch import U_Net  # noqa: E402
from basicsr.utils.flare_util import get_highlight_mask, refine_mask  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def load_params(model_path, device):
    full = torch.load(model_path, map_location=device, weights_only=False)
    for key in ("params_ema", "params"):
        if isinstance(full, dict) and key in full:
            return full[key]
    return full


def gamma_fwd(x, gamma):
    return torch.clamp(x, 1e-7, 1.0) ** gamma


def gamma_inv(x, gamma):
    return torch.clamp(x, 1e-7, 1.0) ** (1.0 / gamma)


def split_6ch(out, gamma=2.2):
    """与 predict_flare_from_6_channel 等价, 但不依赖 CUDA."""
    deflare = out[:, :3]
    flare = out[:, 3:]
    merge_lin = gamma_fwd(deflare, gamma) + gamma_fwd(flare, gamma)
    merge = gamma_inv(torch.clamp(merge_lin, 1e-7, 1.0), gamma)
    return deflare, flare, merge


def luminance(t):
    return 0.2126 * t[:, 0] + 0.7152 * t[:, 1] + 0.0722 * t[:, 2]


def save_gray(path, arr01):
    Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs/flare7kpp")
    ap.add_argument("--model_path", default=os.path.join(REPO, "experiments/flare7kpp/net_g_last.pth"))
    ap.add_argument("--model_type", default="Uformer", choices=["Uformer", "U_Net"])
    ap.add_argument("--output_ch", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", type=int, default=512, help="推理分辨率, 需为 128 的倍数")
    ap.add_argument("--mask_mode", default="otsu", choices=["otsu", "rel", "ratio", "abs"],
                    help="otsu: 对光晕层线性亮度做自适应阈值(默认, 无参数, 跨曝光稳定); "
                         "rel: 阈值 = flare_thresh * max; "
                         "ratio: 光晕能量占比 > flare_thresh (暗区易误检); "
                         "abs: 固定亮度阈值")
    ap.add_argument("--flare_thresh", type=float, default=0.15,
                    help="rel/ratio/abs 模式下的阈值, otsu 模式忽略")
    ap.add_argument("--floor", type=float, default=0.02,
                    help="ratio 模式下光晕线性亮度的下限, 避免暗区噪声被判为光晕")
    ap.add_argument("--light_thresh", type=float, default=0.97, help="光源过曝阈值")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几张, 0 表示不限")
    args = ap.parse_args()

    device = torch.device(args.device)
    if os.path.isdir(args.input):
        paths = sorted(p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(IMG_EXT))
    else:
        paths = [args.input]
    if args.limit:
        paths = paths[: args.limit]
    assert paths, f"no image found under {args.input}"

    if args.model_type == "Uformer":
        model = Uformer(img_size=args.size, img_ch=3, output_ch=args.output_ch)
    else:
        model = U_Net(img_ch=3, output_ch=args.output_ch)
    model.load_state_dict(load_params(args.model_path, device))
    model = model.to(device).eval()

    os.makedirs(args.output, exist_ok=True)
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        pil = Image.open(path).convert("RGB")
        w, h = pil.size
        x = TF.to_tensor(TF.resize(pil, [args.size, args.size])).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
        if args.output_ch == 6:
            deflare, flare, _ = split_6ch(out)
        else:  # 3 通道模型: 光晕层 = 输入 - 输出 (线性域)
            deflare = out
            flare = gamma_inv(torch.clamp(gamma_fwd(x, 2.2) - gamma_fwd(out, 2.2), 1e-7, 1.0), 2.2)
        deflare = torch.clamp(deflare, 0, 1)
        flare = torch.clamp(flare, 0, 1)

        # --- 显式 mask ---
        # 光晕是加性半透明层: merge_linear = deflare_linear + flare_linear
        # 因此在"线性域的光晕亮度"上做分割, 而不是在 gamma 域的像素值上做
        flare_lin = luminance(gamma_fwd(flare, 2.2))[0]
        merge_lin = luminance(gamma_fwd(x, 2.2))[0]
        flare_np = flare_lin.cpu().numpy()

        if args.mask_mode == "otsu":
            from skimage.filters import threshold_otsu
            tau = float(threshold_otsu(flare_np))
        elif args.mask_mode == "rel":
            tau = args.flare_thresh * float(flare_np.max())
        elif args.mask_mode == "abs":
            tau = args.flare_thresh
        else:  # ratio
            tau = None

        if tau is None:
            ratio = flare_lin / merge_lin.clamp_min(1e-6)
            flare_bin = ((ratio > args.flare_thresh) & (flare_lin > args.floor)).float()
            flare_soft = ratio.clamp(0, 1)
        else:
            flare_bin = (flare_lin > tau).float()
            flare_soft = (flare_lin / max(float(flare_np.max()), 1e-6)).clamp(0, 1)

        light_bin = get_highlight_mask(x, threshold=args.light_thresh)[0, 0]
        light_bin = torch.from_numpy(
            refine_mask(light_bin.cpu().numpy().astype(bool)).astype(np.float32)
        )
        flare_bin = torch.clamp(flare_bin + light_bin, 0, 1)    # 光源也算光晕区域

        # 放回原始分辨率
        def back(t):
            return TF.resize(t[None, None], [h, w], antialias=True)[0, 0].cpu().numpy()

        fs, fb, lb = back(flare_soft), back(flare_bin), back(light_bin)
        fb, lb = (fb > 0.5).astype(np.float32), (lb > 0.5).astype(np.float32)

        save_gray(f"{args.output}/{stem}_flare_soft.png", fs)
        save_gray(f"{args.output}/{stem}_flare_mask.png", fb)
        save_gray(f"{args.output}/{stem}_light_mask.png", lb)
        TF.to_pil_image(TF.resize(deflare[0], [h, w], antialias=True)).save(f"{args.output}/{stem}_deflare.png")
        TF.to_pil_image(TF.resize(flare[0], [h, w], antialias=True)).save(f"{args.output}/{stem}_flare_layer.png")

        rgb = np.asarray(pil).astype(np.float32) / 255.0
        ov = rgb.copy()
        ov[..., 0] = np.where(fb > 0.5, 0.5 * ov[..., 0] + 0.5, ov[..., 0])
        ov[..., 1] = np.where(lb > 0.5, 0.5 * ov[..., 1] + 0.5, ov[..., 1])
        Image.fromarray((ov * 255).astype(np.uint8)).save(f"{args.output}/{stem}_overlay.png")

        print(f"[ok] {os.path.basename(path)}  {w}x{h}  "
              f"flare_px={fb.mean()*100:.2f}%  light_px={lb.mean()*100:.2f}%")


if __name__ == "__main__":
    main()
