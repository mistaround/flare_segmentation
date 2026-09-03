"""BracketFlare (CVPR 2023, 光心对称先验) 推理 -> 反射光晕(鬼影) mask.

上游 third_party/BracketFlare/test.py 写死 .cuda() 且不输出 mask。
本脚本在 CPU 上跑通同一个 MPRNet 预训练模型, 并输出显式的鬼影分割:

  ghost_soft.png  预测出的反射光晕层亮度 (连续)
  ghost_mask.png  二值鬼影 mask
  overlay.png     输入图 + 红色鬼影 mask

模型输入是 6 通道 [原图, 绕光心旋转180度并做 gamma 校正的图],
这正是论文的核心先验: 鬼影关于光心与光源中心对称。

用法:
  python tools/bracketflare_mask_demo.py --input <img_or_dir> --output outputs/bracketflare
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, transforms

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "BracketFlare")
sys.path.insert(0, os.path.abspath(REPO))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basicsr.archs.mprnet_arch import MPRNet  # noqa: E402
from basicsr.utils.flare_util import RandomGammaCorrection  # noqa: E402
from mask_utils import flare_mask_from_layer, save_gray, save_overlay, split_6ch  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs/bracketflare")
    ap.add_argument("--model_path", default=os.path.join(REPO, "experiments/net_g_last.pth"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--mask_mode", default="otsu", choices=["otsu", "rel", "ratio", "abs"])
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    paths = (sorted(p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(IMG_EXT))
             if os.path.isdir(args.input) else [args.input])
    if args.limit:
        paths = paths[: args.limit]
    assert paths, f"no image found under {args.input}"

    model = MPRNet(img_ch=6, output_ch=6)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False)["params"])
    model = model.to(device).eval()

    # 光心对称先验: 旋转 180 度 = 水平翻转 + 垂直翻转
    rot = Compose([RandomGammaCorrection(10.0), RandomHorizontalFlip(1.0), RandomVerticalFlip(1.0)])
    to_tensor = transforms.ToTensor()
    resize, crop = transforms.Resize(args.size), transforms.CenterCrop(args.size)

    os.makedirs(args.output, exist_ok=True)
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = crop(resize(to_tensor(Image.open(path).convert("RGB")))).unsqueeze(0).to(device)

        with torch.no_grad():
            lq = torch.cat((img, rot(img)), 1)
            out = model(lq)[0]
        deflare, ghost, _ = split_6ch(out)
        deflare, ghost = deflare.clamp(0, 1), ghost.clamp(0, 1)

        binary, soft, tau = flare_mask_from_layer(ghost, img, args.mask_mode, args.thresh)
        b, s = binary.cpu().numpy(), soft.cpu().numpy()

        save_gray(f"{args.output}/{stem}_ghost_soft.png", s)
        save_gray(f"{args.output}/{stem}_ghost_mask.png", b)
        TF.to_pil_image(ghost[0]).save(f"{args.output}/{stem}_ghost_layer.png")
        TF.to_pil_image(deflare[0]).save(f"{args.output}/{stem}_deflare.png")
        save_overlay(f"{args.output}/{stem}_overlay.png",
                     img[0].permute(1, 2, 0).cpu().numpy(), red_mask=b)

        tau_s = "otsu-auto" if tau is None else f"{tau:.4f}"
        print(f"[ok] {os.path.basename(path)}  tau={tau_s}  ghost_px={b.mean()*100:.2f}%")


if __name__ == "__main__":
    main()
