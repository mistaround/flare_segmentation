"""SpecSeg 推理 -> 镜面高光(specular highlight)二值 mask.

这是三个 demo 里唯一"网络本身就直接输出 mask"的:
上游仓库自带训练好的 Keras 权重 (SpecSeg_weights.hdf5, 256x256x1 -> 256x256x1 sigmoid),
但只有一个 Colab notebook, 不能直接跑。本脚本把它抽成 CPU 命令行推理。

输出:
  spec_prob.png  高光概率图 (连续)
  spec_mask.png  二值高光 mask
  overlay.png    输入图 + 红色 mask

注意: 权重是在 WHU-specular 数据集(物体表面反光)上训的, 换域(夜景灯光)效果会下降,
这里只用于验证推理链路可跑通。

用法:
  .venv-tf/bin/python tools/specseg_mask_demo.py --input <img_or_dir> --output outputs/specseg
"""
import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "SpecSeg")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs/specseg")
    ap.add_argument("--model_path", default=os.path.join(REPO, "SpecSeg_weights.hdf5"))
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = (sorted(p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(IMG_EXT))
             if os.path.isdir(args.input) else [args.input])
    if args.limit:
        paths = paths[: args.limit]
    assert paths, f"no image found under {args.input}"

    model = tf.keras.models.load_model(args.model_path, compile=False)
    os.makedirs(args.output, exist_ok=True)

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        pil = Image.open(path).convert("RGB")
        w, h = pil.size
        gray = np.array(pil.convert("L").resize((SIZE, SIZE)))
        # 与 notebook 训练时一致: keras.utils.normalize(batch, axis=1)
        x = tf.keras.utils.normalize(gray[None].astype(np.float32), axis=1)[..., None]

        prob = model.predict(x, verbose=0)[0, :, :, 0]
        mask = (prob > args.thresh).astype(np.float32)

        prob_full = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h)))
        mask_full = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h))) > 127

        Image.fromarray(prob_full).save(f"{args.output}/{stem}_spec_prob.png")
        Image.fromarray((mask_full * 255).astype(np.uint8)).save(f"{args.output}/{stem}_spec_mask.png")
        ov = np.asarray(pil).astype(np.float32) / 255.0
        ov[..., 0] = np.where(mask_full, 0.5 * ov[..., 0] + 0.5, ov[..., 0])
        Image.fromarray((ov * 255).astype(np.uint8)).save(f"{args.output}/{stem}_overlay.png")

        print(f"[ok] {os.path.basename(path)}  {w}x{h}  "
              f"prob_max={prob.max():.3f}  mask_px={mask.mean()*100:.2f}%")


if __name__ == "__main__":
    main()
