#!/usr/bin/env bash
# 一键复现: clone 上游仓库 + 建 py310 环境 + 下载预训练权重 + 抓几张测试图。
# 全程 CPU, 不需要训练。已在 Linux / Python 3.10.20 / torch 2.14+cpu 上验证。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

# ---------- 1. clone 上游仓库 (pin 到验证过的 commit) ----------
mkdir -p third_party
clone_pin () {  # repo  dir  sha
  if [ ! -d "third_party/$2" ]; then
    git clone -q "https://github.com/$1" "third_party/$2"
    git -C "third_party/$2" checkout -q "$3"
  fi
}
clone_pin ykdai/Flare7K       Flare7K      d1fb66ecb3c75fb3f4bfa715c49ef9265892d56f
clone_pin ykdai/BracketFlare  BracketFlare f48ff7fba92448d0c6856657eb0c6911a307b85d
clone_pin Atif-Anwer/SpecSeg  SpecSeg      0fc1248f0615f73fd44237ac51e13f762c2975fd
clone_pin qulishen/FlareX     FlareX       8b6bfbf31b8d861630546cee41fdc7dd6e28c9e0  # 备用: 只做 removal

# ---------- 2. PyTorch 环境 (Flare7K++ / BracketFlare) ----------
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python addict future lmdb "numpy<2" opencv-python-headless Pillow \
    pyyaml requests scikit-image scipy tqdm yapf timm einops kornia tensorboard gdown

# ---------- 3. TensorFlow 环境 (SpecSeg, 单独 venv 避免和 torch 冲突) ----------
uv venv --python 3.10 .venv-tf
uv pip install --python .venv-tf/bin/python "tensorflow-cpu==2.15.1" "numpy<2" pillow scikit-image opencv-python-headless

# ---------- 4. 预训练权重 (Google Drive) ----------
mkdir -p third_party/Flare7K/experiments/flare7kpp third_party/BracketFlare/experiments
if [ ! -f third_party/Flare7K/experiments/flare7kpp/net_g_last.pth ]; then
  .venv/bin/gdown 17AX9BJ-GS0in9Ey7vw3BVPISm67Rpzho -O /tmp/f7kpp.zip
  unzip -q -o /tmp/f7kpp.zip -d third_party/Flare7K/experiments/
  mv -n third_party/Flare7K/experiments/net_g_last.pth third_party/Flare7K/experiments/flare7kpp/ || true
fi
if [ ! -f third_party/BracketFlare/experiments/net_g_last.pth ]; then
  .venv/bin/gdown 15AzR-VaiQO0l8Av-yE6gVuExcTwYAi45 -O /tmp/bf.zip
  unzip -q -o /tmp/bf.zip -d third_party/BracketFlare/experiments/
fi
# SpecSeg 权重已随仓库附带 (third_party/SpecSeg/SpecSeg_weights.hdf5)

# ---------- 5. 测试图 ----------
mkdir -p data/real_flare
cp -n third_party/Flare7K/test/test_images/*.png data/real_flare/ || true
# 可选: Flare7K 的 645 张真实光晕图 (223MB)
# .venv/bin/gdown 19kLXf8roHoJmxyphYvrCs9zDAXsrL1sU -O /tmp/fc.zip && unzip -q /tmp/fc.zip -d data/

echo "done. 见 docs/RUNNING.md"
