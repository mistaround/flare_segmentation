# 图像光晕（Lens Flare / Glare / Halo）检测与分割调研

> 调研时间：2026-09。目标：梳理"判断图像上有没有光晕"（检测/分类）与"把光晕区域抠出来"（分割/mask）这两类任务的公开工作、数据集与可复用代码。

## 1. 先厘清"光晕"指什么

学界没有一个统一叫 "halo segmentation" 的任务，"光晕"在不同语境下对应不同的退化类型，做法差别很大：

| 中文说法 | 英文术语 | 典型成因 | 视觉形态 |
|---|---|---|---|
| 光晕 / 散射光晕 | scattering flare, glare with shimmer | 镜片划痕、灰尘、油污导致的前向散射 | 光源周围一团彩色辉光 + 放射状 shimmer |
| 鬼影 / 重影 | reflective flare, ghost | 镜片组之间的多次内反射 | 沿光心对称排列的多边形/圆形光斑 |
| 星芒 | streak / starburst | 光圈叶片衍射 | 从光源射出的直线光条 |
| 雾化 / 灰蒙 | veiling glare | 杂散光整体抬高黑位 | 全图对比度下降、发灰 |
| 高光 / 镜面反光 | specular highlight | 物体表面镜面反射 | 物体上的白色过曝斑块 |
| 光晕伪影 | halo artifact | 局部色调映射 / 锐化 / HDR 融合 | 强边缘两侧的亮暗环 |

前四类属于**镜头光晕（lens flare）**，是主流研究对象；第五类是独立方向（specular highlight detection）；第六类属于图像质量评价里的伪影检测。如果本项目做的是"夜间灯光周围的光晕"，对应的是 scattering flare + glare。

## 2. 任务分三层，现状差别很大

1. **图像级检测（有/无光晕）**：工作最少，多为工程性论文和专利。
2. **区域分割（flare mask）**：**几乎没有以分割为主任务的独立论文**，但大量去光晕（flare removal）方法内部会预测一个 flare mask，而且 Flare7K 系列数据集天然带分层标注，可以直接转成分割监督。
3. **图像恢复（flare removal）**：绝对主流，2021 年后爆发，数据集和 SOTA 都集中在这里。

结论：**做光晕分割，最现实的路线是"借 removal 的数据集造分割监督"，而不是去找现成的分割数据集。**

## 3. 数据集

### 3.1 Flare7K / Flare7K++（最重要，直接可用于分割）
- Flare7K（NeurIPS 2022 D&B）：5,000 张散射光晕 + 2,000 张反射光晕，25 种散射类型、10 种反射类型。
- Flare7K++（TPAMI 2024）：= Flare7K（7,000 合成）+ Flare-R（962 张真实拍摄光晕），另配 23,949 张背景图、真实/合成测试对。
- **关键点**：它对每张 flare 图分层提供 *light source / glare with shimmer / streak / reflective flare* 的**分离图**。这几张分层图做阈值或直接当 soft mask，就是现成的**逐像素分割标注**；原论文也把 "lens flare segmentation / light source extraction" 列为数据集的应用之一，并用 PSPNet 做过分割 baseline。
- 合成方式是把 flare 层以加性方式叠到背景上，所以能无限量生成 (flare 图, 无 flare 图, flare 层, 分量 mask) 四元组。
- 代码/数据：https://github.com/ykdai/Flare7K ，许可证 **S-Lab License 1.0（仅限非商业用途）**。
- 评测除了 PSNR/SSIM/LPIPS，还提供带 mask 的分区域指标（光晕区 / 光源区分别算）。

### 3.2 FlareX（2025，物理仿真）
- 12,500 对图像：9,500 对来自 2D 合成（95 种物理光晕类型 × 100 模板，Blender 插件生成），3,000 对来自 60 个场景的 3D 渲染。
- 提供模板的**组件级标注**（light source / streak / iris / glare）。
- 真实测试集用"遮光片遮住光源"的方式拿到真正无光晕 GT，评测时排除被遮挡区域。
- 论文只做 removal，没做分割，但组件标注同样可转分割监督。
- 代码：https://github.com/qulishen/FlareX ，**CC BY 4.0**（比 Flare7K 宽松，商用友好）。

### 3.3 Wu et al., ICCV 2021（合成 + 半经验光学仿真）
- "How to Train Neural Networks for Flare Removal"，用波动光学仿真散射光晕 + 实拍反射光晕，约 2 万条合成数据；提出训练时**保留光源**的技巧（预测光源区域并贴回），这套 loss/mask 处理后来被广泛沿用。Google 有对应专利（US 12,033,309）。

### 3.4 眩光 / 太阳眩光类
- **Robust Glare Detection: Review, Analysis, and Dataset Release**（arXiv 2110.06006）：声称发布首个 glare detection 数据集（多相机拍摄），用改进 U-Net 在 RGB/HSV 等多种表示上做眩光分割。⚠️ 实测其仓库 https://github.com/maesfahani/glaredetection 目前仍只是占位页，**数据未真正放出**。
- **GLARE: A Dataset for Traffic Sign Detection in Sun Glare**（arXiv 2209.08716）：2,157 张强眩光下的实拍图，41 类交通标志**框标注**——是"眩光条件下的检测"，不是"眩光本身的标注"。
- **WoodScape sun glare**（Valeo，环视鱼眼）：论文 *Let The Sunshine in: Sun Glare Detection on Automotive Surround-view Cameras*；另有 NeurIPS 2021 workshop 的自监督太阳眩光检测（靠渲染眩光造监督，评估检测器对眩光的鲁棒性）。多为图像级/区域级判定。

### 3.5 镜面高光（相邻方向，代码最成熟）
- **SHIQ**（ACM MM 2020 / CVPR 2021 JSHDR）：约 16K 四元组（原图 / **二值高光 mask** / 高光分量 / 漫反射图），10k 训练 + 1k 测试，200×200。
- SpecSeg、SHDNet、M2-Net 等都是"检测+去除"多任务网络。
- 如果你的"光晕"其实是物体表面反光，**这一支才是对口的、且有现成像素级 mask 的数据**。

## 4. 方法

### 4.1 显式预测光晕 mask 的（可直接借鉴为分割器）
- **TPRR-Net / 光心对称先验**（CVPR 2023, *Nighttime Smartphone Reflective Flare Removal Using Optical Center Symmetry Prior*）：利用鬼影关于光心对称的物理先验，用 U-Net **分割出反射鬼影区域**再修复——最接近"纯分割"的工作。
- **DeFlare-Net**（PReMI 2023）：显式拆成 Light Source Detection (LSD) 模块 + Flare Removal Network，loss = flare loss + light-source loss + reconstruction loss。
- **FCNet**（CVIU 2025）：先做 flare detection 阶段，用 Spatial-Frequency Complementary Module 在空域+频域感知光晕，产出**区分光晕 / 光源 / 其他区域的三值 mask**，再去除。
- **MFDNet**（arXiv 2406.18079，多频分解）、**DFDNet**（arXiv 2507.17489，动态频率引导）：按频带分离光晕分量，中间产物近似 soft mask。
- **Difflare**（arXiv 2407.14746）：latent diffusion 去光晕。
- 传统法：Bernsen 局部阈值 + 多 mask 融合做眩光分割；*Automatic Flare Spot Artifact Detection and Removal in Photographs*（JMIV 2018 / arXiv 2103.04384）用几何+光度先验检测光斑鬼影，无需训练。

### 4.2 图像级"有没有光晕"的检测
- **A Reference-Free Lens-Flare-Aware Detector for Autonomous Driving**（Sensors 2026, doi:10.3390/s26082359）：轻量 CNN 量化 "flare impact" + 三层 MLP，用 Log-Likelihood Ratio loss 做判定，**无参考图**即可给出是否受光晕影响。这是目前最贴近"检测有无光晕"的正式论文。
- **zarifaziz/DetectingFlares**（GitHub）：简单二分类（faulty/good）工程实现，可作为最小基线。
- 工业标准侧：ISO 18844 用专用图卡定量测 veiling glare；DXOMARK、Image Engineering 有成体系的 flare 评测方法（适合做"检测阈值"的定义参考）。
- 专利方向：Google *Learning-based lens flare removal*（US 12,033,309）、*Flare detection and mitigation in panoramic images*（US 9,692,995）、*Image flare detection using asymmetric pixels*（US 10,848,693，靠传感器硬件检测）。

### 4.3 综述与竞赛
- **Toward Flare-Free Images: A Survey**（arXiv 2310.14354）：把光晕分为 scattering / reflective / glare / orb / starburst 五类，覆盖硬件、传统图像处理、深度学习三条路线，指标为 PSNR/SSIM/LPIPS。入门首选。
- **MIPI Challenge on Nighttime Flare Removal**（CVPRW 2023 / 2024）：赛题、baseline、排行榜齐全，是找 SOTA 方法最快的入口。

## 5. 对本项目（flare_segmentation）的建议

1. **先定义清楚 mask 语义**。光晕是**半透明的加性层**，二值边界本身是人为的。建议：
   - 主监督用**回归 flare 层**（连续 alpha / 强度图），推理时按阈值出二值 mask；
   - 或者按 Flare7K 的分量分别出 4 通道 mask（光源 / 辉光 / 星芒 / 鬼影），比单通道"有光晕"信息量大得多。
2. **数据**：用 Flare7K++ 的 flare 层 + 任意背景图在线合成，mask 由 flare 层直接算（如 luminance > τ 或归一化强度）。想商用就换 FlareX（CC BY 4.0）。真实域用 Flare-R 的 962 张实拍光晕 + 少量人工标注做微调/验证。
3. **模型**：分割侧用 U-Net / SegFormer 起步即可；如果要同时输出"有没有光晕"，加一个全局分类头（对 mask 面积/强度做监督），比单独训分类器更省事、也更好解释。
4. **必须单独处理光源本身**。绝大多数方法都会强调"去光晕但保留光源"，光源和光晕在像素统计上高度重叠，把它们合成一类会让 mask 和后续修复都变差。
5. **评测**：分割用 IoU/F1 + 边界指标；检测用 AUC/TPR@低 FPR（因为负样本远多于正样本）；如果最终服务于 removal，还要报 Flare7K 的分区域 PSNR（光晕区 / 光源区）。
6. **别忽略邻域数据**：如果场景包含物体表面反光，SHIQ 提供现成二值 mask，可以直接做多任务或预训练。

## 6. 主要链接

- Flare7K/Flare7K++：https://github.com/ykdai/Flare7K ｜ 论文 https://arxiv.org/abs/2210.06570 、https://arxiv.org/abs/2306.04236
- FlareX：https://github.com/qulishen/FlareX ｜ https://arxiv.org/abs/2510.09995
- 综述：https://arxiv.org/abs/2310.14354
- ICCV 2021 flare removal：https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_How_To_Train_Neural_Networks_for_Flare_Removal_ICCV_2021_paper.pdf
- CVPR 2023 光心对称先验：https://openaccess.thecvf.com/content/CVPR2023/papers/Dai_Nighttime_Smartphone_Reflective_Flare_Removal_Using_Optical_Center_Symmetry_Prior_CVPR_2023_paper.pdf
- MIPI 2024 challenge：https://openaccess.thecvf.com/content/CVPR2024W/MIPI/papers/Dai_MIPI_2024_Challenge_on_Nighttime_Flare_Removal_Methods_and_Results_CVPRW_2024_paper.pdf
- Glare detection 数据集（占位）：https://github.com/maesfahani/glaredetection ｜ https://arxiv.org/abs/2110.06006
- GLARE 交通标志：https://arxiv.org/abs/2209.08716
- 无参考光晕检测器：https://doi.org/10.3390/s26082359
- SHIQ / JSHDR：https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_A_Multi-Task_Network_for_Joint_Specular_Highlight_Detection_and_Removal_CVPR_2021_paper.pdf
