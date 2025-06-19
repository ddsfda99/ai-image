# AI创意图像工坊：图像风格迁移与文生图系统实验报告
## 一、项目背景与目标
本项目旨在构建一个基于深度学习的“AI创意图像工坊”系统，集成文本生成图像（Text-to-Image）与图像风格迁移（Style Transfer）两大功能模块，面向艺术创作、个性化风格生成等应用场景。

我们使用稳定的基础生成模型（Stable Diffusion v1.5），并通过两种主流个性化微调方法（LoRA 与 Textual Inversion）实现特定风格的图像生成。同时，为风格迁移任务引入 AnimeGANv2 模型，实现对动漫风格图像的快速转换。
## 二、系统架构
整个项目采用前后端分离架构：
后端（Flask）：提供图像上传、风格迁移、文本生成图像接口，支持多种模型加载和推理。
* **前端（HTML + Tailwind CSS + JS）**：美观展示生成结果，用户可上传内容图和风格图或直接输入文本 Prompt。
* **模型与训练部分**：另设 `lora/` 与 `textual_inversion_training/` 模块用于模型微调。

## 三、基础模型与微调方法
### 3.1 Stable Diffusion v1.5模型
Stable Diffusion v1.5 是由 Stability AI 发布的一款开源的文本到图像生成模型，基于 Latent Diffusion Models（LDM）架构构建。
核心原理：
(1) 文本编码
用户输入一个文本prompt，模型使用CLIP Text Encoder将这段文字转换成一个语义向量表示。

(2) 潜空间建模
原始图像先通过VAE编码器压缩为一个低维的潜空间张量，模型不在像素空间处理，而是在这个潜空间中进行训练和生成。这样可以大幅减少显存与算力需求，提高训练和生成效率。

(3) 扩散过程
起点是纯噪声，模型通过UNet网络预测噪声成分，在每一步逐渐去噪，重复若干步，最终得到干净的潜空间图。

(4) 图像解码
将上述潜空间输入到VAE解码器中，输出为512*512的RGB图像。

在本项目中，使用 Stable Diffusion v1.5作为基础模型，进行个性化微调（使用 LoRA、Textual Inversion 等技术），以适应特定风格或任务需求。其开放的权重和 Hugging Face/Diffusers 兼容性，使其易于集成到 Web 服务或其他下游应用中。

模型下载链接 https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5

### 3.2 AnimeGAN 模型（自定义，在fine_tuning/training/animegan.py）
AnimeGAN是一款专为图像风格迁移设计的轻量级生成对抗网络（GAN），其核心任务是将输入的真实世界照片或图像转换为具有动漫风格的输出图像。与文本驱动的生成模型（Stable Diffusion）不同，AnimeGAN 是一种典型的图像到图像转换模型，强调结构保持和风格变化。
核心原理：
(1) 图像编码与解码结构（Encoder-Decoder）
AnimeGAN 的生成器采用卷积结构进行图像编码与重建，先通过多层卷积对图像进行下采样，提取语义特征；中间通过多个残差块（Residual Blocks）维持图像结构稳定；再通过反卷积层（ConvTranspose）上采样恢复到原图大小，从而生成动漫风格图像。

(2) 感知损失 + 对抗损失
AnimeGAN 的训练结合了对抗损失（用于保持图像的“真实感”）和感知损失（Perceptual Loss，用 VGG19 网络衡量生成图和原图的高层特征差异），从而保证生成图像不仅具备动漫的色彩和线条风格，同时保留了原图的结构信息。

(3) 判别器对抗优化
判别器用于判断一张图像是否为真实动漫图像，训练过程中与生成器形成博弈，推动生成器学习更加逼真的动漫风格映射。

### 3.3 LoRA（Low-Rank Adaptation）微调
#### 3.2.1 训练印象派（Monet）风格
基于 Stable Diffusion v1.5 模型，通过 LoRA微调方法学习印象派（Monet）风格的绘画特征。
数据集：从 WikiArt 数据集中提取的 Claude Monet 画作图像共计 1365 张，格式为 .jpg
模型与训练配置
| 项目           | 值                                                      |
| ------------ | ------------------------------------------------------ |
| 预训练模型路径      | `fine_tuning/models/stable-diffusion-v1-5` |
| 输出目录         | `fine_tuning/outputs/lora_monet`           |
| Prompt Token | `<monet-style>`                              |
| 图像分辨率        | 512 × 512                                              |
| 批大小          | 1                                                      |
| 训练步数         | 200                                                    |
| 精度设置         | `fp16`                                       |
| 启动命令         | `accelerate launch` 通过 shell 脚本自动执行                    |

训练日志显示：
训练步数：200
每步平均耗时约 0.5 秒，总时长约 1 分钟 30 秒
最终损失（loss）：约 0.273

#### 3.2.2 训练后印象派（Van Gogh）风格
基于 Stable Diffusion v1.5 模型，通过 LoRA 微调方法学习后印象派（Van Gogh）风格的绘画特征。
数据集：从 WikiArt 数据集中提取的 Vincent van Gogh 画作图像，共计 1929 张，格式为 .jpg
模型与训练配置
| 项目           | 值                                          |
| ------------ | ------------------------------------------ |
| 预训练模型路径      | `fine_tuning/models/stable-diffusion-v1-5` |
| 输出目录         | `fine_tuning/outputs/lora_vangogh`         |
| Prompt Token | `<vangogh-style>`                          |
| 图像分辨率        | 512 × 512                                  |
| 批大小          | 1                                          |
| 训练步数         | 200                                        |
| 精度设置         | `fp16`                                     |
| 启动命令         | `accelerate launch` 通过 shell 脚本自动执行        |

训练日志显示：
训练步数：200
每步平均耗时：约 0.5 秒，总时长约 1 分钟 30 秒
最终损失（loss）：约 0.115

#### 3.2.3 训练抽象表现派（Francis）风格
基于 Stable Diffusion v1.5 模型，通过 LoRA微调方法学习抽象表现派（Francis）风格的绘画特征。
数据集：从 WikiArt 数据集中提取的 Sam Francis 画作图像共计 370 张，格式为 .jpg
模型与训练配置：
| 项目           | 值                                          |
| ------------ | ------------------------------------------ |
| 预训练模型路径      | `fine_tuning/models/stable-diffusion-v1-5` |
| 输出目录         | `fine_tuning/outputs/lora_francis`         |
| Prompt Token | `<francis-style>`                          |
| 图像分辨率        | 512 × 512                                  |
| 批大小          | 1                                          |
| 训练步数         | 200                                        |
| 精度设置         | `fp16`                                     |
| 启动命令         | `accelerate launch` 通过 shell 脚本自动执行        |

训练日志显示：
训练步数：200
每步平均耗时约 0.5 秒，总时长约 1 分半
最终损失（loss）：约 0.234

#### 3.2.4 训练动漫风格
基于 AnimeGAN 模型，通过 LoRA 微调方法，学习动漫风格的图像转换特征。该过程通过kaggle动漫人物头像数据，完成了对现有预训练模型的风格学习与优化。
数据集：
从动漫图像数据集提取的图像（共计 500 张，格式为 .jpg），包括多种动漫角色与场景。
模型与训练配置：
| 项目     | 配置值                                                             |
| ------ | --------------------------------------------------------------- |
| 模型结构   | 自定义 CNN + 残差块 + ConvTranspose                                   |
| 损失函数   | 对抗损失 + 感知损失（VGG19）                                              |
| 数据集    | 500 张动漫风格图像，尺寸 512×512                                          |
| 批大小    | 4                                                               |
| 训练周期   | 5\~10 epoch（支持参数配置）                                             |
| 优化器    | Adam，学习率生成器为 2e-4，判别器为 4e-4                                     |
| 模型保存路径 | `outputs/animegan_anime/gen_final.pth`                          |
| 日志记录   | 每个 epoch 写入 `training_log.txt`，含 D/G loss 和学习率                  |
| 启动脚本   | `python training/animegan.py --dataset_dir trainsets/anime ...` |

训练日志显示：
训练步数：2050
总时长约5分钟
Loss_D：0.4993
Loss_G：1.2007
最终损失（loss）：1.7000

### 3.3 Textual Inversion微调(textual_inversion_training/textual_inversion.py)
Textual Inversion是一种轻量化的训练方式，让模型学会新的概念和风格，仅需少量数据和较短训练时间。
核心原理：
(1) 添加新token
引入一个新的占位词<my_style>，为它分配一个词向量(embedding)

(2) 训练该token的embedding向量
利用提供的图像数据作为训练集，使用

(3) 生成特定风格的图像
在prompt中使用<my_style>，模型会调用之前训练好的embedding来生成风格一致的图像。



## 五、模型比较
### 5.1 LoRA vs LoRA + DreamBooth vs Textual Inversion（LPIPS 指标）
图中展示了 LoRA 与 Textual Inversion（TI）在五个不同 prompt 下的 LPIPS 分数对比。LPIPS（Learned Perceptual Image Patch Similarity）是衡量生成图像与原图（base）之间感知差异的指标，分数越高代表图像在风格、细节等方面的变化越大，风格迁移效果越强烈。
![alt text](lpips.png)

从图中可以看出：

LoRA 模型整体风格迁移更强烈。在 prompt 1（田野小屋）与 prompt 5（沙漠仙人掌）上，LoRA 的 LPIPS 分数显著高于 TI，表明其引入的风格变化更丰富。

Textual Inversion 更注重内容保留。在 prompt 2～4（城市夜景、人物肖像、中世纪城堡）上，TI 的分数略高，说明其变化程度稍大，但整体趋近于 base 图，风格迁移较温和。

两者在风格迁移上有互补倾向：LoRA 适合生成具有强烈风格特征的图像，而 TI 更适合保持场景结构和语义一致性。

综合评估，LoRA 更适用于艺术风格迁移任务，如印象派或油画模拟；Textual Inversion 更适合内容增强或细节微调场景。实际部署中可以结合使用，根据需求动态切换或组合两者效果。

三类模型原理及优缺点对比分析

| 模型                    | 优点             | 缺点         | 适合研究什么？                |
| --------------------- | -------------- | ---------- | ---------------------- |
| **Textual Inversion** | 超轻量，训练快速       | 表达力弱，容易崩坏  | 小样本词向微调效果、embedding 能力 |
| **LoRA**              | 风格拟合快，泛化能力强    | 易过拟合、无类别约束 | 结构保持与风格表达的权衡           |
| **LoRA + DreamBooth** | 风格+结构平衡，生成质量最高 | 训练时间稍久     | 类别正则对生成效果的提升           |

#### 5.1.1 数据集选取
instance图像是风格图像，用于让模型学习特定风格。

class图像是希望保持的通用结构，不因风格化遗忘。

| 模型                | instance 图数量 | class 图数量 | 说明             |
| ----------------- | ------------ | --------- | -------------- |
| Textual Inversion | 20 张        | 无       | 太多图反而不稳定       |
| LoRA              | 20 张       | 无       | 快速收敛、适合风格微调    |
| LoRA + DreamBooth | 20 张       | 20 张    | 效果稳定、支持泛化与结构保持 |

## 七、总结与改进方向
### 成果：

* 成功集成多模型图像生成系统
* 展示效果良好，适合课程展示或用户交互

### 可改进：

* 引入自动风格分类器（提升风格检测能力）
* 加入用户上传风格图动态微调 Token
* 增加 Diffusion 模型生成过程可视化
