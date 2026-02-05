
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1 style="font-size: 4rem; font-weight: bold; color: #667eea; margin: 20px 0; display: flex; align-items: center; justify-content: center; gap: 20px;">
    <!-- <img src="assets/tsail_rdt.png" alt="TSAIL RDT" style="height: 8rem; width: auto;" /> -->
    RDT2: 通过扩展 UMI 数据实现零样本跨本体泛化
  </h1>
</div>
<!-- <hr> -->
<div align="center" style="line-height: 1;">
  <a href="https://rdt-robotics.github.io/rdt2/"><img alt="Homepage"
    src="https://img.shields.io/badge/RDT%202-Homepage-4287f5?logo=probot&logoColor=#009BD5"/></a>
  <a href="https://huggingface.co/collections/robotics-diffusion-transformer/rdt-2-68ce9ddbf7dc520a231220d5"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TSAIL%20RDT-ffc107?color=ffc107&logoColor=white"/></a>
  <!-- <br> -->
  <a href="https://discord.gg/vsZS3zmf9A"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-RDT-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <br>
<a href="https://rdt-robotics.github.io/rdt2/feishu.html"><img alt="Feishu"
    src="https://img.shields.io/badge/Feishu-RDT-blue?logo=lark&logoColor=white"/></a>
  <a href="https://x.com/songming_liu/status/1971643908372550108"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-RDT-white?logo=x&logoColor=white"/></a>
  <!-- <br>
  <a href="LICENSE"><img alt="License"
    src="https://img.shields.io/badge/License-Apache--2.0-f5de53?logo=apache&color=f5de53"/></a>
  <!-- <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL"><img alt="Model License"
    src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53"/></a> -->
  <!-- <br> -->
  <!-- <a href="https://arxiv.org/pdf/2412.19437"><b>Blog Link</b>👁️</a>  -->
  <a href="https://arxiv.org/abs/2602.03310"><img alt="Paper"
    src="https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv"/></a>
  <!-- <a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL"><img alt="Model License"
    src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53"/></a> -->
  <br>
  <!-- <a href=""><b>Paper Link</b>📄</a> -->
</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Updates](#updates)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Checkpoints](#model-checkpoints)
- [Running Inference for a Pre-Trained Model](#running-inference-for-a-pre-trained-model)
  - [1. \[IMPORTANT\] Hard-ware Set up and Calibration](#1-important-hard-ware-set-up-and-calibration)
  - [2. Run Inference](#2-run-inference)
- [Fine-Tuning Models on Your Own Data](#fine-tuning-models-on-your-own-data)
  - [1. Convert your data to WebDataset shards](#1-convert-your-data-to-webdataset-shards)
  - [2. Defining training configs and running training](#2-defining-training-configs-and-running-training)
  - [3. Run training](#3-run-training)
    - [RDT2-VQ](#rdt2-vq)
    - [RDT2-FM](#rdt2-fm)
  - [精度设置](#精度设置)
- [故障排除](#故障排除)

## 概述

RDT2 是 [RDT-1B](https://rdt-robotics.github.io/rdt-robotics/) 的续作，是首个能够在**未见过的机器人本体**上实现**零样本部署**的**简单开放词汇**任务（如抓取、放置、摇晃、擦拭等）的基础模型。这一里程碑的实现得益于多方面的努力：

- 我们通过采用更高强度的材料和更精确的跟踪方法重新设计了 [UMI 硬件](https://umi-gripper.github.io)，确保其在大规模数据收集中具有可靠性。
- 我们在**100+ 个不同的室内场景**中收集了**超过 10,000 小时**的人类操作视频，涵盖了夹爪可以执行的大部分家庭任务。

目前，本仓库包含以下模型：
- [RDT2-VQ](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ)：一个自动视觉-语言-动作模型（VLA），采用 [Residual VQ](https://arxiv.org/abs/2107.03312) 作为动作标记器，基于 [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) 并使用我们的 UMI 数据集进行适配，实现了卓越的零样本指令跟随能力。
- [RDT2-FM](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM)：一个改进的 RDT 模型，作为动作专家，采用流匹配（flow-matching）目标函数，推理延迟显著降低。

对于所有模型，我们提供了检查点和示例，可以直接使用或基于您自己的数据集进行微调。目前，我们已在包括 [双臂 UR5e](https://www.universal-robots.com/products/ur5e/) 和 [双臂 Franka Research 3](https://franka.de/franka-research-3) 在内的平台上验证了模型的有效性，我们乐观地认为，通过遵循我们的[指南](#运行预训练模型推理)，未来可以在更多平台上成功部署这些模型。


## 更新日志

- [Sept 2025] We released [RDT2-VQ](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ) \& [RDT2-FM](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM), the sequel of RDT-1B with better open-world generalization and zero-shot deployment on unseen embodiments.
- [Feb 2026] We released the [arXiv](https://arxiv.org/abs/2602.03310) paper.

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism or offload into CPU to reduce per-GPU memory. Since RDT2 is based on Qwen2.5-VL-7B, you basiclly need to follow the hard-ware requirements for Qwen2.5-VL-7B:

| Mode               | RAM Required | VRAM Required | Example GPU        |
| ------------------ | --------------- | --------------- | ------------------ |
| 推理          | > 32 GB      | ~ 16 GB | RTX 4090           |
| 微调 RDT2-FM (RDT 专家) |   -     | ~ 16 GB | RTX 4090           |
| 微调 RDT2-VQ (LoRA) |   -     | > 32 GB | A100 (40GB)           |
| 微调 RDT2-VQ (全参数) |   -    |  > 80 GB  | A100 (80GB) / H100 / B200|

对于零样本部署，您需要购买指定的*末端执行器*和*相机*，并根据[硬件设置与标定](#1-重要-硬件设置与标定)进行 3D 打印相应的*相机支架*和*法兰*。

本仓库已在 Ubuntu 24.04 上测试，我们目前不支持其他操作系统。

## 安装

克隆本仓库并创建 conda 环境：

```bash
# 克隆仓库
git clone https://github.com/thu-ml/RDT2.git
cd RDT2

# 创建 conda 环境
conda create -n rdt2 python=3.10 -y
conda activate rdt2

# 安装 torch (cuda12.8)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 安装 flash attention,可能报错缺少psutil包 直接pip install即可
pip install -U psutil
pip install flash-attn --no-build-isolation

# 安装其他依赖
pip install -r requirements.txt

# 升级 nvidia-nccl-cu12
pip install --upgrade --force-reinstall nvidia-nccl-cu12==2.27.5

# 再次确认已安装正确的 transformers 4.51.3
pip list | grep transformers

# 部署到 UR5e
pip install -r requirements/ur5e.txt

# 部署到 Franka Research 3
pip install -r requirements/franka_research_3.txt
```
下载对应的归一化文件
```bash
http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt
```
## 模型检查点

<!-- ###  Models -->
我们提供了多个 VLA 模型检查点，能够在各种机器人平台和简单词汇任务上部署。如果您想在自己的机器人平台上使用其他末端执行器和相机进行部署，可以从基础模型进行微调。


| 模型        | 使用场景    | 描述                                                                                                 | 检查点路径                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| normalizer      | Inference & Fine-Tuning (Freeze) | Normalizer for action normalization   | [umi_normalizer_wo_downsample_indentity_rot.pt](https://huggingface.co/robotics-diffusion-transformer/RVQActionTokenizer/blob/main/umi_normalizer_wo_downsample_indentity_rot.pt)    |
| Residual VQ  | Inference & Fine-Tuning (Freeze) |  Residual VQ (RVQ) as the action tokenizer   | [`robotics-diffusion-transformer/RVQActionTokenizer`](https://huggingface.co/robotics-diffusion-transformer/RVQActionTokenizer)    |
| RDT2-VQ      | Inference & Fine-Tuning | Auto-regressive VLA with Residual VQ as the action tokenizer   | [`robotics-diffusion-transformer/RDT2-VQ`](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ)    |
| RDT2-FM      | Inference & Fine-Tuning | Auto-regressive VLA (RDT2-VQ) with Flow-Matching Action Expert   | [`robotics-diffusion-transformer/RDT2-FM`](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM)    |

<!-- | $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
|| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      | -->

<!-- ### Fine-Tuned Models -->


<!-- | Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
|| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
|| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
|| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
|| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
|| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
|| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
|| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
|| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      | -->

## 运行预训练模型推理

### 1. [重要] 硬件设置与标定

1. 根据我们的[硬件指南](https://docs.google.com/document/d/1HUeM4Wlt4PyINoEwci-hxm8U9wAxiPMgR3sHyaOAsck/edit?tab=t.0#heading=h.sbdalb8w1kk1)获取部署硬件。

2. 设置机器人

- 2.1 设置 UR5e  
   - 获取 IP 地址并更新 [configs/robots/eval_bimanual_ur5e_config.yaml](configs/robots/eval_bimanual_ur5e_config.yaml) 中的 `robots/robot_ip`。  
  - 在安装 > 负载中  
    - 将质量设置为 0.82 kg  
    - 将惯性矩阵设置为  
      ```python
      [0.001106, 0, 0,
       0, 0.001106, 0,
       0, 0, 0.001106]
      ```
    - 将速度设置为 30%（推荐）
  
- 2.2 设置 Franka FR3  
  - 获取 IP 地址并更新 [configs/robots/eval_bimanual_fr3_config.yaml](configs/robots/eval_bimanual_fr3_config.yaml) 中的 `robots/robot_ip`。  
  - 在 Franka 界面网站上  
    - 将夹爪质量设置为 1.9 kg  
    - 将惯性张量设置为  
      ```python
      [0.001, 0, 0,
       0, 0.001, 0,
       0, 0, 0.001]
      ```

3. 设置相机
   * 从 [海康机器人网站](https://www.hikrobotics.com/cn/machinevision/service/download/?module=0) 下载 SDK 并安装所有 `.deb` 文件。
   * 运行 `cd /opt/MVS/bin && ./MVS.sh`。选择您的相机，并将采集控制 -> 曝光时间设置为 20000。
  
4. 将机器人标定到跟踪器的 TCP 空间
 * 按照[硬件指南](https://docs.google.com/document/d/1HUeM4Wlt4PyINoEwci-hxm8U9wAxiPMgR3sHyaOAsck/edit?tab=t.0#heading=h.sbdalb8w1kk1)中的标定设置说明进行操作。
 * 根据此[教程](https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEbrFK7b1OnJe54ltw/edit?tab=t.0#heading=h.yxlxo67jgfyx)设置 Vive Tracker -> 软件设置教程 -> VIVE tracker 设置
 * 运行以下代码将机器人 TCP 空间标定到跟踪器空间。
 * 重要提示：此脚本会使机器人执行小幅度的正弦运动；在运行脚本之前，请确保机器人处于安全位置，工作空间内没有障碍物。
    ```bash
    python deploy/calibration/calibrate_franka.py --franka_ip <your_franka_server_ip> --franka_port <your_franka_server_port> # 如果使用 Franka Research 3
    # 或者
    python deploy/calibration/calibrate_ur5e.py --ur5e_ip <your_ur5e_ip> # 如果使用 UR5e
    ```
  * 标定后，运行以下脚本获取标定矩阵：
    ```bash
    python deploy/calibration/compute_calibration_matrix.py
    ```
    然后将标定矩阵粘贴到 `eval_bimanual_ur5e_config.yaml` 的 `tx_tracker_to_tcp`（如果使用 FR3，则粘贴到 `eval_bimanual_fr3_config.yaml` 的 `tx_tracker_to_tcp`）。

### 2. 运行推理

我们的预训练模型检查点可以用几行代码运行（这里以我们的 [RDT2-VQ 模型](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ) 为例）：
```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vqvae import MultiVQVAE
from models.normalizer import LinearNormalizer
from utils import batch_predict_action

# 假设使用 gpu 0
device = "cuda:0"


processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "robotics-diffusion-transformer/RDT2-VQ",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device
).eval()
vae = MultiVQVAE.from_pretrained("robotics-diffusion-transformer/RVQActionTokenizer").eval()
vae = vae.to(device=device, dtype=torch.float32)

valid_action_id_length = (
    vae.pos_id_len + vae.rot_id_len + vae.grip_id_len
)
# TODO: 修改为您自己下载的归一化器路径
# 从 http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt 下载
normalizer = LinearNormalizer.from_pretrained("umi_normalizer_wo_downsample_indentity_rot.pt")  # 

result = batch_predict_action(
    model,
    processor,
    vae,
    normalizer,
    examples=[
        {
            "obs": {
                # 注意：遵循 UMI 的设置，camera0_rgb 用于右臂，camera1_rgb 用于左臂
                "camera0_rgb": ..., # 右臂 RGB 图像，np.ndarray 格式，形状为 (1, 384, 384, 3)，数据类型为 np.uint8
                "camera1_rgb": ..., # 左臂 RGB 图像，np.ndarray 格式，形状为 (1, 384, 384, 3)，数据类型为 np.uint8
            },
            "meta": {
                "num_camera": 2
            }
        },
        ...,    # 我们支持批量推理，因此您可以传递一个示例列表
    ],
    valid_action_id_length=valid_action_id_length,
    apply_jpeg_compression=True,
    # 由于模型主要使用 jpeg 图像进行训练，我们建议开启此选项以获得更好的性能
    instruction="Pick up the apple."
    # 我们建议使用格式为"动词 + 对象"的指令，首字母大写并以句号结尾
)

# 从示例 0 获取预测的动作
action_chunk = result["action_pred"][0] # torch.FloatTensor 格式，形状为 (24, 20)，数据类型为 torch.float32
# action_chunk (T, D)，其中 T=24，D=20
#   T=24：我们的 action_chunk 在 fps=30 下预测未来 0.8 秒，即 24 帧
#   D=20：遵循 UMI 的设置，我们从右到左预测双臂的动作
#   - [0-2]：右臂末端执行器位置 x, y, z（单位：米）
#   - [3-8]：右臂末端执行器旋转，6D 旋转表示
#   - [9]：右臂夹爪宽度（单位：米）
#   - [10-12]：左臂末端执行器位置 x, y, z（单位：米）
#   - [13-18]：左臂末端执行器旋转，6D 旋转表示
#   - [19]：左臂夹爪宽度（单位：米）

# 将夹爪宽度从 [0, 0.088] 重新缩放到 [0, 0.1]
for robot_idx in range(2):
    action_chunk[:, robot_idx * 10 + 9] = action_chunk[:, robot_idx * 10 + 9] / 0.088 * 0.1
```

您也可以使用以下代码测试 [RDT2-FM](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM)：
```python
# 在我们的仓库根目录下运行
import yaml

from models.rdt_inferencer import RDTInferencer


with open("configs/rdt/post_train.yaml", "r") as f:
  model_config = yaml.safe_load(f)

model = RDTInferencer(
  config=model_config,
  pretrained_path="robotics-diffusion-transformer/RDT2-FM",
  # TODO: 修改 `normalizer_path` 为您自己下载的归一化器路径
  # 从 http://ml.cs.tsinghua.edu.cn/~lingxuan/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt 下载
  normalizer_path="umi_normalizer_wo_downsample_indentity_rot.pt",  
  pretrained_vision_language_model_name_or_path="robotics-diffusion-transformer/RDT2-VQ", # 使用 RDT2-VQ 作为 VLM 骨干网络
  device="cuda:0",
  dtype=torch.bfloat16,
)

result = model.step(
    observations={
        'images': {
            'left_stereo': ..., # 左臂 RGB 图像，np.ndarray 格式，形状为 (384, 384, 3)，数据类型为 np.uint8
            'right_stereo': ..., # 右臂 RGB 图像，np.ndarray 格式，形状为 (384, 384, 3)，数据类型为 np.uint8
        },
        # 当前使用零输入当前状态
        # 保留输入接口以便未来微调
        'state': np.zeros(model_config["common"]["state_dim"]).astype(np.float32)
    },
    instruction="Pick up the apple." # 语言指令
    # 我们建议使用格式为"动词 + 对象"的指令，首字母大写并以句号结尾
)


# 相对动作块，np.ndarray 格式，形状为 (24, 20)，数据类型为 np.float32
# 格式与 RDT2-VQ 相同
action_chunk = result.detach().cpu().numpy()

# 将夹爪宽度从 [0, 0.088] 重新缩放到 [0, 0.1]
for robot_idx in range(2):
    action_chunk[:, robot_idx * 10 + 9] = action_chunk[:, robot_idx * 10 + 9] / 0.088 * 0.1
```

<!-- You can also test this out in the [example notebook](examples/inference.ipynb). -->

我们提供了在 [双臂 UR5e](examples/ur5e/README.md) 和 [双臂 Franka Research 3](examples/fr3/README.md) 机器人上运行预训练检查点推理的详细分步示例。

重要提示：如果在检查所有设置、配置和标定后，推理成功率仍然较低，您可以参考[部署技巧](./examples/DEPLOYMENT_TIPS.md)寻求帮助。

<!-- **Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate. -->

<!-- **Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details. -->


## 在您自己的数据上微调模型

我们将以在 [双臂 UR5e 示例数据集](https://huggingface.co/datasets/robotics-diffusion-transformer/BimanualUR5eExample) 上微调 RDT2 模型为例，说明如何在您自己的数据上微调基础模型。我们将解释三个步骤：
1. 将您的数据转换为 [webdataset](https://github.com/webdataset/webdataset) 分片（我们使用此格式进行训练以实现高效 IO）
2. 定义训练配置
3. 运行训练

### 1. 将数据转换为 WebDataset 分片

<!-- We provide example scripts for converting assumed data sturcture to a webdataset dataset in [`data/preprocess/robot`](data/preprocess/robot) with detailed [guidelines](data/preprocess/robot/README.md). You can easily modify it to convert your own data!  -->
您应该将数据转换为处理后的 webdataset 分片，具有以下结构：

```bash 
shard-000000.tar
├── 0.image.jpg   # 双目（左手腕相机 + 右手腕相机）RGB 图像，np.ndarray 格式，形状为 (384, 768, 3)，数据类型为 np.uint8
├── 0.action.npy  # 相对动作块，np.ndarray 格式，形状为 (24, 20)，数据类型为 np.float32
├── 0.action_token.npy # 对应的动作标记，np.ndarray 格式，形状为 (27,)，取值范围 0 到 1024，数据类型为 np.int16
├── 0.meta.json # 元数据，包括键 `sub_task_instruction_key`，用于从 `instructions.json` 中索引对应的指令
├── 1.image.jpg
├── 1.action.npy
├── 1.action_token.npy
├── 1.meta.json
├── ...
shard-000001.tar
shard-000002.tar
...
```

此外，我们在 Hugging Face 上提供了使用双臂 UR5e 收集的处理后的[示例数据](https://huggingface.co/datasets/robotics-diffusion-transformer/BimanualUR5eExample)。您可以下载并直接使用。

### 2. 定义训练配置并运行训练

按照 [`configs/datasets/example.yaml`](configs/datasets/example.yaml) 中的格式定义您的数据集配置
```yaml
# 在此定义您的数据集名称
name: <your_dataset_name> # 例如：bimanual/ur_example
type: single
shards_dir: <your_shards_dir> # 例如：/ssd/rdt2/bimanual_fold_cloth/shards 
kwargs:
  instruction_path: <your_instruction_path> # 例如：/ssd/rdt2/ur_example/instruction.json
  normalizer_path: <your_normalizer_path> # 例如：/ssd/rdt2/umi_normalizer_wo_downsample_indentity_rot.pt
```

对于提供的示例数据，其对应的配置在 [`configs/datasets/example.yaml`](configs/datasets/example.yaml) 中。请记住将 `<root_dir>` 和 `<path_to_normalizer>` 替换为您自己下载的路径。

### 3. 运行训练

#### RDT2-VQ

目前，我们支持以下微调方法：

- DeepSpeed 训练
- LoRA（低秩适应）训练

由于 RDT2-VQ 基于 Qwen2.5-VL，您可以自由应用其他技术（例如 fsdp、量化），遵循 Qwen2.5-VL 的微调实践。
我们提供了[全参数](scripts/finetune_full_param.sh)和 [LoRA](scripts/finetune_lora.sh) 微调的示例微调脚本，您可以直接使用这些脚本来启动自己的训练。

为了更好地理解，我们详细解释了使用示例数据的全参数微调脚本（[`scripts/finetune_full_param.sh`](scripts/finetune_full_param.sh)）的逐行说明：

```bash
# 在此定义您的环境设置
# 例如：nccl、网络、代理等

TASK="bimanual-ur5e-example"  # 在此定义您的任务名称
DATASET_CONFIG_PATH="configs/datasets/example.yaml"  # 在此定义您的数据集配置路径

export TOKENIZER_ID="Qwen/Qwen2.5-VL-7B-Instruct"
export VAE_ID="robotics-diffusion-transformer/RVQActionTokenizer" 
export MODEL_ID="robotics-diffusion-transformer/RDT2-VQ"
export OUTPUT_DIR="outputs/vqvla-sft-${TASK}" # 在此定义您的输出目录

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

accelerate launch main.py \
    --deepspeed="scripts/zero1.json" \  # DeepSpeed 配置文件，您可以修改为使用其他分片策略
    --tokenizer_name=$TOKENIZER_ID \
    --vae_name=$VAE_ID \
    --pretrained_model_name_or_path=$MODEL_ID \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=64 \
    --eval_batch_size=32 \
    --max_train_steps=10000 \ # 我们建议训练少于 5 个 epoch 以避免过拟合，
                              # 您应该根据数据估算步数并相应设置
    --eval_strategy="no" \
    --logging_steps=25 \
    --checkpoints_total_limit=20 \
    --checkpointing_step=1000 \
    --lr_scheduler="cosine" \
    --learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=16 \
    --gradient_checkpointing \
    --log_level="info" \
    --report_to="wandb" \
    --lr_warmup_steps=500 \
    --dataset=$DATASET_CONFIG_PATH \
    --image_corruption \ # 我们建议开启此选项以获得更好的视觉鲁棒性
    --use_default_collate_fn_for_eval
```

尽管我们的 RVQ 在手持夹爪数据和真实机器人数据之间都表现出高度的泛化性。如果您想使用我们的残差 VQ 作为动作标记器在您自己的数据上进行微调，
我们真诚地建议您首先检查数据的统计信息是否在我们残差 VQ 的范围内，然后测试数据的重建误差。

<!-- **Note:** We provide a [script]() for compute normalization statistics fo action normalization for bound violation check. This can be beneficial if you are fine-tuning to a new task on a robot.  -->

#### RDT2-FM

目前，我们支持使用 DeepSpeed 微调 RDT2-FM 的动作专家：我们提供了[全参数动作专家](scripts/finetune_rdt.sh)微调的示例微调脚本。在指定您自己的[数据集配置路径](scripts/finetune_rdt.sh#L20)并将[全参数动作专家](scripts/finetune_rdt.sh#L42)中的 `<repository-path>` 替换为您自己的仓库路径后，您可以直接运行此脚本来启动训练。

### 精度设置

不同模型有特定的精度设置：

**动作标记器（残差 VQ）：**

由于残差 VQ 的尺寸非常小，我们在训练和推理中都使用 `float32`。

**RDT VLM ([RDT2-VQ](https://huggingface.co/robotics-diffusion-transformer/RDT2-VQ))：**

遵循 Qwen2.5-VL，使用完整的 `bfloat16`（默认）。您可以遵循 [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) 的实践，通过应用混合精度或量化等技术来调整精度。

<!-- **RDT Action Expert ([RDT2-FM](robotics-diffusion-transformer/RDT2-FM) \& [RDT2-FM-UltraFast](robotics-diffusion-transformer/RDT2-FM-UltraFast)):** -->
**RDT 动作专家 ([RDT2-FM](robotics-diffusion-transformer/RDT2-FM))：**

在训练和推理中都使用完整的 `bfloat16`。

## 故障排除

我们将在此收集常见问题及其解决方案。如果您遇到问题，请先在此处查看。如果找不到解决方案，请在仓库上提交问题（请参阅[此处](CONTRIBUTING.md)了解指南）。

| 问题                                     | 解决方案                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
| 🚧 In progress 🚧 | 🚧 In progress 🚧 |

## Citation

If you find our work helpful, please cite us:

```bibtex
@misc{liu2026rdt2exploringscalinglimit,
      title={RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization}, 
      author={Songming Liu and Bangguo Li and Kai Ma and Lingxuan Wu and Hengkai Tan and Xiao Ouyang and Hang Su and Jun Zhu},
      year={2026},
      eprint={2602.03310},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.03310}, 
}
```
Thank you!
