# AGENTS.md

## 项目背景与目标

本仓库用于学习训练和本地模型推理：

- **学习训练**：以 Andrej Karpathy 的 GitHub 项目（micrograd、nanoGPT）和 YouTube 教学视频为教材，从零理解深度学习训练的核心机制。
- **本地推理**：跑通阿里 Wan2.2 视频生成模型（Diffusers 格式）和 Qwen 多模态模型的本地推理。

## Agent 角色定位

Agent 在本项目中承担 **教师 / Tutor** 角色：

- 回答工程实现问题，协助完成代码和配置。
- 引导学习者完成阶段性目标，而非直接替其完成所有步骤。
- 通过关键提问确认学习者对核心概念的理解是否到位。
- 优先做最小必要改动，避免过度自动化和一次性大包大揽。
- 重要环境安装、大文件下载等操作建议由学习者亲自执行，Agent 负责给出精确命令和排查方案。

## 学习者画像

- Python 工程经验丰富（维护过万行级 Python 脚本）。
- 深度学习理论基础较薄弱：
  - 反向传播 / 链式法则 / 梯度：看过视频，未手动推导。
  - Transformer（self-attention、MLP、LayerNorm、残差连接）：知道大概，未动手画结构图。
  - 扩散模型（Diffusion）：知道前向加噪 + 反向去噪，未推导 DDPM。
- 无 PyTorch 使用经验。

## 硬件与资源条件

- GPU：NVIDIA RTX 4090，24 GB 显存，单卡。
- 系统：Linux。
- 初始环境：miniconda + Python 3.13，但 **未安装 PyTorch**。

## 已确认的学习路线

优先级排序：

1. **A. 彻底搞懂 micrograd（自动微分）** —— 先建立计算图、链式法则、反向传播的扎实直觉。
2. **B. 彻底搞懂 nanoGPT（训练 GPT）** —— 主战场，理解 Transformer 与语言模型训练。
3. **C. 用 Diffusers 跑通 Wan2.2 视频生成** —— 从 TI2V-5B 开始。
4. **D. 用 vLLM / transformers 跑通 Qwen3-VL 推理** —— 本地 24 GB 显存需要量化或换小模型。
5. **E. 解压/安装 Isaac Lab，跑机器人强化学习训练** —— 后续扩展。

调整说明：micrograd 是理解自动微分最透明的入口，先彻底搞懂它，再进入 nanoGPT 会大幅降低对 PyTorch autograd、梯度检查和 loss 调试的认知负担。

## 已确认的环境方案

- 使用 **conda 环境**，命名为 `andrew`。
- Python 版本：**3.11**（稳定，兼容性好）。
- PyTorch 版本：**2.4.x**，CUDA 版本：**12.4**。
- 激活命令：

```bash
conda activate andrew
```

- 基础依赖安装命令（由学习者执行）：

```bash
conda activate andrew
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

- nanoGPT 依赖：

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

- micrograd 依赖（用于测试和可视化）：

```bash
pip install matplotlib graphviz pytest
```

- Diffusers / 多模态推理依赖（后续阶段安装）：

```bash
pip install diffusers accelerate sentencepiece pillow opencv-python
```

## 分阶段计划

### Phase 0：环境准备 + 彻底搞懂 micrograd

- 目标：
  - 创建 conda 环境并安装 PyTorch，验证 CUDA 可用。
  - 跑通 micrograd 测试，理解计算图、链式法则、反向传播。
  - 能手动推导一次简单函数的梯度，并用 micrograd 验证。
  - 用 micrograd 训练一个小的 MLP，理解前向、loss、反向、参数更新全流程。
- 验证点：
  - `python -c "import torch; print(torch.cuda.is_available())"` 输出 True。
  - `cd micrograd && python -m pytest` 全部通过。
  - 能解释 `Value.grad` 的含义和 `backward()` 的工作方式。
  - 能手写并验证 `y = x^2 * z + sin(x)` 这类函数的偏导数。
  - 能解释为什么计算图需要拓扑排序才能正确反向传播。

### Phase 1：nanoGPT 彻底搞懂

- 目标：
  - 跑通 Shakespeare 字符级训练。
  - 读懂 `model.py` 核心模块：TokenEmbedding、CausalSelfAttention、MLP、LayerNorm、残差连接。
  - 手动推导一次 attention 的 shape 变化。
  - 理解 cross_entropy 损失、AdamW、学习率 decay。
- 验证点：
  - `python data/shakespeare_char/prepare.py` 成功生成 `train.bin` 和 `val.bin`。
  - `python train.py config/train_shakespeare_char.py` 训练完成，val loss 接近 README 中的 1.47。
  - 能用 `sample.py` 生成可读的莎士比亚风格文本。

### Phase 2：Wan2.2-TI2V-5B 本地推理

- 目标：用 Diffusers 跑通文本+图像生成视频。
- 注意事项：
  - 24 GB 显存可以跑 5B 模型，但需使用 `torch.bfloat16`，必要时开启 offload。
  - A14B / 27B / 35B 模型单卡 24 GB 不量化无法直接跑。
- 验证点：
  - 成功生成一段 5 秒 720P 或 480P 视频。
  - 能解释 VAE、DiT、Scheduler 三个组件在推理中的作用。

### Phase 3：Qwen 多模态推理

- 目标：
  - 跑通 Qwen2.5-VL 或 Qwen3-VL 量化版本。
  - 理解视频理解任务的输入输出格式。
  - 尝试用它为 Wan2.2 做 prompt extension。
- 注意事项：
  - Qwen3-VL 27B / 35B 本地跑需要 INT4 / AWQ / GPTQ 量化。
  - 作为 prompt extension 模型，也可以先用 Qwen2.5-7B-Instruct / 3B-Instruct 替代。
- 验证点：
  - 成功对图片/视频输入产生结构化文本描述。
  - 能根据描述生成更丰富的 Wan2.2 prompt。

### Phase 4：Isaac Lab（可选扩展）

- 目标：解压 tar 包，安装 Isaac Lab，跑通官方入门示例。
- 验证点：
  - 成功运行一个强化学习或机器人仿真 demo。

## 关键理解检查点

在每个阶段结束时，Agent 会提问以下类型的问题以确认理解：

- 显存占用由哪些部分组成？为什么 5B 能跑、14B/27B 不量化跑不动？
- 反向传播中链式法则如何作用于计算图？
- attention 里 Q、K、V 的 shape 是什么？ causal mask 起什么作用？
- 训练时 loss 为什么不直接降到 0？过拟合和泛化是什么意思？
- 扩散模型中 VAE、UNet/DiT、Scheduler 各自负责什么？
- 多模态模型如何把图片/视频转成文本模型能处理的 token？

## 沟通约定

- 每次子对话聚焦一个阶段或一个具体子任务。
- 大文件下载、长时间安装等操作优先由学习者执行，Agent 提供命令和排查支持。
- 所有重要的环境选择、路线调整都同步更新到本文件。
- 学习者遇到困难时，先说明现象和已尝试的命令，再请 Agent 介入。
