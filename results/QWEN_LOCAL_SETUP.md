# NVIDIA GPU 运行 Qwen3-VL-8B-Instruct

这份指南用于在**能直接访问 NVIDIA GPU 的 Linux 环境**中安装 Qwen3-VL 环境。项目已在 RTX 3090 上完成正式运行，也可在配置合适驱动和 PyTorch CUDA wheel 的 RTX 5090 上运行。模型通过 Transformers 本地加载，不需要 API endpoint 或 API key。

Qwen3-VL 官方 Transformers 示例要求 Transformers 4.57.0 或更新版本，使用 `AutoModelForImageTextToText` 与 `AutoProcessor`。RTX 5090 属于 Blackwell 显卡，需要支持 Blackwell 的较新 NVIDIA 驱动和 CUDA wheel；RTX 3090 属于 Ampere 架构。PyTorch wheel 自带 CUDA runtime；通常不需要另装完整 CUDA Toolkit，但必须有正常工作的 NVIDIA 驱动。

参考：

- [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-8B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [PyTorch 本地安装选择器](https://pytorch.org/get-started/locally/)
- [Transformers Qwen3-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)

## 1. 确认你在有 GPU 的机器上

在该机器终端中运行：

```bash
nvidia-smi
```

确认输出列出分配给当前任务的 GPU，例如 `GeForce RTX 3090` 或 `GeForce RTX 5090`。如果是在容器或远程作业中运行，作业环境也必须能看到 GPU；宿主机能看到 GPU 不代表当前容器或登录节点自动获得了 GPU。

如果你所在集群使用 Environment Modules，先在**实际分配到 GPU 的节点**执行：

```bash
module load cuda12.8
nvidia-smi
```

`module load cuda12.8` 只加载 CUDA Toolkit。它不会安装/启动 NVIDIA 驱动，也不会为没有 GPU 的登录节点提供 GPU。若 `nvidia-smi` 失败，应申请带 GPU 的计算节点或修复 GPU passthrough；此时继续重装 Python 包没有作用。

## 2. 创建独立 Python 环境

下面使用 Conda 和 Python 3.11：

```bash
conda create -n qwen3vl python=3.11 -y
conda activate qwen3vl
python -m pip install --upgrade pip
```

也可用 Python 3.10 或 3.12；首次安装建议采用 Python 3.11，减少新旧包的兼容性变量。

## 3. 安装 PyTorch CUDA wheel

请根据 GPU 架构和集群驱动选择兼容的 PyTorch/CUDA 组合。优先打开 PyTorch 官方安装选择器，选择 Linux、Pip、Python、CUDA 12.8 或网站提供的更新 CUDA 版本，并复制当时显示的命令：

```text
https://pytorch.org/get-started/locally/
```

如果该选择器仍提供 CUDA 12.8 wheel，可使用：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

不要安装 CPU-only 的 PyTorch。安装后检查：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU capability:", torch.cuda.get_device_capability(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
```

必须看到 `CUDA available: True` 和分配的 GPU 型号。若为 `False`，先解决驱动、容器 GPU 映射或 PyTorch wheel 问题，不要继续下载模型。

## 4. 安装 Qwen3-VL 依赖

```bash
python -m pip install "transformers>=4.57.0" accelerate qwen-vl-utils pillow safetensors
```

验证版本：

```bash
python - <<'PY'
import torch, transformers
from transformers import AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available())
print("Qwen model loader imports: OK")
print("Qwen vision utility imports: OK")
PY
```

## 5. 下载并加载模型做冒烟测试

模型 ID：`Qwen/Qwen3-VL-8B-Instruct`。下载需要数 GB 到十余 GB 的存储空间，实际容量以模型仓库当前文件为准。可以将 Hugging Face 缓存放到空间充足的磁盘：

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

项目中已经提供安装和下载脚本。安装环境：

```bash
QWEN_CUDA_MODULE=cuda12.8 bash results/setup_qwen3vl.sh
```

下载到项目的 `models/` 目录（支持断点续传）：

```bash
python results/download_qwen3vl.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output models/Qwen3-VL-8B-Instruct
```

如果模型仓库要求登录，先运行 `huggingface-cli login`，或设置 `HF_TOKEN` 环境变量；不要把 token 写进脚本或提交到 Git。

从 `class_project` 目录运行下面的加载检查；首次运行会下载模型：

```bash
python - <<'PY'
import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as ModelClass

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = ModelClass.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()
print("Loaded:", model_id)
print("GPU:", torch.cuda.get_device_name(0))
print("Device map:", model.hf_device_map)
PY
```

也可以直接运行项目提供的单图测试：

```bash
python results/test_qwen3vl.py \
  --model models/Qwen3-VL-8B-Instruct
```

## 6. 跑 12 张校准 pilot

在 `class_project` 目录运行：

```bash
QWEN_CUDA_MODULE=cuda12.8 bash results/run_qwen_pipeline.sh
```

pipeline 会自动激活 `qwen3vl` Conda 环境，避免误用系统中的 CPU-only Python。若只想检查 GPU：

```bash
QWEN_CUDA_MODULE=cuda12.8 bash results/check_qwen_gpu.sh
```

若模型已经下载到项目目录，显式传入本地路径，避免再次从网络查找：

```bash
python results/run_qwen_local_calibration.py \
  --model models/Qwen3-VL-8B-Instruct \
  --limit 12
```

如果脚本输出 `nvidia-smi` 或 `CUDA is not visible`，先不要重复下载模型。请确认 `nvidia-smi` 能列出分配给作业的 GPU，并确认当前终端/容器确实获得 GPU passthrough。脚本会自动激活 `qwen3vl` 环境；CUDA 模块则通过 `QWEN_CUDA_MODULE` 显式加载。

例如，若日志显示 PyTorch 已安装 CUDA 12.8 wheel，但 `torch.cuda.is_available()` 为 `False`，通常需要确认 GPU 是否分配给当前作业，以及驱动设备是否暴露。请在同一个 shell 运行：

```bash
module load cuda12.8
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
ls -l /dev/nvidia* 2>/dev/null || true
```

- `nvidia-smi -L` 失败或没有 `/dev/nvidia*`：当前是登录节点、未申请 GPU 的作业，或容器没有 GPU passthrough。需要通过集群调度器申请 GPU 节点，然后在该作业内运行 pipeline。
- `nvidia-smi -L` 能列出 GPU，但 PyTorch 仍为 False：在同一 shell 激活 `qwen3vl`，再运行 `python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'`；这时才继续排查驱动版本或 `LD_LIBRARY_PATH` 冲突。

你也可以运行项目诊断脚本：

```bash
bash results/diagnose_qwen_cuda.sh
```

重点看三项：

1. 驱动版本。CUDA 12.8 GA 通常要求 Linux NVIDIA 驱动至少为 `570.26`；如果版本更低，需要管理员升级驱动，或使用集群提供的兼容 CUDA/PyTorch 组合。[NVIDIA CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
2. `CUDA_VISIBLE_DEVICES`。如果它是空值，当前作业把 GPU 对进程屏蔽了；如果是 `0`，应允许看到第一张卡。
3. `LD_LIBRARY_PATH`。如果 `env -u LD_LIBRARY_PATH python ...` 能看到 CUDA，而普通 Python 看不到，说明 `cuda12.8` module 的动态库路径冲突，尤其要检查路径中是否包含 `.../stubs`。此时可先用：

```bash
env -u LD_LIBRARY_PATH python results/test_qwen3vl.py --model models/Qwen3-VL-8B-Instruct
```

不要把 `libcuda.so` 的 stub 路径加入运行时 `LD_LIBRARY_PATH`；PyTorch 需要使用 NVIDIA 驱动提供的真实 `libcuda.so`。

脚本会把 12 张图均匀分布在 432 张清单中，每张图片单独推理，答案标签和图片路径不发给模型。默认输出为：

```text
results/calibration_review/qwen_local_pilot_raw.jsonl
```

查看 pilot：

```bash
python results/score_calibration.py \
  --raw results/calibration_review/qwen_local_pilot_raw.jsonl \
  --allow-subset
```

重点确认模型加载正常、返回字段能被解析、没有 OOM 或格式错误。12 张仅用于验证安装与数据流，样本太少，不能作为模型识别能力的正式结论。

## 7. 跑完整 432 张

Pilot 运行正常后执行：

```bash
python results/run_qwen_local_calibration.py \
  --model models/Qwen3-VL-8B-Instruct \
  --limit 432 \
  --output results/calibration_review/qwen_local_calibration_raw.jsonl
```

完整判分：

```bash
python results/score_calibration.py \
  --raw results/calibration_review/qwen_local_calibration_raw.jsonl \
  --out results/calibration_review/qwen_local_scored
```

输出会包括属性准确率、联合准确率、混淆信息和错误清单。正式实验也应使用同一模型 ID 与固定推理设置。

## 常见问题

- **`torch.cuda.is_available()` 为 False：** 先运行 `bash results/check_qwen_gpu.sh`。如果 `nvidia-smi` 也失败，这是 NVIDIA 驱动或容器 GPU passthrough 问题；如果 `nvidia-smi` 正常但 PyTorch 为 False，激活 `qwen3vl` 后重新安装 CUDA 版 PyTorch。
- **`no kernel image` 或不支持 Blackwell：** PyTorch/CUDA wheel 太旧，升级驱动并按 PyTorch 官方选择器安装更新 CUDA wheel。
- **显存不足：** 关闭同卡其他任务，先确认加载的是 8B Instruct；不要同时运行其他大模型。脚本按单张图顺序推理以控制显存占用。
- **模型下载需登录：** 若 Hugging Face 要求接受条款或登录，按模型卡提示登录。不要把访问令牌写入项目文件。
- **工作容器看不到 GPU：** 在已分配 GPU 的节点上运行，或为远程容器配置 GPU passthrough；容器必须能够访问 NVIDIA 驱动设备。
