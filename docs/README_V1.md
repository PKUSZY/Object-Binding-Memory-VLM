# v1 历史说明

本页保留早期版本的结果与复现步骤，不代表 v2 或 LLaVA 的当前成绩。新版入口见[根 README](../README.md)。

## 历史结果

正式模型为 `Qwen3-VL-8B-Instruct`，使用单张 NVIDIA RTX 3090、确定性推理（`do_sample=False`）。主实验、校准、双图控制与文字查询已运行完成；强重组控制中 5 题因缺失测试图片路径而失败。

| 轨道 | 规模 | 结果 |
|---|---:|---:|
| 主图片探测 | 864 题 | 837/864 = **96.88%** |
| 感知校准 | 432 图 | 颜色、形状、纹理和联合准确率均 **100%** |
| 双图属性匹配 | 2,150 题 | **100%** |
| 强重组诱饵 | 27 题 | 旧汇总 2/27 = **7.41%**；成功执行 22 题中 2 题正确（9.09%），另 5 题为输入路径错误 |
| 图片学习、文字探测 | 864 题 | 829/864 = **95.95%** |

主实验的主要错误来自 `critical_recombination`：模型有时会把三个熟悉属性误判为一个曾经出现过的完整物体。文字探测条件的 target 命中率为 92.36%，但 critical recombination 正确拒绝率提高到 98.61%。这些轨道分别统计，不能合并成一个总分。

完整方法和解释边界见 [`PROJECT_REPORT.md`](../PROJECT_REPORT.md)，执行清单见 [`PROJECT_GUIDE.md`](../PROJECT_GUIDE.md)。

## 仓库结构

```text
.
├── binding_memory_integrated/       # 可重建的数据集
│   ├── core/                         # 主实验：864 题
│   └── auxiliary/three_study_v0.1/   # 独立补充基准：1,750 题
├── results/                          # runner、评分器、Slurm 脚本和已汇总结果
├── PROJECT_REPORT.md                 # 面向读者的项目报告
├── PROJECT_GUIDE.md                  # 资源与执行说明
└── README.md
```

主数据集 `core/` 包含 6 种颜色、6 种形状、6 种纹理，24 个随机种子，学习图数量 4/8/12，干扰图数量 0/4。共有 144 个条件组和 864 道题，dev 216、test 648，正负样本各 432。图片文件名使用无语义哈希；标准答案位于 `labels/`，不能发送给模型。

## 1. 获取与验证数据集

需要 Node.js 18 或更高版本。进入数据集目录后安装锁定依赖并验证结构：

```bash
cd binding_memory_integrated
npm ci
npm run validate
```

如需从生成器重建主轨道：

```bash
npm run build
npm run validate
```

`validate` 会检查题目数量、图片格式、输入字段隔离、标签逻辑和诱饵约束。生成器重建前建议备份已有数据；模型运行时只读取 `core/inputs/{dev,test}.jsonl`，不要把 `trials.jsonl`、`groups.jsonl` 或 `labels/` 发送给模型。

## 2. 安装 Qwen 环境

推理脚本需要 Python 3.11、CUDA 版 PyTorch、Transformers、Accelerate、Qwen 视觉工具和 Pillow。集群用户应先申请 GPU 节点，再加载 CUDA 模块：

```bash
module load cuda12.8
conda create -n qwen3vl python=3.11 -y
conda activate qwen3vl
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r results/requirements-qwen.txt
```

复现实验使用的版本为 Python 3.11.17、PyTorch 2.10.0+cu128、Transformers 5.19.0、Accelerate 1.15.0、`qwen-vl-utils` 0.0.14、Pillow 12.3.0 和 Safetensors 0.8.0。PyTorch/CUDA wheel 需根据目标 GPU 与集群驱动调整；其他 Python 依赖已在 requirements 文件中固定。

也可以使用 [`results/QWEN_LOCAL_SETUP.md`](../results/QWEN_LOCAL_SETUP.md) 中的完整安装和 CUDA 排查说明。安装后必须确认：

```bash
nvidia-smi -L
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`torch.cuda.is_available()` 必须为 `True`。模型下载需要较大的磁盘空间，模型权重不会提交到 Git 仓库：

```bash
python results/download_qwen3vl.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output models/Qwen3-VL-8B-Instruct
```

如需使用国内 Hugging Face 镜像，可在下载前按集群规定设置 `HF_ENDPOINT`；不要把访问令牌写入脚本或提交到仓库。

## 3. 运行实验

### 主图片探测

本地直接运行：

```bash
python results/run_qwen_local_experiment.py \
  --model models/Qwen3-VL-8B-Instruct \
  --split both
python results/score_experiment.py
```

Slurm 运行完整主实验：

```bash
sbatch results/run_qwen_binding.sbatch
```

### 感知校准

校准单独评估颜色、形状和纹理识别，不参与主实验分数：

```bash
sbatch results/run_qwen_calibration.sbatch
```

### 双图和强重组控制

控制实验用于区分基础视觉/直接比较能力与完整绑定能力：

```bash
sbatch results/run_qwen_controls.sbatch
```

### 图片学习、文字探测

该条件使用同样的学习图片和干扰图片，但最后不发送测试图片，改用文字询问颜色、形状和纹理组合：

```bash
sbatch results/run_qwen_text_query.sbatch
```

所有 runner 都会逐题保存结果，并根据已有 `trial_id` 断点续跑。可用以下命令查看作业：

```bash
squeue -u "$USER"
tail -f slurm-qwen-binding-<JOBID>.out
```

## 4. 查看结果

已完成的汇总文件包括：

- [`results/experiment_scored/summary.json`](../results/experiment_scored/summary.json)：主图片探测；
- [`results/calibration_review/qwen_local_scored/calibration_summary.json`](../results/calibration_review/qwen_local_scored/calibration_summary.json)：感知校准；
- [`results/control_benchmark/summary.json`](../results/control_benchmark/summary.json)：双图和强重组控制；
- [`results/text_query_benchmark/scored/summary.json`](../results/text_query_benchmark/scored/summary.json)：文字探测；
- [`PROJECT_REPORT.md`](../PROJECT_REPORT.md)：面向他人的完整说明。

评分规则是：只有严格的 `YES` 或 `NO` 才算有效；无效输出按错误计入准确率。主实验需要同时报告总体准确率、target 命中率和各类诱饵错误接受率。置信区间应按 seed 或学习列表整组重采样，因为同一学习列表中的多道题共享学习图片。

## 5. 复现注意事项

- 这是上下文内绑定识别实验：学习图片保留在当前题目的上下文中，不能直接解释为跨会话长期记忆。
- 主实验、三学习图补充基准、校准、控制和文字探测是不同轨道，不能直接合并统计。
- `models/`、Slurm 日志、临时进度流和缓存文件默认被 Git 忽略；逐题原始回答与汇总结果保留，用于复查和重新评分。提交前仍应确认没有包含模型权重、API token 或个人路径。
- 当前结果是单模型、单次固定推理设置的结果。模型间比较和按 seed 的置信区间属于后续工作。

## 引用与复用

如果使用本仓库，请引用项目报告并注明模型版本、数据集版本、推理参数和运行硬件。数据集生成代码、验证器、推理 runner 和评分器均保留在仓库中，便于复查实验协议。
