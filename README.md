# Object Binding Memory VLM · BindMem

评测视觉语言模型能否区分“见过颜色、纹理和形状”与“见过由它们共同构成的完整物体”。项目包含受控合成图像生成、严格重组负例、Qwen3-VL／LLaVA 推理、按随机组统计及可追溯的真实错误案例。

当前版本以 **BindMem v2 + 匹配重组梯度** 为主，只做行为实验。每题的学习图仍在当前上下文中，不测试跨会话长期记忆；探针相关代码仅作历史保留。

## 先看报告

- [完整项目报告](PROJECT_REPORT_COMPLETE.md)：实验目的、输入输出、全套 Qwen 成绩、LLaVA 同题比较、置信区间与真实误认。
- [图文简报](reports/bindmem_complete_20261029/brief.md)：面向不了解项目的读者。
- [逐任务成功率](reports/bindmem_complete_20261029/appendix_qwen_tasks.md)、[17 个实际输入输出案例](reports/bindmem_complete_20261029/appendix_input_output.md)。
- [14 张设计与结果图](reports/bindmem_complete_20261029/figures/)、[完整精度表格](reports/bindmem_complete_20261029/tables/)、[案例证据](reports/bindmem_complete_20261029/evidence/)。

GitHub 保存 Markdown、轻量 PNG/SVG、CSV/JSON 和小型案例网页。大型 HTML、PDF、ZIP 导出留在本地，不上传；本地完整版本的使用说明见[报告目录](reports/bindmem_complete_20261029/README.md)。

## 已完成的实验

| 模型／实验 | 正式 test 题数 | 范围 |
| --- | ---: | --- |
| Qwen3-VL-8B，原 v2 | 69,840 | L0–L5、全部问法；加 dev 共 93,120 题 |
| Qwen3-VL-8B，六物体梯度 | 5,760 | Q1/Q2/Q3，已见候选属性对数 k=0/1/2/3 |
| LLaVA-1.5-7B，核心比较 | 1,680 | 二维校准及 L2 的 Q1/Q2/Q3 |
| LLaVA-1.5-7B，六物体梯度 | 3,840 | Q1/Q3，k=0/1/2/3 |

LLaVA 的 5,520 题均与 Qwen 按相同 trial_id 比较，保留全部 240 个正式随机组；不代表 LLaVA 已完成全部 v2。全量 LLaVA 的部分输出只作探索性记录，范围缩减过程已在报告中披露。

Qwen 的基础识别和简单图片判断较好，但严格图片判断对强重组负例的误认率为 56.25%；匹配梯度的图片误认率随 k 从 0 增至 3，由 0% 上升至 39.17%。LLaVA 在当前多图图片判断协议下全答 YES，准确率 50% 来自接受偏向。完整解释需同时看目标命中、负例误认、基础识别与模型多图适配限制。

## 仓库结构

```text
bindmem/             v2 生成、渲染、验证、Qwen 推理与统计
bindmem_extra/       LLaVA 适配、匹配梯度、同题比较、真实案例
scripts/             Slurm、流程检查与报告生成脚本
tests/              CPU 协议、评分和断点复用测试
reports/             可分享的轻量报告、图表与证据
docs/               历史版本说明
binding_memory_integrated/   v1 数据与实现，历史保留
results/             v1 环境脚本与历史结果
models/              本地权重，不上传
data/                本地生成的 v2／梯度数据，不上传
runs/                本地原始回复、评分、采集数据，不上传
```

入口：[研究目标](goal.md) · [v2 运行说明](bindmem/README.md) · [脚本索引](scripts/README.md) · [补充实验设计](SUPPLEMENT_EXPERIMENTS.md) · [已完成精简实验](PRIORITY_EXPERIMENTS.md)。历史资料见 [v1 README](docs/README_V1.md)、[v1 报告](PROJECT_REPORT.md)。旧版本与 v2 的结果不混用。

## 环境

原计算节点使用 Python 3.11、torch 2.10.0+cu128、transformers 5.19.0、Pillow 12.3.0、NumPy 2.4.6、accelerate 1.15.0，单张 RTX 3090。**现有 `qwen3vl` 环境无需调整。**

CPU 生成／统计／测试的依赖见 [requirements-cpu.txt](requirements-cpu.txt)。新机器使用独立环境安装，GPU 环境按[现有安装流程](results/QWEN_LOCAL_SETUP.md)及[模型依赖](results/requirements-qwen.txt)配置；不要在已有可运行环境中盲目覆盖 PyTorch/CUDA。

```bash
git clone https://github.com/PKUSZY/Object-Binding-Memory-VLM.git
cd Object-Binding-Memory-VLM
conda activate qwen3vl
python -m unittest discover -s tests -p 'test_*.py' -v
```

报告依赖单独列在 [requirements-report.txt](requirements-report.txt)，可装在独立报告环境。完整报告重建需要本地全量运行结果和中文字体，见[脚本说明](scripts/README.md)；阅读已发布 Markdown 和图片无需安装依赖。

## 从零运行

下列命令从仓库根目录执行。生成和验证用 CPU；模型推理必须在已分配的 GPU 计算节点执行。先用小规模目录检查流程，正式生成使用新的目录，生成器拒绝覆盖非空目录。

```bash
# 小规模全流程输入检查，不加载模型
python -m bindmem.generate --output data/bindmem_v2_smoke --groups 8
python -m bindmem.validate --data data/bindmem_v2_smoke
python -m bindmem.run --data data/bindmem_v2_smoke --dry-run

# 正式数据：80 dev 组、240 test 组
python -m bindmem.generate --output data/bindmem_v2 --groups 320
python -m bindmem.validate --data data/bindmem_v2
python -m bindmem_extra.gradient --output data/bindmem_gradient_n6 --groups 320 --objects 6
python -m bindmem_extra.gradient --output data/bindmem_gradient_n6 --validate-only
```

准备 Qwen 本地权重至 `models/Qwen3-VL-8B-Instruct`，下载流程见[环境说明](results/QWEN_LOCAL_SETUP.md)。LLaVA 可在可联网的登录节点下载固定版本：

```bash
python scripts/prepare_llava.py \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --output models/llava-1.5-7b-hf
```

在 GPU 节点上，分别运行两个模型完全相同的严格绑定核心任务：

```bash
python -m bindmem.run --data data/bindmem_v2 \
  --model models/Qwen3-VL-8B-Instruct --output runs/fresh/qwen_strict \
  --split test --levels L2 --questions Q1 Q2 Q3
python -m bindmem_extra.inference --backend llava --data data/bindmem_v2 \
  --model models/llava-1.5-7b-hf --output runs/fresh/llava_strict \
  --split test --levels L2 --questions Q1 Q2 Q3
python -m bindmem_extra.score --data data/bindmem_v2 \
  --run-a runs/fresh/qwen_strict --run-b runs/fresh/llava_strict \
  --output runs/fresh/strict_comparison
```

这个最小例子是 1,440 道 L2 test，不包含正式核心比较中的 240 道校准题。完整覆盖与原目录续跑见运行文档；六物体梯度可对同一 `data/bindmem_gradient_n6` 使用 `bindmem_extra.inference` 的 `--backend qwen` / `--backend llava` 和 `--questions Q1 Q3`。

需要 Slurm 长期作业时，查看[脚本索引](scripts/README.md)。提交前核对本集群的 partition、qos、排除节点和 CUDA module；原 `bindmem_llava_priority.sbatch` 依赖已有 Qwen 结果与复用来源，不是新克隆的一键从零脚本。请用新输出目录，勿覆盖已完成运行。

## 数据与评分约定

输入与标签分离：模型只读取 `inputs/{dev,test}.jsonl`，标准答案、属性标签和题目类型不额外送入模型。每题独立、确定性生成、最多 80 个新 token；无效回答按错误计分。统计报告包含准确率、目标命中、误认、拒绝、成对正确率及按随机组 bootstrap 的 95% 区间，不跨任务混算一个总准确率。

运行目录记录数据、模型、依赖和代码指纹；相同命令支持断点续跑，配置变化会拒绝混写。探针不在当前实验范围内。四物体无法在固定候选单属性频数下覆盖完整 0–3 梯度，因此正式梯度使用六物体；原因与控制边界见完整报告。

本次上传不新增模型权重、生成数据、原始回复、缓存、Slurm 日志或大型报告附件。先前提交的 v1 文件保留，以免破坏历史复现路径。
