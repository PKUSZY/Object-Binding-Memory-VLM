# BindMem v2：计算节点运行说明

2026-10-29：Qwen 原基准与六物体梯度、LLaVA 精简核心及梯度实验均已完成。新克隆先看[根 README](../README.md)；本页保留原集群的完整命令与排障记录，历史阶段的“当前／后续”以日期为准。`data/` 和 `runs/` 的大型本地文件不随 GitHub 发布，报告图表在 `reports/`。

本实现对应修改后的 [`goal.md`](../goal.md)，覆盖 L0–L5、Q1–Q8、行为评分。历史代码与实验结果继续保留；新版使用独立的 `data/`、`runs/`，不能把旧结果当作新版成绩。现状核对见 [`STATUS.md`](STATUS.md)。

当前范围仅包含行为评测；不再运行探针或采集隐藏状态，已有相关文件保留供历史复查。

## 环境

**沿用已有 `qwen3vl` 环境，不要求新增包或升级。** 已核对本机：Python 3.11、torch 2.10.0+cu128、transformers 5.19.0、Pillow 12.3.0、NumPy 2.4.6；其余依赖沿用 `results/requirements-qwen.txt`。新增代码用 NumPy 做统计、PyTorch 做模型推理、Pillow 和 NumPy 做程序化渲染，图表输出 SVG。无需 Blender、OpenGL、scikit-learn、SciPy 或 Matplotlib。

```bash
cd /ceph/home/shizhongyu/class_project
conda activate qwen3vl
# CPU 轻量检查，不读取 8B 模型权重，不占用 GPU
bash scripts/check_bindmem.sh
```

GPU 推理继续采用现有 `cuda12.8`、3090 分区和单卡配置。若集群分区或时间限制不同，修改 `scripts/bindmem_gpu.sbatch` 的 `#SBATCH` 行即可。

## 建议执行顺序

以下命令都从仓库根目录运行。各规模使用不同目录；生成器拒绝覆盖非空目录。

### 1. 先做小规模流程检查

在 CPU 计算节点生成 8 组，覆盖所有任务，共 2,328 题；其中 dev 2 组、test 6 组。这个规模只用于查错，不用于正式统计结论。

```bash
python -m bindmem.generate --output data/bindmem_v2_smoke --groups 8
python -m bindmem.validate --data data/bindmem_v2_smoke
python -m bindmem.run --data data/bindmem_v2_smoke --dry-run
```

在已分配 GPU 的交互计算节点上先运行少量题目：

```bash
python -m bindmem.run \
  --data data/bindmem_v2_smoke \
  --model models/Qwen3-VL-8B-Instruct \
  --output runs/bindmem_v2_smoke/qwen8b_behavior \
  --levels L0 L2 L3 --questions CAL Q2 \
  --limit 12
```

去掉 `--limit 12` 并保持其他参数不变，即续跑该选择范围的所有题。`--limit` 限制**本次待运行数**。需要扩大层级或问法时用新输出目录，避免混用实验设置。

### 2. 生成正式数据

默认 320 个独立随机组，80 dev / 240 test，每个严格反事实条件在 test 中有 240 对。完整数据共 93,120 题；可先用 `--groups 32` 生成独立 pilot（9,312 题）估计耗时。生成工作在 CPU 上完成。

```bash
python -m bindmem.generate --output data/bindmem_v2 --groups 320
python -m bindmem.validate --data data/bindmem_v2
```

生成器提供 `--seed`、`--levels`、`--questions`；若选择子集，该目录只包含子集，不能在同一目录追加。推荐完整生成一次，然后在推理时筛选。8 张学习图和多物体图都会增加推理资源消耗；实际耗时以 pilot 的逐题日志为准。

### 3. 提交 GPU 作业

全新运行可先做校准及核心 Q2 严格反事实与负载实验（共 6,400 题，包括 dev）。当前已有的 `qwen8b_core` 已完成，无需重跑；以下使用新的纯行为输出目录：

```bash
sbatch --export=ALL,STAGE=run,LEVELS='L0 L2 L3',QUESTIONS='CAL Q2',CAPTURE=0,RUN_DIR=runs/bindmem_v2/qwen8b_core_behavior \
  scripts/bindmem_gpu.sbatch
```

运行全套行为实验（93,120 题），使用另一个目录：

```bash
sbatch --export=ALL,STAGE=run,CAPTURE=0,RUN_DIR=runs/bindmem_v2/qwen8b_full \
  scripts/bindmem_gpu.sbatch
```

后续仅运行行为评测：直接调用 runner 时不传 `--capture`；通用提交脚本使用 `CAPTURE=0`。Q1 和 Q4/Q5 专用脚本已不启用表示采集。已有带 capture 的运行结果仍可用于行为评分，不需要重新推理，也不要改变其原目录的运行配置。

默认 `STAGE=all` 会在推理结束后评分；上面示例的 `STAGE=run` 仅执行推理。作业达到 48 小时时限时，重新提交**相同命令**即可续跑。已完成题按 ID 跳过；失败题重试；无效模型回答保留并按错误计分。一次运行错误会立即终止，避免生成大量错误记录。重复写同一目录的并发作业会被文件锁拒绝。

也可以在已分配的交互 GPU 节点运行：

```bash
DATA_DIR=data/bindmem_v2 RUN_DIR=runs/bindmem_v2/qwen8b_full bash scripts/run_bindmem.sh all
```

`all` 执行生成/验证、推理、评分；已有数据不会重建。`STAGE=run` 仅推理，`STAGE=score` 仅评分。如果采用上面的 `STAGE=run` 提交命令，完成后需要执行下一节评分命令。

### 4. 评分与对照

图片查询 Q1 也提供专用的 48 小时批处理脚本，可接续已有
`runs/bindmem_v2/qwen8b_image` 结果，并在推理完成后自动评分 dev/test：

```bash
sbatch scripts/bindmem_image.sbatch
```

该脚本固定运行 L2/L3 的 Q1（共 5,120 题），默认暂避此前失败的
`node001`、`node015`，由 Slurm 在分区内调度。若希望固定到已通过
CUDA 诊断的 `node010`，可提交 `sbatch --nodelist=node010 scripts/bindmem_image.sbatch`。
启动时检查 CUDA 初始化与一次小型 GPU 运算；只有清除 `LD_LIBRARY_PATH`
的独立检查成功时才对当前作业采用该回退，不修改 Conda 环境或 shell 配置。
日志为 `slurm-bindmem-image-<JOBID>.out/.err`。提交后可以关闭终端；
若遇时间限制或中断，重新提交同一命令即可续跑。不要同时在交互终端运行
同一输出目录，文件锁会拒绝并发写入。

```bash
python -m bindmem.score --data data/bindmem_v2 --run runs/bindmem_v2/qwen8b_full
# 另报 dev，不把它混入正式 test
python -m bindmem.score --data data/bindmem_v2 --run runs/bindmem_v2/qwen8b_full --split dev
```

输出在运行目录的 `scored/test/` 或 `scored/dev/`：

- `summary.json`：分条件准确率、无效数、target 命中率、各类 lure 错误接受率、critical gap、严格三元绑定平衡准确率、反事实 pair accuracy、关系准确率及属性准确率。
- `scored.jsonl`：逐题可复查评分，含标准答案，**不得发送给模型**。
- `conditions.csv`、`accuracy.svg`：可直接用于整理报告的表格与图。
- 同一 JSON 内包括 Q1/Q2、Q1/Q3、Q4/Q5 配对比较、总数量/每图数量对照、2D/3D 对照，以及 L3 上的探索性带 ridge 惩罚逻辑回归（数量、问法及交互，按组聚类的近似 sandwich 区间）。

默认拒绝对未完成的任务出正式评分。检查部分进度可显式加 `--allow-partial`；此时报告 coverage，仅对已有回答计算分数，不能当作完整基准成绩。模型无效回答始终算错误，未运行题不会被伪装成模型错误。

所有 bootstrap 以独立 `cluster_id` 为单位，保留同组内的两种世界、不同问法与条件依赖；默认 1,000 次。跨任务不计算混合总分。回归是探索性分析，小于 10 个组、因变量恒定或设计矩阵秩不足时明确跳过；正式解释应考虑多重比较和估计的不确定性。

当前先整理 Qwen3-VL-8B 的完整行为结果，不安排同系列 4B 对照。后续模型由用户另行指定；其他架构需先验证聊天模板、多图输入与环境兼容性。跨模型分析应匹配数据和任务条件，并记录各自预处理设置。

## 任务与生成协议

| 等级 | 实现 |
|---|---|
| L0 | 单物体属性校准；2D、固定视角、另一视角和遮挡的 3D 图分别报告 |
| L1 | 2/3/4 个物体，target、unrelated、near1、near2、critical；每种负例匹配一个正例 |
| L2 | 2×2×2 属性立方体奇偶分组，4 对 4 的严格反事实，含 pairwise-complete 负例 |
| L3 | 严格反事实基础上，总数量 4/8；每图数量 1/2/4/8（仅整除条件）；增加的 4 个物体在两世界完全相同 |
| L4 | 左右、上下、相机深度前后、同图共现；两世界保持完整物体集合相同，仅改变关系 |
| L5 | 同一严格反事实集合的 2D、固定视角 3D、跨视角 3D、查询图部分遮挡 |

属性空间是 goal 中的 **6×4×6=144**，与旧实验的 6×6×6 不同。3D 物体由轮廓挤出为网格，正交投影后用三角形栅格化、z-buffer、物体坐标纹理绘制；包含真实深度和自遮挡，但只是一类受控挤出形状，不代表自然场景或任意 3D 物体。

Q1 图片判断、Q2 文本判断、Q3 图片+文本判断使用相同学习图片和相同查询三元组；Q4 同图二选一与 Q5 分图二选一使用相同候选、顺序和学习内容；Q6 为四选一；Q7 为颜色+形状唯一定位后的纹理回忆；Q8 为关系判断。Q8 仅在 L4，L0 使用独立 `CAL` 协议。不同任务机会水平不同，Q7、Q8 不硬套为同一种识别题。

严格反事实两侧共用相同测试图/文字及候选顺序，正确答案随学习集合改变；候选位置按组循环平衡。严格反事实 Q6 的两个核心候选分别来自两种世界，额外两个候选从两种世界均未出现的组合中选择；因此它不是四个同等熟悉候选之间的纯绑定选择，解释 Q6 与二选一差异时需考虑候选熟悉度。Q7 若线索不唯一，生成器会拒绝。图像名字是哈希，标准答案与属性只保存在独立 labels/assets 文件中。runner 只读取 inputs，传给模型的消息不含 trial ID、题型标签或答案。

新版每题均为独立上下文，无填充图或额外干扰图。物体数量变化仍必然改变输入图像/视觉 token 总量；每个物体的名义像素大小固定，多物体画布随布局扩大，runner 记录实际 `image_grid_thw` 与输入 token 数以供控制。这不构成独立上下文距离实验。3D 的跨视角/遮挡操作施加在**测试图**；Q2/Q7 没有测试图，因此固定/跨视角/遮挡条件下输入相同，属于文本控制，不应把重复条件当成额外独立证据。

## 文件索引与复现

- `generate.py` / `render.py`：确定性标签与 2D/3D 图像生成。
- `validate.py`：验证所有属性、诱饵定义、反事实频数、选择题唯一性、相同查询、关系标签、跨问法学习图一致性、分组切分及 PNG 哈希。
- `run.py`：本地推理、断点恢复、独占写入、配置指纹。
- `score.py` / `stats.py` / `compare.py`：分任务评分、配对比较、整组 bootstrap、逻辑回归与模型比较。
- `probes.py` / `../scripts/bindmem_probes.sbatch`：历史功能保留，不属于当前实验计划。
- `../scripts/`：直接运行、Slurm GPU/CPU 提交及检查脚本。

数据 manifest 保存生成种子、依赖版本、输入/标签/图像哈希。运行 manifest 保存数据指纹、模型配置指纹、权重文件大小与 mtime、包版本、解码/分辨率/随机种子及代码哈希；本地大权重不逐字节哈希。Hub 模型可用 `--revision <commit>` 固定版本，推荐正式实验使用本地固定权重。重启时配置、代码或依赖变化会拒绝混写，应保留旧输出并使用新目录。

截至 2026-10-26，Qwen3-VL-8B 的 L0–L5 全量行为推理与评分已完成；当前结果仅支持该模型和本数据集下的行为结论。

## Q1/Q2 完成后的二选一实验

```bash
sbatch scripts/bindmem_choices.sbatch
```

此脚本在 L2/L3 跑 Q4（同图二选一）和 Q5（分图二选一），共 10,240 题
（dev 2,560 / test 7,680），输出到 `runs/bindmem_v2/qwen8b_choices`。
申请 48 小时，沿用图片实验的 CUDA 检查、节点排除及断点续跑设置，
结束后自动评分两种 split；`scored/test/summary.json` 中的
`paired_formats.Q4_vs_Q5` 给出 Q5 减 Q4 的配对准确率差及区间。
无需改变环境；重提相同命令可接续，勿同时写入同一输出目录。
二选一用于检查判断形式与候选布局的影响，不能直接将跨题型差异全归因于回答偏置。

## 补齐 L2/L3 的 Q3、Q6、Q7

```bash
sbatch scripts/bindmem_formats.sbatch
```

Q3 是图片加文字的 Yes/No 判断，Q6 是四选一，Q7 是根据颜色和形状
回忆纹理。共 15,360 题（dev 3,840 / test 11,520），输出到
`runs/bindmem_v2/qwen8b_formats`，不重复已完成的 Q1/Q2/Q4/Q5。
脚本使用现有环境、单张 3090、48 小时时限及 CUDA 启动检查；
不采集隐藏状态、不运行探针，完成后自动评分 dev/test。
日志为 `slurm-bindmem-formats-<JOBID>.out/.err`，中断后重提同一命令可续跑。
各问法分开报告，不将 Q6/Q7 与 Yes/No 的准确率混成总分。
Q1 与 Q3 位于不同运行目录，后续需按 `semantic_id` 匹配做配对分析。

Q3/Q6/Q7 启动排查（2026-10-25）：作业 165256、165257 在 node014
上立即失败，Slurm 记录退出码 1，未生成标准输出/错误日志及运行目录；
具体根因尚未确定。`bindmem_formats.sbatch` 暂时排除 node014
（同时保留 node001/node015 排除），并增加 shell 启动及失败位置日志。
如需明确固定此前通过运行验证的 node010，可使用：

```bash
sbatch --nodelist=node010 scripts/bindmem_formats.sbatch
```

若 Slurm 在启动 shell 或打开日志前失败，脚本内诊断无法捕获，
需结合 `sacct` 或管理员侧 slurmd 日志继续定位。无需据此更改 Python 环境。

## L1 基础绑定与诱饵对照

```bash
sbatch --nodelist=node010 scripts/bindmem_basic.sbatch
```

在已完成 L2/L3 的 Q1–Q7 后，补跑 L1 的 Q1–Q7，共 35,520 题
（dev 8,880 / test 26,640）。包含 2/3/4 个学习物体条件；判断题比较
真正出现物体与 unrelated、near1、near2、critical 诱饵，配合选择和属性回忆任务。
使用新目录 `runs/bindmem_v2/qwen8b_basic`，不重复 L2/L3。
推荐固定此前两次成功完成作业的 node010；脚本同时暂避 node001/014/015。
申请单张 3090、48 小时时限，沿用现有环境，包含 CUDA 启动检查；
不采集隐藏状态、不运行探针，结束后自动评分 dev/test。
日志：`slurm-bindmem-basic-<JOBID>.out/.err`；中断后重新提交同一命令可续跑。
评分位于 `runs/bindmem_v2/qwen8b_basic/scored/{dev,test}/`，
重点查看 target 命中率、不同诱饵的错误接受率及 critical lure gap。

## L1–L3 完成后的空间关系与 3D 扩展

```bash
sbatch --partition=IAI_SLURM_3090 --qos=singlegpu --nodelist=node010 scripts/bindmem_extensions.sbatch
```

同一作业运行 L4/Q8（左右、上下、深度前后、同图共现），以及
L5/Q1–Q7（2D、固定视角 3D、跨视角、查询图遮挡）。共 20,480 题：
L4 2,560 题，L5 17,920 题；合计 dev 5,120 / test 15,360。
输出到 `runs/bindmem_v2/qwen8b_extensions`，单张 3090，48 小时时限，
沿用当前环境与 CUDA 启动检查，不采集隐藏状态、不运行探针。
完成后自动评分 dev/test，日志为 `slurm-bindmem-extensions-<JOBID>.out/.err`。
中断后重提相同命令可续跑。L5 的 Q2/Q7 无查询图，固定/跨视角/遮挡
条件属于重复的文本控制，不作为独立视觉效应证据。

2026-10-26 核对：L1 作业 165450 已完成 35,520 题，dev/test 均完成评分。
重复 L1 作业 165455 被提交到 5090 分区，因该分区不允许 singlegpu QoS
而处于 pending；无需继续该重复实验。新命令明确指定 3090 分区与 singlegpu QoS。

## 8B 全量行为结果汇总

当前只整理已完成的 Qwen3-VL-8B 结果，不安排 4B 对照，不包含探针；其他模型待用户指定。

```bash
conda activate qwen3vl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/summarize_bindmem_runs.py
```

脚本在 CPU 上运行，无需 GPU 或新依赖。核对六批运行配置、输入标签指纹、
全量题目覆盖、重复题和失败题，合计 93,120 题（dev 23,280 / test 69,840）。
正式 test 使用 240 个独立组，dev 使用 80 个组；bootstrap 按组采样。

结果位于 `runs/bindmem_v2/qwen8b_report/`：

- `overview.md` / `overview.html`：面向非项目成员的通俗总结；HTML 可独立打开、分享或打印。
- `report.md`：中文结果报告、主要发现、解释边界及图表入口。
- `summary.json`：完整指标、配对比较、回归与来源指纹。
- `accuracy.csv` / `conditions.csv`：分任务与细分条件结果。
- `paired_comparisons.csv`：问法、数量、布局及 3D 条件的配对差异。
- `invalid_responses.csv`：无效回答原文与标签，供人工复核。
- `figures/*.svg`：含组 bootstrap 95% 区间的独立矢量图。

不生成跨任务混合总分；无效输出按错误计分；L5 的重复文本控制不能作为独立视觉效应。
该目录是分析报告，不是伪造的单次推理运行目录。

## 按实验设计组织的可视化报告

新版 `PROJECT_REPORT_V2.md` 参考根目录历史 `PROJECT_REPORT.md` 的结构，
包含项目概述、数据制作、测试协议、运行设置，以及逐实验的目的、方法、样本量、结果图和解释。
旧报告保留为 v1 历史记录，不合并旧数据。

报告目录 `runs/bindmem_v2/qwen8b_report/` 中：

- `visual_report.html`：内嵌全部图片，可单文件分享或浏览器打印。
- `visual_report.md`：与根目录新版报告内容一致，使用相对图片路径。
- `visual_figures/`：8 张 PNG/SVG 图、图表数据与实际示例题来源。
- `实验结果图集.pdf`：8 页独立图集，不含完整报告正文。

复现图表和报告：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /ceph/home/shizhongyu/anaconda3/bin/python scripts/build_bindmem_visual_report.py
```

此脚本使用现有基础环境中的 Matplotlib 和 Markdown 库及系统中文字体，
不需要安装新依赖，不修改用于推理的 qwen3vl 环境。输入为已核对的 summary.json
及数据集中的实际示例图片；不运行模型。

## 实际输入与模型输出

`INPUT_OUTPUT_REPORT.md` 和 `runs/bindmem_v2/qwen8b_report/input_output_report.html`
说明每题消息的真实组成，展示 CAL、Q1–Q8，以及多物体、三维遮挡、严格负例和无效输出。
共 17 个实际案例；常规案例按正式数据顺序选取，无效回答例子明确按诊断条件选取。
每例包含实际学习图片、按顺序排列的查询文字/图片、允许输出、原始模型回复、
未发送给模型的标准答案及评分解释。Q4–Q6 还给出评测器的候选对应关系。

主报告 `PROJECT_REPORT_V2.md` 和 `visual_report.html` 已包含同一输入输出说明。
所有图片在 HTML 中内嵌，可单文件分享。`io_examples/cases.json` 保存完整
应用层 messages、原始输入/标签/预测、来源路径与图片指纹，便于逐题复查。

运行 `scripts/build_bindmem_visual_report.py` 时自动调用
`scripts/bindmem_report_io.py` 重建该部分；不重新运行模型。
