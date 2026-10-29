# 补充实验：LLaVA 7B、匹配的重组梯度、真实误认案例

**2026-10-29 最新进展：** [精简实验](PRIORITY_EXPERIMENTS.md)已全部完成：作业 167544 正常结束，LLaVA 的二维识别、严格绑定与图片／图文梯度共 5,520 道正式 test 已评分并完成同题比较。[完整项目报告](PROJECT_REPORT_COMPLETE.md)已汇总全部正式结果。Qwen 六物体梯度也已完成；原完整作业 167469 已停止。下文保留 2026-10-28 的方案和全量命令，所述“尚未运行／下载”是当时状态，当前不要重复提交全量任务。

更新时间：2026-10-28。新增代码在 `bindmem_extra/`，保持已完成实验的 `bindmem/*.py` 不变。当前仍只做行为实验，不运行探针。

## 当前已完成什么

- Qwen3-VL-8B 的原有 93,120 题已有结果。
- LLaVA-1.5-7B 的适配、长度预检、断点续跑、评分、同题模型比较和 Slurm 脚本已实现；**尚未运行正式 LLaVA GPU 推理**。
- 已下载 LLaVA 的配置和处理器小文件，锁定 Hugging Face commit `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；**模型权重尚未下载**。完整权重由提交脚本下载，重复下载会续传并沿用该 commit。
- 两套梯度的数据生成器与严格校验已实现，正式 320 组数据由批处理脚本生成；没有把临时测试数据当作正式实验。
- 已从真实 Qwen 输出导出 6 个误认案例：[可独立分享的案例网页](reports/bindmem_complete_20261029/cases_qwen.html)、[逐例证据 JSON](reports/bindmem_complete_20261029/evidence/cases_qwen.json)。

## 1. 新模型与可比范围

默认模型是 [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf)，不是 LLaVA-NeXT 或 OneVision。使用本地权重、FP16、SDPA、单张 3090、确定性生成、最多 80 个新 token。学习图片、查询语义、题目 ID、切分和严格输出评分沿用原基准；保留 USER/ASSISTANT 模板和原始图文顺序。

官方说明：LLaVA 可处理多图输入，但没有专门进行多图训练；表现因此也反映模型对多图协议的适应程度。[官方文档](https://huggingface.co/docs/transformers/model_doc/llava)

LLaVA 原生上下文为 4,096 token，每图约 576 个视觉 token。已用真实处理器检查原基准：

| 范围 | 原题数 | 可保留完整输入 | 明确排除 |
|---|---:|---:|---:|
| dev + test | 93,120 | 88,640 | 4,480 |
| test | 69,840 | 66,480 | 3,360 |

排除项全部是 L3 的“8 个学习物体、每图 1 个”条件，涉及 Q1–Q7。**不截断图片或文本、不扩展上下文上限、不把这些题计为错误。**预检明细在 `runs/bindmem_v2/llava7b_preflight/preflight.json`。模型比较明确使用相同 trial_id 的交集，并对 Qwen 重新计算这一交集的成绩；不能将 66,480 题的 LLaVA 汇总与 69,840 题的 Qwen 汇总直接比较。

为避免 CLIP 默认中心裁剪切掉宽图中的物体，LLaVA 输入先按 CLIP 均值补成正方形，再用原生处理器缩放到 336×336。没有裁掉候选或学习物体。Qwen 沿用原生动态分辨率，因此这是两个完整模型系统的同题比较，不能声称视觉 token 数或有效分辨率相同。输入包装、图像策略、处理器指纹和实际 token 数均有记录。

## 2. 为什么不能固定 4 个物体做完整 0–3 梯度

令候选为 `(A,B,C)`，且完整候选不在学习集中。如果 AB、AC、BC 三对均出现，则每种候选属性至少出现两次：例如 AB 与 AC 不能来自同一个学习物体，否则就出现了 ABC。因而至少有 6 次候选属性出现。

在 0 对条件，每个学习物体最多包含一种候选属性；4 个物体最多容纳 4 次。**固定候选属性频数时，这两个条件无法同时成立。**这不是生成器搜索不足，而是组合约束矛盾。

代码提供两种设计。在没有另行指定前，脚本以完整的六物体梯度为默认。

| 方案 | 学习物体数 | 每图物体数 | 负例已见属性对数 k | 每个候选属性的频数 | 320 组总题数 / test |
|---|---:|---:|---|---|---:|
| 推荐完整梯度 | 6 | 2 | 0、1、2、3 | 都是 2 | 7,680 / 5,760 |
| 四物体补充梯度 | 4 | 1 | 1、2、3 | 都是 2 | 5,760 / 4,320 |

每组随机选择属性取值，构造同一个候选在所有 k 下的负例学习集合。程序保证：

1. 固定完整候选及查询图片；候选在所有负例学习集合中都没有出现。
2. 固定学习物体数量、每图数量、二维渲染流程和渲染参数；每个学习物体身份不同。
3. 固定**所有单属性取值的频数**，不仅固定候选三种属性的频数。六物体方案的每个维度有三个取值，各出现两次；四物体方案为 2/1/1。
4. 六物体方案中，每个已经出现的候选属性对恰好出现一次，避免将“属性对种类数”混同于“同一属性对重复次数”。四物体的 k=1 必须将唯一已见属性对重复两次，代码和标签明确记录这一限制，两套设计分别分析。
5. 每个 k、每种问法均为目标/负例 1:1。正例在同组不同 k 下使用完全相同的输入，不把这些重复正例视作独立证据。
6. 问法为 Q1 图片判断、Q2 文字描述判断、Q3 图文判断；采用 80 组 dev / 240 组 test 的整组切分。

这里操纵的是候选与学习集合的二阶重合程度；不同学习物体及其其他属性对关系必然随构造改变，不能声称所有像素或所有二阶统计都相同。原 L2 的正负世界一阶、二阶边际均匹配，检验的是另一问题，两者分别报告。

核心结果应看 **k=0/1/2/3 的误认率、正确拒绝率和目标命中率**，并报告相邻 k 的配对差及按组 bootstrap 95% 区间。输出不预设误认率一定随 k 单调增加。

## 3. 真实强重组误认案例

已有案例来自正式 test 的逐题记录。Qwen 的 L2/Q1 有 135/240 道强重组负例误答 YES，L2/Q3 为 88/240。分别按 trial_id 字典顺序取前 3 条，共 6 条；选择规则公开，不按照图片效果挑选。

例如，实际学习物体为：

- 红色、纯色、星形。
- 红色、条纹、正方形。
- 蓝色、条纹、星形。
- 蓝色、纯色、正方形。

查询图片是 **红色、纯色、正方形**。颜色＋纹理出现在第 1 个物体，颜色＋形状出现在第 2 个，纹理＋形状出现在第 4 个，完整三元组没有出现。标准答案 `NO`，模型原始回复 `YES`。trial_id：`0141c5681033d7edc0a0eaab9bcf`。对应另一学习世界中的正例也回答 `YES`。

[案例网页](reports/bindmem_complete_20261029/cases_qwen.html)包含原始图片、实际提示、原始回复、三种属性对的证据来源；JSON 另存输入、标签、预测、对应正例和文件哈希。它们是**程序生成场景中的真实模型错误**，不等同于自然照片案例。未来 LLaVA 若没有符合条件的错误，脚本会明确报告“无案例”，不会生成虚构回复。

## 4. 现在提交什么

从仓库根目录，在现有环境中提交两个作业即可。默认主梯度为六物体 0–3，不重跑 Qwen 的原有基准。

```bash
cd /ceph/home/shizhongyu/class_project
conda activate qwen3vl

# LLaVA：下载权重、校验数据、CPU 预检、GPU smoke、原基准可运行部分、六物体梯度、评分及案例
sbatch scripts/bindmem_llava.sbatch

# Qwen：仅运行新增六物体梯度，不重跑已完成 L0–L5
sbatch --export=ALL,MODEL_KIND=qwen,SUITE=gradient scripts/bindmem_llava.sbatch
```

两个模型共享梯度数据目录，生成阶段有文件锁；不会同时写入数据。模型结果放在不同目录。作业时限 48 小时，超时后重新提交同一命令即可续跑；不要同时提交两个相同模型、相同输出目录的作业。原始回答即使格式无效也按错保留；运行错误会立即停止并在下次重试。

如需要两套梯度都做，在以上两条提交命令中分别加入 `GRADIENT_OBJECTS='6 4'`：

```bash
sbatch --export=ALL,GRADIENT_OBJECTS='6 4' scripts/bindmem_llava.sbatch
sbatch --export=ALL,MODEL_KIND=qwen,SUITE=gradient,GRADIENT_OBJECTS='6 4' scripts/bindmem_llava.sbatch
```

若计算节点不能访问 Hugging Face，先在可联网节点下载权重与准备数据：

```bash
STAGE=prepare bash scripts/run_bindmem_extra.sh
# 准备完毕后不再请求下载
sbatch --export=ALL,STAGE=run scripts/bindmem_llava.sbatch
```

模型下载固定 commit。不要在已经有结果的输出目录中更换模型、修改代码或修改推理设置，否则指纹检查会拒绝混合续跑。使用新的输出目录需要直接调用模块，并自行指定 `--output`。

## 5. 环境和验证状态

继续使用已有 `qwen3vl`：Python 3.11、torch 2.10.0+cu128、transformers 5.19.0、Pillow 12.3.0、NumPy 2.4.6、accelerate 1.15.0、huggingface-hub 1.33.0。**目前无需安装或升级依赖**。仓库下载的 fast tokenizer 可以直接加载，不需要补装 sentencepiece/protobuf。没有启用量化或 Flash Attention 额外依赖。

本次已实际验证：两套小规模数据的完整约束、标签破坏检测、无效/缺失回答和排除题处理、逐题同模型合并与跨模型配对、案例提取、图片不被裁掉、真实 LLaVA 处理器的 59 类输入格式与长度估算、Shell 语法与 `sbatch --test-only`。另用微型随机初始化 LLaVA 在 CPU 验证了四图输入的前向与生成接口；临时测试结果未写入正式实验。尚未验证真实 7B 权重在计算节点上的 GPU 生成和显存占用；脚本会先执行单独的 dev smoke，再继续正式运行。

```bash
python -m unittest discover -s tests -p test_bindmem_extra.py -v
python -m bindmem_extra.inference --backend llava --model models/llava-1.5-7b-hf \
  --data data/bindmem_v2 --output runs/bindmem_v2/llava7b_preflight --context-policy exclude --preflight
```

## 6. 结果位置与统计

- 原基准新模型：`runs/bindmem_v2/llava7b_behavior/`。
- 原基准同题模型比较：`runs/bindmem_v2/qwen8b_vs_llava7b/comparison.csv`。
- 梯度：`runs/bindmem_gradient_n6/{qwen8b,llava7b}/`，四物体对应 `n4`。
- 每个运行的 `scored/test/report.md`、`rates.csv`、`summary.json`：正式分条件指标及 95% 区间。
- `selection.json`：选中题及上下文超限题清单；`predictions.jsonl`：实际原始回复、运行错误、token 数与耗时。
- `strong_error_cases/cases.html`：该模型的真实案例；网页图片已内嵌，可独立分享。
- 两个梯度模型均完成评分后，后完成的作业自动生成 `qwen8b_vs_llava7b/` 比较结果。

按随机组重采样 1,000 次，将同组条件与重复正例一起重采样。无效输出计错；运行失败不会混进正式完整成绩。对全对/全错样本，bootstrap 区间可能退化，不能解释为总体不确定性为零；多条件区间未做多重比较校正。不计算跨题型混合总准确率。

手工重新生成梯度同题比较：

```bash
python -m bindmem_extra.score --data data/bindmem_gradient_n6 \
  --run-a runs/bindmem_gradient_n6/qwen8b --run-b runs/bindmem_gradient_n6/llava7b \
  --output runs/bindmem_gradient_n6/qwen8b_vs_llava7b
```
