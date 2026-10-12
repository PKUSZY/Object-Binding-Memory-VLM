# 物体绑定数据集 v0.1（预览版）

用途：构建可控的多图物体绑定识别任务，供后续两个 VLM 的零样本 baseline 测试。本次只生成数据与预览，尚未运行 VLM 或完成感知校准。

## 先看图片

- 用浏览器打开 `preview/index.html`，可切换开发/测试划分、350 个学习列表、按结构难度筛选、显示/隐藏标签、查看各颜色的形状×纹理图谱。
- `preview/sample_episode.png`：一组学习图片和五类测试项的带标签总览。
- `preview/attribute_atlas.png`：蓝色物体的 6×6 形状/纹理图谱。

预览含答案，只供人检查；绝不能把拼图或 HTML 截图作为模型输入。

## 数据规模

| 划分 | 学习列表 | 每列表测试项 | 测试题 | 原始图片 |
|---|---:|---:|---:|---:|
| dev | 50 | 5 | 250 | 400 |
| test（候选，未冻结） | 300 | 5 | 1,500 | 2,400 |
| calibration | — | — | — | 432 |
| 合计 | 350 | — | 1,750 | 3,232 |

每个学习列表包括三张独立图片；五个测试项各自独立调用，不能一次全部呈现。同一列表的学习图会在其五道题中复用，统计应按学习列表聚类。

`test` 是预先独立抽样的候选测试划分。正式跑 baseline 前，先在 dev 和 calibration 校准可识别性，固定生成器/提示词/processor 后冻结版本。不能用测试结果筛选刺激。

## 标签定义

物体身份 = 颜色 + 形状 + 纹理。位置、大小和纹理相位的小变化不改变身份。每个学习列表中，每维有三个不同取值。

| test_type | 定义 | 答案 |
|---|---|---|
| target | 完整三元组出现过，测试时独立重新渲染 | YES |
| critical_2plus1 | 两个属性来自一个物体，第三个来自另一物体 | NO |
| critical_1plus1plus1 | 三个属性来自三个物体 | NO |
| near_2seen | 两个属性来自同一物体，第三个在学习集合未出现 | NO |
| unrelated | 三个属性值都未在学习集合出现 | NO |

## 结构难度：三个档位与共同正例对照

当前按照“有多少个新属性可用于排除测试物体”作先验分层，**尚未通过模型实测校准**。这不是对感知难度或模型错误率的保证，也不改变原来的五个 `test_type`。

| structural_difficulty | 结构档位 | 对应类型 | 新属性数 | dev / test 题数 |
|---|---|---|---:|---:|
| easy | L1 简单 | unrelated | 3 | 50 / 300 |
| medium | L2 中等 | near_2seen | 1 | 50 / 300 |
| hard | L3 困难 | critical_2plus1 + critical_1plus1plus1 | 0 | 100 / 600 |
| control | 正例对照，不参与排序 | target | 0，且完整组合已出现 | 50 / 300 |

L1 中任意一个属性的陌生性都足以支持 NO。L2 只剩一个新属性，需要识别这一线索。L3 没有这种线索，必须核对跨属性的绑定关系。Target 作为共同正例对照，避免模型通过总答 NO 通过各档评测。

两个 hard 子类不继续强行排序：2+1 与最近的学习物体只差一个属性，1+1+1 差两个属性；更高的相似性与更多的属性来源是不同因素，实测难度可以不同。应同时保留 `test_type`、`seen_feature_count`、`novel_feature_count`、`nearest_hamming_distance` 作分组分析，不把 Hamming 距离当作统一难度分数。

预览中选择 L1/L2/L3 会同时显示该列表的 Target。隐藏答案时会自动切回全部并禁用难度筛选，避免筛选结果直接暴露答案类别。预览始终仅供人工检查。

评测读取 `difficulty/{dev,test}.json`：`shared_target_ids` 是各档复用的正例；`tiers.*.lure_ids` 是该档负例。它们只是已有 trial_id 的分组索引，没有新生成或重复计入独立题目。每个原始 trial 只需跑一次。

每档报告共同的 Target 命中率 H、该档错误接受率 FA、正确拒绝率 R 和无效输出率。分母均使用该组全部题目：H=YES/Target总数，FA=YES/诱饵总数，R=NO/诱饵总数；无效输出记为错误，因此有无效输出时 R 不等于 1−FA。可另报 `(H + R) / 2` 作为正负类平衡准确率；hard 的 R 对两个 Critical 子类的 R 等权平均，同时分别报告各子类 FA。总答 NO 时平衡准确率为 50%，但单独诱饵拒绝率为 100%。

统计区间和难度之间的差值应按 episode 聚类重采样，保留同一列表内的所有题目与共享 Target。不能把各档共用的 Target 当作多份独立观测。无效输出单独计数，不能只在有效答案上报告较好结果。

这里的注释方案版本是 `difficulty_scheme.version = 1.0`，与刺激版本 `0.1.0-preview` 分开记录。已有图片、题目 ID、模型输入和答案没有改动；迁移验证见 `difficulty_annotation_report.json`。`dataset.json` 中原有 `generator_sha256` 保留为原始图像生成的来源记录，迁移脚本哈希另行记录。新运行生成器会直接产生这些难度字段。

三元组词表：

- color: red, blue, green, yellow, purple, orange。
- shape: circle, square, triangle, star, cross, semicircle。
- texture: solid, horizontal_stripes, vertical_stripes, diagonal_stripes, dots, checkerboard。

全部图片为 448×448 RGB PNG，无文字、编号和属性标签。物体的标称面积一致，面积抖动 ±5%，中心抖动 ±7 像素，2 倍超采样后抗锯齿缩小；纹理深色部分保持底色的色相。图形轮廓/纹理亮度仍可能影响感知，不宣称严格的感知难度相等。

每个 split 的学习属性频数精确平衡（默认大小下 dev 每值25次，test每值150次）；target来源位置和替换维度差异至多1。各 query 的属性频数不是强制完全相等，完整分布见 `summary.json`。

## 文件结构

```text
generate.py                 # Pillow 生成器
difficulty.py               # 结构难度定义、分组索引与旧版注释迁移
validate.py                 # 独立数据验证
build_preview.py            # 离线预览和联系图
preview_template.html       # 预览模板
requirements.txt
data/v0.1/
  dataset.json              # 版本、种子、词表、生成器哈希、环境
  summary.json              # 规模与属性分布
  validation_report.json    # 实际验证报告
  difficulty_annotation_report.json # 已有数据注释迁移报告（仅迁移时产生）
  difficulty/{dev,test}.json# 按难度分组的已有题目 ID、共同 target ID
  images/<opaque-id>.png    # 文件名不包含语义标签
  episodes/{dev,test}.jsonl # 每行一个学习列表+五个测试项
  manifests/{dev,test}.jsonl# 每行一题，含标签和来源，供审计
  manifests/calibration.jsonl
  inputs/{dev,test}.jsonl   # 供模型 runner 使用的白名单字段
  labels/{dev,test}.jsonl   # 判分答案，不能输入模型
preview/
  index.html
  sample_episode.png
  attribute_atlas.png
```

所有 JSONL 内 `image` 路径相对于 `data/v0.1/`。来源 `feature_sources` 为从0开始的学习图索引，`null` 表示该属性在整个学习集合未出现。预览显示为从1开始的编号。

难度字段只存在于审计用 episodes/manifests 与分组索引，不进入 `inputs/*.jsonl`，也不传给模型。

模型 runner 读取 `inputs/*.jsonl`：只有 trial_id、study_images、test_image 和固定 instruction。实际请求仅传任务说明、依序三张学习图片和一张测试图片；trial_id 留在日志，不用作输入。只要求输出 YES/NO。

`manifests` 和 `labels` 含标准答案，不可整体序列化给模型。它们供评测、调试和人工核验。

## 复现

建议 Python 3.12，依赖只有 Pillow：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate.py --output data/v0.1 --seed 20261007 --dev 50 --test 300 --calibration-variants 2
python validate.py --data data/v0.1
python build_preview.py --data data/v0.1 --output preview
```

已存在数据不会被默认覆盖。只在明确重建本生成器的数据目录时加 `--overwrite`；该选项会替换指定数据目录。对可比的像素级复现应固定 Python/Pillow 版本与代码文件。预览字体按平台回退，不影响无文字的原始刺激图。

为旧预览数据补充难度标签，可运行 `python difficulty.py --data data/v0.1`，再运行验证和预览构建。这只更新审计元数据及分组索引，并校验原图片、输入与答案字节未变。它不重新渲染刺激图。

也可运行极小样本：

```bash
python generate.py --output data/smoke --dev 6 --test 6 --calibration-variants 1
python validate.py --data data/smoke
```

离线 HTML 无需网络；若浏览器不允许访问本地相对路径，可从本目录启动：

```bash
python -m http.server 8765 --bind 127.0.0.1
```

随后访问 `http://127.0.0.1:8765/preview/index.html`。

## 验证与解释边界

`validate.py` 重新计算成员关系、熟悉特征数、最近 Hamming 距离和属性来源，检查 split 之间无重复学习集合、PNG模式与大小、字节哈希、无像素完全相同的图片、输入字段无标签泄漏、全部校准组合覆盖，以及结构难度和共同正例索引。

少量不同渲染参数可能落在相同像素网格上；生成器检测到完全相同的像素时，使用确定性的新种子重采样，并在 `render_resample_count` 记录次数。该过程不修改三元组或答案。

运行生成器回归检查：`python -m unittest discover -s tests -v`。检查同配置字节级复现、像素碰撞重采样，以及损坏的答案/来源/模型输入能否被验证器拒绝。

这些是数据正确性检查，不是 VLM 感知校准。学习图仍在上下文中，实验测量上下文绑定识别，不直接测量撤除原图后的内部记忆。

baseline 后续只报告实际结果；五个子类等量时总回答 NO 可得80%微平均准确率，故应以 target命中率、各 lure错误接受率、宏平均和无效输出率为主。
