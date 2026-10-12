# 物体绑定记忆整合数据集

这是一个包含**主实验轨道**和**补充基准轨道**的发布包。主实验 `core/` 严格按桌面《物体绑定记忆数据集设计简述》组织：24 种子 × 学习图 4/8/12 张 × 干扰图 0/4 张 × 每组 6 个独立试次 = **864 题**，YES/NO 各 432 题。`auxiliary/three_study_v0.1/` 来自桌面 `binding_dataset.zip`，保留其 1,750 题作为单独基准。两轨道的词表、学习量和答案比例不同，结果分别报告。

## 从哪里开始

- [异同与整合决策](COMPARISON.md)
- [下一步实验流程](NEXT_STEPS.md)
- `dataset_index.json`：机器可读的轨道与规模索引。
- `validation_report.json`：结构与文件校验结果。

## 主实验目录

```text
core/
  images/                 216 张学习物体 PNG，384×384
  probes/                 432 张独立重绘的测试物体 PNG
  distractors/            4 张灰色干扰图 PNG
  calibration/images/     432 张感知校准 PNG，每个三属性组合 2 个变体
  calibration/inputs.jsonl
  calibration/labels.jsonl
  inputs/dev.jsonl        216 题，可用于流程调试与提示词确定
  inputs/test.jsonl       648 题，正式测评时使用
  labels/dev.jsonl
  labels/test.jsonl
  trials.jsonl            864 题完整审计清单，含答案与属性
  groups.jsonl            144 个条件组
  metadata.json
```

主实验共有 6 色、6 形、6 纹理，形状包括圆、方、三角、六边形、星形、菱形；纹理包括圆点、斜条纹、网格、棋盘格、波纹、折线纹。每组 3 道 `target`（YES），以及各 1 道 `critical_recombination`、`near_lure`、`unrelated_lure`（NO）。关键重组题的三种属性分别出现过，任意两项从未在同一张学习图中配对。近似诱饵的颜色未出现，形状和纹理各自出现但两者未配对。无关诱饵的三个属性都未出现。

同一种子与学习量的 0/4 干扰条件复用相同学习图、测试物体、测试顺序和测试 PNG；只增加四张干扰图。每个测试试次仍必须开启**全新会话**。查询图片经过轻微平移、缩放，防止 `target` 靠完全相同的像素匹配；身份只按颜色、形状和纹理三项定义。

模型 runner 只读取 `inputs/*.jsonl`，把 `instruction` 作为文本、`study_images` 按顺序逐张作为图片、再把 `distractor_images` 逐张作为图片，最后把 `test_image` 作为图片。路径和 `trial_id` 是本地索引，不传给模型。**不要把 `trials.jsonl`、`labels/*.jsonl`、`dataset_index.json` 或图片文件名文本传给模型。** 图像文件名已改为不带属性含义的哈希。

## 补充基准目录

`auxiliary/three_study_v0.1/` 保留上传压缩包中的生成器、验证器、预览、数据和 README；只清理了 macOS 元数据与 Python 缓存。其 `data/v0.1/inputs/{dev,test}.jsonl` 可用于单独的三学习图测评，`labels/` 供判分。这个轨道无干扰图，且 5 类题中只有 1 类为 YES，不能并入主实验的 864 题分母。它原本标记为尚未经过 VLM 感知校准的预览版。

## 复现与校验

当前发布包的结构与标签已通过 `node validate_integrated.js`。如需重建主实验，安装 Node.js 后在此目录运行：

```powershell
npm.cmd ci
npm.cmd run build
npm.cmd run validate
```

`build` 仅重建 `core/`，不会修改补充基准。补充基准自身的 Python 复现方式见其 README。打包时不包含 `node_modules`，安装命令会按锁文件获取固定依赖。

结构校验只证明数量、图片格式、标签逻辑、输入隔离和诱饵约束成立；模型能否稳定看清每种属性，仍需按 `NEXT_STEPS.md` 做感知校准。
