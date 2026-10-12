# 物体绑定记忆项目整理与执行方案

## 1. 项目目标

本项目测量视觉语言模型能否记住“颜色 + 形状 + 纹理”组成的**同一物体**，而不是把三个属性分别见过就误判为出现过。主实验在新会话中按序呈现学习图片，最后再呈现测试图片。另有一条配对扩展条件保留图片学习，但最后以文字询问属性组合是否出现过；两条轨道分别统计。

DOCX 中的正式设计是：6 种颜色 × 6 种形状 × 6 种纹理；24 个随机种子；学习图数量为 4、8、12；干扰图数量为 0、4；每组 6 道测试题（3 个正例、3 个负例），共 144 组、864 道题。

## 2. 资源分工

| 路径 | 身份 | 使用方式 |
|---|---|---|
| `物体绑定记忆数据集设计简述.docx` | 需求与约束的唯一来源 | 写报告、检查实验是否符合设计 |
| `binding_memory_integrated/core/` | **正式主实验** | 先做校准，再跑 dev，冻结流程后跑 test |
| `binding_memory_integrated/README.md` | 主实验说明 | 了解字段、输入隔离和复现命令 |
| `binding_memory_integrated/COMPARISON.md` | 主实验与旧数据的差异 | 解释为什么不能混合统计 |
| `binding_memory_integrated/NEXT_STEPS.md` | 原始后续步骤 | 作为执行清单的技术依据 |
| `binding_memory_integrated/auxiliary/three_study_v0.1/` | 补充基准（3 张学习图、5 类题） | 可另跑，结果单独报告 |
| `object_binding_memory_dataset/` | 早期未整合副本 | 只用于追溯；不要与 `core` 混用 |
| `binding_dataset.zip` | 旧版原始压缩包 | 只做来源存档 |
| `binding_memory_integrated.zip` | 当前整合发布包 | 交付或迁移到其他机器时使用 |
| `object_binding_memory_dataset.zip` | 早期副本压缩包 | 只做来源存档 |
| `binding_dataset_sample.png` | 单张人工视觉检查图 | 只确认刺激风格，不作为模型输入 |
| `Vision1_Homework1.pdf` | 另一份 Computer Vision I 作业 | 与本项目无关，放在项目外或标为 unrelated |

主线只读取 `core/inputs/{dev,test}.jsonl`，标准答案只从 `core/labels/{dev,test}.jsonl` 读取。`trials.jsonl`、`groups.jsonl`、`metadata.json`、校准标签和任何预览页面都不能发送给模型。

## 3. 当前状态

- `core` 已生成完整数据：216 张学习图、432 张独立测试图、4 张干扰图、432 张校准图。
- `core` 已通过 `node validate_integrated.js`：144 组、864 题，YES/NO 各 432，输入字段白名单和诱饵配对约束通过。
- Qwen3-VL-8B-Instruct 已在 RTX 3090 上完成主实验：864/864 道回答有效，总准确率 96.88%。
- 完整感知校准已完成：432/432 张图的颜色、形状、纹理及联合准确率均为 100%。
- 控制实验已完成：双图属性匹配 2150/2150 正确；强重组诱饵 27 题中 22 条有效、2 条正确拒绝。强重组样本较少，按诊断性结果单独报告。
- 图片学习、文字探测条件已生成 864 道与主实验逐题配对的题目，尚未运行模型推理。
- 校准准备材料在 `results/calibration_review/`：6 张供人工浏览的 contact sheet、432 行响应模板和单独的答案表；评分脚本为 `results/score_calibration.py`。
- 正式运行使用本地 Qwen 环境和 RTX 3090 Slurm 分区；安装文档在 `results/QWEN_LOCAL_SETUP.md`，当前执行环境本身看不到 GPU。
- `core` 的图片文件名是无语义哈希；测试图有确定性的轻微平移和缩放，避免像素级复制。
- 当前整合包中的 `node_modules` 已按 Linux x64 修复并完成校验。迁移机器时应删除 `node_modules` 后按 README 重新安装，压缩包不应包含它。

## 4. 推荐执行顺序

### 阶段 A：人工与模型感知校准

1. 人工快速抽查 `core/calibration/images/`，重点看网格/棋盘格、波纹/折线纹、六边形/菱形是否容易混淆。
2. 用 `core/calibration/inputs.jsonl` 让待测模型识别颜色、形状、纹理。
3. 用 `core/calibration/labels.jsonl` 判分。校准必须与正式试次分开，不把标签或文件名传入模型。
4. 若某些组合辨认率明显偏低，先记录问题；不要用 test 结果筛选刺激。必要时只在 dev 阶段决定是否调整流程，并记录刺激版本。

### 阶段 B：冻结模型调用协议

在 `core/inputs/dev.jsonl` 的 216 道题上确定：模型版本、提示词、图片尺寸处理、温度/随机性、请求顺序、超时和 `YES/NO` 解析规则。每题都建立全新会话；同一组的下一题也不能沿用上下文。确认后把配置写入结果目录的运行清单，再开始 test。

### 阶段 C：运行正式 test

对 `core/inputs/test.jsonl` 的 648 道题逐题提交：

1. 发送 `instruction`。
2. 按数组顺序逐张发送 `study_images`。
3. 若 `distractor_images` 非空，逐张发送并说明它们不计入学习集合。
4. 发送一张 `test_image`。
5. 只保存模型原始回答和解析状态；不要把答案、题型、属性字段或路径文本放入请求。

调用顺序可以随机化，但一题内部的图片顺序不能改变。若模型回答含有解释文字，按预先冻结的规则标记为 invalid；invalid 单独统计，并按错误计入准确率。

### 阶段 E：运行图片学习、文字探测扩展

该扩展沿用相同学习列表、干扰图片、试次 ID 和答案标签，只把最后的测试图片替换为文字问题，例如询问“在刚才展示的学习图片中，是否出现过一个红色圆形、表面具有点状纹理的物体？”。模型输入不包含测试图片或结构化属性字段。使用独立结果目录评分，并与原图片探测条件按 `trial_id` 逐题配对；不要合并两条轨道的准确率。提交 `results/run_qwen_text_query.sbatch` 运行。

### 阶段 D：判分与报告

用 `trial_id` 将原始回答连接到 `core/labels/{dev,test}.jsonl`。至少报告：

- 总体准确率、invalid 比例；
- target 命中率；
- `critical_recombination`、`near_lure`、`unrelated_lure` 各自的错误接受率；
- 学习图数量 4/8/12 和干扰图数量 0/4 的条件结果；
- 同一种子、学习量和测试题在 0/4 干扰条件下的配对差值。

置信区间或显著性检验按 seed 或 study-list 整组重采样。因为同一学习列表共享学习图和 target，不能把 6 道题当成 6 个完全独立样本。

## 5. 推荐结果目录

建议在本目录建立以下结构；原始数据保持不动：

```text
results/
  README.md                 # 已提供：字段、命名和判分口径
  run_manifest.json         # 模型、版本、参数、提示词、数据版本
  calibration_raw.jsonl     # 校准原始回答
  dev_raw.jsonl             # dev 原始回答
  test_raw.jsonl            # test 原始回答
  scored.jsonl              # 连接 labels 后的逐题结果
  summary.json              # 汇总指标
  figures/                  # 条件准确率和错误接受率图
```

建议每行原始结果至少包含：`trial_id`、`model`、`model_version`、`timestamp`、`temperature`、`raw_response`、`parsed_answer`、`valid`、`latency_ms`、`error`。不要在发送给模型的请求中使用这些审计字段。

## 6. 统计与解释边界

主实验的正负样本已经平衡，因此总体准确率可以作为辅助指标；核心结论应同时看 target 命中率和三类诱饵错误接受率。补充基准有 20% YES、80% NO，全部回答 NO 也能得到 80% 微平均准确率，所以它必须报告 target 命中率、各诱饵错误接受率和宏平均，不能与主实验合并。

学习图始终留在当前会话历史中。因此该实验测量的是**当前上下文中的物体属性绑定识别**，不能直接声称测量了清除原图后的长期记忆。若要研究保持时间或跨会话记忆，需要另行规定延迟、上下文清除和可见性协议。

## 7. 最小交付物

完成项目时建议交付：

1. 一份固定的 `run_manifest.json`；
2. 校准、dev、test 的原始 JSONL；
3. 带标准答案连接的 `scored.jsonl`；
4. 按 probe type、学习量、干扰量和 seed 聚类的汇总表/图；
5. 一页方法说明，明确主实验与补充基准分开统计，并写出 invalid 处理和解释边界。
