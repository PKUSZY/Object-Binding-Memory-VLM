# 实验结果目录

这里保存模型运行记录和统计结果；数据集本身仍在 `binding_memory_integrated/core/`。

## 校准准备与运行

`calibration_review/` 中已有 432 条空白响应模板、6 张供人工抽查的无标签 contact sheet、单独保存的人工答案表和运行清单模板。contact sheet 和答案表只用于人工核验，不能传给模型。

NVIDIA GPU 安装与运行说明见 [QWEN_LOCAL_SETUP.md](QWEN_LOCAL_SETUP.md)。本地 runner 是 `run_qwen_local_calibration.py`；`run_qwen_calibration.py` 用于托管 OpenAI-compatible endpoint。

在能访问 NVIDIA GPU 的 Linux 终端按安装指南配置后，先运行 12 张 pilot 并评分：

```bash
python results/run_qwen_local_calibration.py --limit 12
python results/score_calibration.py \
  --raw results/calibration_review/qwen_local_pilot_raw.jsonl \
  --allow-subset
```

pilot 确认模型加载、输出解析与图片路径正常后，运行完整校准：

```bash
python results/run_qwen_local_calibration.py \
  --limit 432 \
  --output results/calibration_review/qwen_local_calibration_raw.jsonl
python results/score_calibration.py \
  --raw results/calibration_review/qwen_local_calibration_raw.jsonl \
  --out results/calibration_review/qwen_local_scored
```

本地脚本每张图片单独推理，校准 ID、图片路径和答案标签不发给模型，并支持从已有输出继续。不要以校准表现为由查看或筛选正式 test 题。

正式实验提交脚本见 [run_qwen_binding.sbatch](run_qwen_binding.sbatch)，正式 runner 为 `run_qwen_local_experiment.py`。Slurm 会保存 `slurm-qwen-binding-<jobid>.out` 和 `.err`；runner 还会逐题追加 `results/experiment_raw/progress.jsonl`，其中包含 `trial_start`、`trial_end`、原始回答、解析结果、耗时和错误。`dev.jsonl`、`test.jsonl` 是可用于最终判分的逐题原始结果。

任务完成后，`score_experiment.py` 会生成 `results/experiment_scored/scored.jsonl` 和 `summary.json`。主实验应得到 864 道结果：dev 216、test 648；汇总包含总体准确率、invalid 比例、target 命中率、三类诱饵错误接受率，以及 4/8/12 学习图和 0/4 干扰图条件结果。若任务中途中断，重新提交同一个 sbatch 会跳过已存在的 `trial_id`，继续未完成题目。

## 图片学习、文字探测扩展

`run_qwen_text_query.sbatch` 运行配对文字探测条件：学习图和干扰图与主实验相同，最后以文字询问某个颜色、形状和纹理组合是否出现过，不发送测试图片。该轨道与图片探测主实验分开评分，便于按 `trial_id` 比较两种探测方式。

```bash
sbatch results/run_qwen_text_query.sbatch
```

生成器和推理器分别为 `generate_text_query_benchmark.py`、`run_qwen_text_query_experiment.py`；完成后由 `score_text_query_experiment.py` 生成单独的逐题结果与汇总。任务支持断点续跑，不会覆盖主实验结果。

## 原始回答格式

正式测试每道题一行 JSON，至少记录：

```json
{
  "trial_id": "s07_n08_d4_t3",
  "model": "模型名称",
  "model_version": "版本或日期",
  "timestamp": "ISO-8601 时间",
  "temperature": 0,
  "raw_response": "YES",
  "parsed_answer": "YES",
  "valid": true,
  "latency_ms": 0,
  "error": null
}
```

`trial_id` 用于连接 `core/labels/{dev,test}.jsonl`；请求发送给模型时只使用 `core/inputs/{dev,test}.jsonl` 的白名单字段。保留 `raw_response`，便于复核解析规则。

## 判分口径

- 明确的 `YES`/`NO` 才算有效；其他输出记为 `invalid`，并按错误计入准确率。
- `target` 报命中率。
- 三类 lure 报错误接受率（模型回答 `YES` 的比例）和正确拒绝率。
- 置信区间按 seed 或学习列表整组重采样。
- 主实验 `core` 与 `auxiliary/three_study_v0.1` 永远分开汇总。

推荐结果文件：`run_manifest.json`、校准/dev/test 原始 JSONL、`scored.jsonl`、`summary.json` 和图表。不要把运行令牌或 API key 保存到本目录。
