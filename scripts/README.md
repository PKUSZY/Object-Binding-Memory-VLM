# 脚本索引

从仓库根目录运行；Slurm 分区、节点排除列表、CUDA module 和 Conda 环境名来自原集群，迁移时先核对这些配置。

| 用途 | 脚本 |
| --- | --- |
| Qwen 通用生成／推理／评分 | `run_bindmem.sh`、`bindmem_gpu.sbatch` |
| Qwen 按任务分批运行 | `bindmem_basic.sbatch`、`bindmem_image.sbatch`、`bindmem_choices.sbatch`、`bindmem_formats.sbatch`、`bindmem_extensions.sbatch` |
| 下载并固定 LLaVA 权重版本 | `prepare_llava.py` |
| LLaVA／Qwen 补充实验及匹配梯度 | `run_bindmem_extra.sh`、`bindmem_llava.sbatch` |
| 原集群精简实验续跑 | `prepare_bindmem_priority.py`、`run_bindmem_priority.sh`、`bindmem_llava_priority.sbatch` |
| Qwen 六批结果合并 | `summarize_bindmem_runs.py` |
| Qwen 图表、实际输入输出与逐任务报告 | `build_bindmem_visual_report.py`、`bindmem_report_io.py`、`bindmem_report_tasks.py` |
| 双模型核心比较报告 | `report_bindmem_priority.py`；`report_bindmem_priority_dev.py` 仅生成探索性 dev 预览 |
| 完整多版本项目报告 | `build_complete_project_report.py`、`report_pdf.py` |
| CUDA 启动诊断 | `check_bindmem_cuda.py` |
| 历史综合检查／探针 | `check_bindmem.sh`、`bindmem_probes.sbatch`；包含历史表示采集／探针检查，不是当前行为实验的必要步骤 |

当前行为代码的 CPU 检查：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

`run_bindmem_priority.sh` 是原运行目录的续跑与汇总脚本，依赖先前 Qwen 三批核心结果、Qwen 梯度及 LLaVA 复用来源；新克隆不要直接把它当作从零启动命令。新机器的最小流程见[根 README](../README.md)。

完整报告生成脚本会核对原始运行文件，需要本地 `data/`、`runs/` 中的全部正式结果，不能只用 GitHub 上的汇总表重新运行推理或重建全部证据。已发布的 Markdown、图片和表格可直接阅读。排版所用中文字体为 `/usr/share/fonts/google-droid/DroidSansFallback.ttf`；其他机器需提供该字体或调整绘图脚本中的字体路径。PDF 的可选依赖也可安装在独立报告环境中。
