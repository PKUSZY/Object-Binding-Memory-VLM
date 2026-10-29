# BindMem 完整报告交付包

日期：2026-10-29。正式结果：Qwen 原基准 test 69,840 题、梯度 test 5,760 题；LLaVA 同题 test 5,520 题。

| 阅读用途 | 文件 |
| --- | --- |
| 完整内容、可编辑 | [完整 Markdown](report.md) |
| 浏览器阅读、单文件分享正文 | 本地导出 `report.html`，正文图片全部内嵌；大型附件不上传 GitHub |
| 打印、提交、全文搜索 | 本地导出 `report.pdf`，包含完整正文；PDF 不上传 GitHub |
| 面向不了解项目的读者 | [图文简版 MD](brief.md)、[HTML](brief.html)；PDF 为本地导出 `brief.pdf` |
| 单独使用图表 | [14 张 PNG/SVG 图](figures/)；图集 PDF 为本地导出 `figure_atlas.pdf` |
| 检查每项分数 | [逐任务附表 MD](appendix_qwen_tasks.md)、[HTML](appendix_qwen_tasks.html) |
| 查看真实输入输出 | [17 例说明 MD](appendix_input_output.md)；大型 HTML 为本地导出 `appendix_input_output.html` |
| 查看强重组错误 | [Qwen](cases_qwen.html)、[LLaVA 核心](cases_llava_core.html)、[LLaVA 梯度](cases_llava_gradient.html) |
| 核对数值与来源 | [完整精度 CSV/JSON](tables/)、[案例证据](evidence/)、[来源文件哈希](source_audit.json) |
| 一次性打包分享 | 本地导出 `report_bundle.zip`；不上传 GitHub |

Markdown 的图片使用相对路径，请保留整个目录结构。每个 HTML 的正文图片均已内嵌，可单独打开；链接到其他报告或证据时需保留对应附件。CSV 保存原始精度，正文按百分比保留两位；旧汇总与同题重新汇总的 bootstrap 随机组遍历顺序可能不同，置信区间有少量 Monte Carlo 差异。

重新生成：在仓库根目录运行 ` /ceph/home/shizhongyu/anaconda3/bin/python scripts/build_complete_project_report.py `。PDF 依赖仅位于 `/tmp/bindmem-report-deps`；若被清理，可用同一 Python 执行 `-m pip install --target /tmp/bindmem-report-deps reportlab pypdf`。不改变模型推理环境，不需要 GPU，不运行模型。

本交付包含图表与案例证据，不包含模型权重和全量原始数据。原始运行、生成／推理脚本及数据在项目仓库，完整报告第 10 节提供索引。
