合并完成；原始数据未移动或修改。
训练配置：dataset.yaml
新图片使用 garbage2/images 无框原图，标签来自 garbage2/data/labels；未使用带框预览。
测试集：image418-447、image488-527、image598-617，共90张，均来自garbage2。
新增验证集：image358-397，共40张；保留旧验证集。
逐张来源见 split_manifest.csv；统计和去重见 merge_report.json。
注意：自动标注不等于人工正确答案。训练前检查新标签，尤其测试标签需独立人工核查，否则测试指标会受到原模型预测错误影响。
按连续编号块留出图片不保证不同视频/场景独立；旧数据随机划分保留，仍可能存在近似场景跨集合问题。

2026-10-31：按用户已完成人工审核的 garbage2/data/labels 同步419份标签，其中142份内容更新。图片未变，train/val/test划分未变。旧标签备份：D:\learning\FPGA\Guidance\dataset_inspection\OurYOLOData_before_label_update_20261031_174207。最新统计见 label_update_report.json。
