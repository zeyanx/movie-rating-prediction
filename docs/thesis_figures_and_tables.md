# 论文图表索引与引用说明

|编号|题名|数据来源|建议引用位置|
|---|---|---|---|
|表2-1|MovieLens 100K字段与规模|`data/raw/dataset_info.json`|2.1节|
|表4-1|五模型统一测试集指标|`reports/stage5/metrics_comparison.csv`|4.1节|
|表4-2|配对bootstrap置信区间|`reports/stage5/paired_bootstrap_summary.csv`|4.1节|
|表4-3|模型文件与CPU推理效率|`reports/stage5/efficiency_benchmark.csv`|4.4节|
|图4-1|RMSE/MAE/R²对比|`reports/stage5/metrics_comparison.html`|4.1节|
|图4-2|残差与误差分布|`reports/stage5/residual_analysis.html`|4.2节|
|图4-3|分群误差对比|`reports/stage5/segment_comparison.html`|4.2节|
|图4-4|集成模型与MLP特征组重要性|`reports/stage5/feature_importance_comparison.html`|4.3节|
|图4-5|文件大小与推理时间|`reports/stage5/efficiency_comparison.html`|4.4节|

注意：HTML 图可在浏览器中导出 PNG/SVG 后插入最终论文；图注必须写明“固定测试集 n=20000、随机种子42”。不要把网页截图当作实验原始图，也不要把相关性或置换重要性表述为因果关系。
