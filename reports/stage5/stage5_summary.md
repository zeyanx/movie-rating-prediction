# 第5阶段：集成学习与神经网络对比实验

## 阶段结论

前置第2、3、4阶段验证全部通过。本阶段未重新训练任何模型，在相同20000条固定测试记录上统一复算指标。

| model | rmse | mae | r2 | rmse_improvement_vs_baseline_percent | rmse_rank |
|---|---|---|---|---|---|
| mlp | 0.929017 | 0.733321 | 0.318865 | 17.469080 | 1 |
| random_forest | 0.957029 | 0.759237 | 0.277170 | 14.980609 | 2 |
| xgboost | 0.970324 | 0.770739 | 0.256947 | 13.799463 | 3 |
| adaboost | 1.062963 | 0.871165 | 0.108292 | 5.569713 | 4 |
| global_mean | 1.125659 | 0.944690 | -0.000000 | 0.000000 | 5 |

测试集排名第一为 **mlp**。第3阶段按训练集内部5折CV选择的最佳集成模型仍为 **random_forest**；第4阶段神经网络为 **mlp**。

## 成对Bootstrap

| comparison | metric | point_estimate_delta | confidence_lower | confidence_upper | mlp_win_probability |
|---|---|---|---|---|---|
| mlp_vs_random_forest | rmse | -0.028012 | -0.032625 | -0.023293 | 1.000000 |
| mlp_vs_random_forest | mae | -0.025916 | -0.029876 | -0.021795 | 1.000000 |
| mlp_vs_adaboost | rmse | -0.133946 | -0.141541 | -0.126538 | 1.000000 |
| mlp_vs_adaboost | mae | -0.137844 | -0.144595 | -0.131255 | 1.000000 |
| mlp_vs_xgboost | rmse | -0.041307 | -0.046687 | -0.035558 | 1.000000 |
| mlp_vs_xgboost | mae | -0.037418 | -0.041679 | -0.032796 | 1.000000 |

差异定义为MLP减集成模型，负值表示MLP更好。置信区间不跨0时，只表述为本测试集上的差异较稳定。

## 分组与残差

- 未见电影：30部、33条测试记录。
- MLP整体平均误差：0.014584。
- MLP近似命中率（绝对误差≤0.5）：40.84%。
- 用户活跃度、电影流行度阈值只由训练集计算，未使用测试评分次数。

## 重要性说明

集成模型使用原生树分裂重要性；MLP使用8000条内部验证集的分组排列重要性。两类数值不可直接等比例比较，也不代表因果关系。

MLP排列重要性前五项：

| feature_group | rmse_increase_mean | rmse_increase_std | rank |
|---|---|---|---|
| movie_embedding | 0.160728 | 0.004746 | 1 |
| user_embedding | 0.160705 | 0.005041 | 2 |
| genres | 0.029721 | 0.002120 | 3 |
| release_year | 0.013841 | 0.001294 | 4 |
| movie_age_at_rating | 0.012059 | 0.001360 | 5 |

## 效率和第6阶段建议

| model | artifact_size_mb | load_time_mean_seconds | inference_time_mean_seconds | rows_per_second |
|---|---|---|---|---|
| global_mean | 0.000250 | 0.000305 | 0.000022 | 892060612.890679 |
| random_forest | 37.995707 | 1.270755 | 0.427360 | 46798.962654 |
| adaboost | 0.554913 | 0.083253 | 0.930224 | 21500.197802 |
| xgboost | 0.551972 | 0.144019 | 0.268250 | 74557.393760 |
| mlp | 0.548397 | 0.017554 | 1.659482 | 12051.950511 |

建议第6阶段默认模型：**mlp**；最佳集成对照：**random_forest**；故障兜底：**global_mean**。该建议使用测试结果做工程选择，后续不得继续使用同一测试集调参。

## 局限

数据规模较小、使用随机划分而非时间划分、树模型与MLP特征路线不同，且没有未见用户测试样本。结果适用于本课程实验范围。

## 主要产物

- `metrics_comparison.*`：统一指标；
- `paired_bootstrap_summary.*`：成对Bootstrap；
- `segment_metrics.csv`、`residual_summary.csv`：误差分析；
- `ensemble_*importance.csv`、`mlp_permutation_importance*.csv`：解释性；
- `efficiency_benchmark.csv`：本机CPU效率；
- 五个Plotly HTML：论文和答辩图表；
- `paper_results_material.md`：论文结果素材。

第6阶段尚未开始。
