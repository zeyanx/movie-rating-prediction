# 第3阶段：特征工程与集成学习模型

## 1. 实验目标与数据划分

本阶段在第2阶段固定的80000条训练记录上构造无泄漏特征，对随机森林、AdaBoost和XGBoost进行5折交叉验证，再用交叉验证RMSE选择最佳集成模型。20000条固定测试记录只用于三个最终模型的一次性评估，没有参与特征参数学习、交叉验证或模型选择。

## 2. 特征工程

最终完整训练集Pipeline产生 **54** 个数值特征。

| 特征组 | 主要内容 | 设计理由 |
|---|---|---|
| 用户属性 | 年龄、性别独热、职业独热 | 表达不同用户群体的评分差异 |
| 电影类型 | 19类电影多标签特征 | 表达影片内容类型 |
| 时间 | 上映年、评分年/月/星期/小时、影片年龄 | 表达年代和评分时间影响 |
| 训练统计 | 用户/电影平滑均值、对数计数、均值差 | 表达偏好与受欢迎程度 |

`user_id` 和 `movie_id` 是任意编号，直接作为连续数值会产生不存在的大小与距离关系，因此本阶段只将它们作为训练统计映射的键。`interaction_id`、评分目标、标题、IMDb链接和邮政编码均未进入特征矩阵。

## 3. 平滑统计与逐行留一

对验证集或测试集，用户平滑均值为：

$$
\mu_u = \frac{S_u + \alpha_u\mu}{N_u+\alpha_u}
$$

其中用户平滑系数为10，电影平滑系数为20。未见用户或电影回退到当前训练部分的全局均值，对应计数为0。

对训练记录$j$，使用逐行留一统计：

$$
\mu_{u,-j} = \frac{S_u-r_j+\alpha_u\mu_{-j}}{N_u-1+\alpha_u}
$$

这保证当前记录自己的评分不会进入其目标统计特征。每一折Pipeline只在折内训练部分拟合类别、填充值和统计映射，再转换验证折，从而避免交叉验证泄漏。

## 4. 三种模型

- **随机森林**：通过多棵随机化决策树平均降低方差，具有稳定、可解释的重要性；缺点是模型较大、训练和预测成本较高。
- **AdaBoost**：依次关注前序弱学习器误差，以浅层回归树提升拟合能力；优点是结构清晰，缺点是对异常点和参数较敏感。
- **XGBoost**：采用梯度提升、子采样和L1/L2正则化逐步优化残差，通常具有较强预测能力；缺点是参数较多且需要控制复杂度。

固定参数如下：

```json
{
  "random_forest": {
    "n_estimators": 200,
    "max_depth": 18,
    "min_samples_leaf": 2,
    "max_features": 0.7,
    "random_state": 42,
    "n_jobs": -1
  },
  "adaboost": {
    "estimator": {
      "max_depth": 6,
      "min_samples_leaf": 5,
      "random_state": 42
    },
    "n_estimators": 200,
    "learning_rate": 0.05,
    "loss": "square",
    "random_state": 42
  },
  "xgboost": {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1
  }
}
```

## 5. 五折交叉验证结果

使用 `KFold(n_splits=5, shuffle=True, random_state=42)`，每条训练记录恰好作为验证样本一次。最佳模型仅按平均验证RMSE选择。

| 模型 | CV RMSE | CV MAE | CV R² |
|---|---:|---:|---:|
| random_forest | 0.955915 ± 0.006563 | 0.759745 ± 0.004606 | 0.278820 ± 0.005972 |
| xgboost | 0.970373 ± 0.005091 | 0.771375 ± 0.004600 | 0.256837 ± 0.004764 |
| adaboost | 1.048129 ± 0.019593 | 0.859117 ± 0.015023 | 0.132836 ± 0.028066 |

## 6. 固定测试集结果

第2阶段全局均值基线测试RMSE为1.125659。

| 模型 | RMSE | MAE | R² | 相对基线RMSE改善 |
|---|---:|---:|---:|---:|
| random_forest | 0.957029 | 0.759237 | 0.277170 | 14.98% |
| adaboost | 1.062963 | 0.871165 | 0.108292 | 5.57% |
| xgboost | 0.970324 | 0.770739 | 0.256947 | 13.80% |

根据平均5折CV RMSE，选择 **random_forest** 作为最佳集成模型，其CV RMSE为0.955915。该决定在查看测试指标之前完成，测试集结果不参与选择。

## 7. 特征重要性

以下重要性来自各模型在完整训练集上拟合后的原生 `feature_importances_`，归一化后各模型总和为1。重要性表示模型分裂贡献，不直接等价于因果影响。

### random_forest

| 排名 | 特征 | 分组 | 归一化重要性 |
|---:|---|---|---:|
| 1 | movie_smoothed_mean | 训练集统计 | 0.334310 |
| 2 | user_smoothed_mean | 训练集统计 | 0.306108 |
| 3 | movie_rating_count_log | 训练集统计 | 0.055678 |
| 4 | user_movie_mean_difference | 训练集统计 | 0.049518 |
| 5 | user_rating_count_log | 训练集统计 | 0.040389 |
| 6 | user_age | 用户属性 | 0.032435 |
| 7 | rating_hour | 时间 | 0.025212 |
| 8 | release_year | 时间 | 0.019330 |
| 9 | movie_age_at_rating | 时间 | 0.017462 |
| 10 | rating_day_of_week | 时间 | 0.016090 |

### adaboost

| 排名 | 特征 | 分组 | 归一化重要性 |
|---:|---|---|---:|
| 1 | movie_smoothed_mean | 训练集统计 | 0.403932 |
| 2 | user_smoothed_mean | 训练集统计 | 0.321353 |
| 3 | movie_rating_count_log | 训练集统计 | 0.051751 |
| 4 | user_rating_count_log | 训练集统计 | 0.051591 |
| 5 | release_year | 时间 | 0.028211 |
| 6 | user_age | 用户属性 | 0.026609 |
| 7 | user_movie_mean_difference | 训练集统计 | 0.015193 |
| 8 | rating_hour | 时间 | 0.014370 |
| 9 | movie_age_at_rating | 时间 | 0.008858 |
| 10 | rating_day_of_week | 时间 | 0.008697 |

### xgboost

| 排名 | 特征 | 分组 | 归一化重要性 |
|---:|---|---|---:|
| 1 | movie_smoothed_mean | 训练集统计 | 0.056709 |
| 2 | user_smoothed_mean | 训练集统计 | 0.041942 |
| 3 | occupation=other | 用户属性 | 0.031481 |
| 4 | occupation=writer | 用户属性 | 0.028986 |
| 5 | occupation=engineer | 用户属性 | 0.027831 |
| 6 | occupation=educator | 用户属性 | 0.025449 |
| 7 | user_movie_mean_difference | 训练集统计 | 0.025128 |
| 8 | movie_rating_count_log | 训练集统计 | 0.023669 |
| 9 | user_rating_count_log | 训练集统计 | 0.023174 |
| 10 | gender=M | 用户属性 | 0.022888 |

## 8. 冷启动处理与局限性

固定测试集中训练阶段未见电影使用训练全局均值作为电影统计回退值，计数特征为0；静态电影类型和时间特征仍可使用。该方法不能完全解决新电影冷启动，只是提供确定、无泄漏的回退方案。

本阶段只比较三种集成学习方法，参数为事先固定的课程实验配置，并未进行大规模超参数搜索。目标统计特征依赖历史显式评分，在线系统还需考虑时间顺序、概念漂移和新用户问题。集成学习与神经网络的最终对比必须等第4阶段神经网络和第5阶段统一实验完成后再下结论。
