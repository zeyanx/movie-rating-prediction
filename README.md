# 基于集成学习与神经网络的电影评分预测对比研究

本项目使用 MovieLens 100K 数据，对集成学习与神经网络评分预测方法进行对比，并提供可公开访问的 Streamlit 电影评分预测和个性化推荐系统。

第1至第7阶段均已完成：数据治理、固定划分、基线、集成学习、PyTorch MLP、统一实验、六页面网页、公开部署、论文与答辩材料均已验收。

- GitHub：<https://github.com/zeyanx/movie-rating-prediction>
- 在线应用：<https://mccr78dsncofymbfxhdwfo.streamlit.app/>

## 1. Windows环境创建

在 Anaconda Prompt 中进入项目根目录，然后执行：

```powershell
conda create -n movie python=3.10 -y
conda activate movie
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果环境已存在，不要删除，可直接检查并补齐依赖：

```powershell
conda run -n movie python --version
conda run -n movie python -m pip install -r requirements.txt
```

`sqlite3` 是 Python 标准库的一部分，无需也不应通过 pip 单独安装。

## 2. 下载并转换MovieLens 100K

```powershell
conda activate movie
python scripts/download_movielens.py
```

脚本从 GroupLens 官方地址下载 `ml-100k.zip`，验证官方 MD5，然后把 `u.data`、`u.item`、`u.user` 转换为：

- `data/raw/ratings.csv`
- `data/raw/movies.csv`
- `data/raw/users.csv`
- `data/raw/dataset_info.json`

重复执行时，已经通过快速检查的数据不会再次下载。需要完全重建时执行：

```powershell
python scripts/download_movielens.py --force
```

网络或代理受限时，请从以下官方地址手动下载：

<https://files.grouplens.org/datasets/movielens/ml-100k.zip>

例如将文件保存为 `D:\Downloads\ml-100k.zip` 后执行：

```powershell
python scripts/download_movielens.py --archive "D:\Downloads\ml-100k.zip"
```

## 3. 运行第1阶段验收

```powershell
conda activate movie
python scripts/verify_stage1.py
```

验收脚本会检查 Python 3.10、全部依赖、CSV字段、100000条评分、943个用户、1682部电影、评分范围、主键和引用完整性。成功时最后输出 `第1阶段验证通过`。

## 4. 在PyCharm中选择解释器

1. 打开 `File > Settings > Project > Python Interpreter`。
2. 选择 `Add Interpreter > Add Local Interpreter`。
3. 选择 `Conda Environment > Existing environment`。
4. 选择 `movie` 环境中的 `python.exe`。

可以用下列命令获得准确路径：

```powershell
conda run -n movie python -c "import sys; print(sys.executable)"
```

选好后，在 PyCharm Terminal 中运行 `python --version`，应显示 Python 3.10。

## 5. 数据来源与许可

数据来自 GroupLens Research 的稳定基准数据集 MovieLens 100K：

- 数据集：<https://grouplens.org/datasets/movielens/100k/>
- README与使用条款：<https://files.grouplens.org/datasets/movielens/ml-100k-README.txt>

课程论文、演示或公开部署前，应阅读并遵守官方README中的使用条款，并在论文中注明数据来源。`data/external/` 不纳入版本控制，可由下载脚本重新生成；三个规范CSV未被 `.gitignore` 忽略。

## 6. 常见问题

| 问题 | 解决方法 |
|---|---|
| PowerShell提示找不到`conda` | 使用Anaconda Prompt，或执行`conda init powershell`后重启PowerShell |
| 无法激活环境 | 使用`conda run -n movie python ...`直接在指定环境运行 |
| `conda run`回显中文时出现GBK编码错误 | 直接运行`movie`环境中的`python.exe`，或先设置`$env:PYTHONIOENCODING='utf-8'` |
| PyCharm仍使用其他Python | 重新选择`movie`环境中的`python.exe`并检查终端中的`sys.executable` |
| 下载超时或SSL/代理错误 | 从官方地址手动下载ZIP，再使用`--archive`参数 |
| MD5校验失败 | 删除损坏的ZIP并从官方地址重新下载，不要使用未知镜像 |
| 电影名乱码 | 不要修改脚本编码；官方`u.item`按`latin-1`读取，CSV按UTF-8写入 |
| PyTorch安装较慢 | 保持网络连接；本项目最低要求为CPU可运行，不需要CUDA工具包 |

## 7. 第2阶段：固定划分与全局均值基线

第2阶段完成数据质量检查、固定训练/测试划分和全局均值基线。划分采用 `test_size=0.20`、`random_state=42`、启用随机打乱，并按 `rating` 分层。所有后续模型必须直接复用 `data/processed/train_ratings.csv` 和 `data/processed/test_ratings.csv`，不得重新随机划分。

全局均值严格只使用训练集计算：

```text
global_mean = train_ratings["rating"].mean()
```

测试集不参与均值或任何训练参数估计，从而避免数据泄漏。

### 7.1 新增结构

```text
src/movie_rating/data.py                 数据读取、校验、质量统计与固定划分
src/movie_rating/baseline.py             全局均值基线及指标计算
scripts/run_stage2.py                     第2阶段主入口
scripts/verify_stage2.py                  第2阶段独立验收
data/processed/train_ratings.csv          固定训练集
data/processed/test_ratings.csv           固定测试集
data/processed/split_manifest.json        划分配置、哈希和冷启动统计
models/global_mean_baseline.joblib        可加载的基线模型
reports/stage2/data_quality.json          数据质量报告
reports/stage2/baseline_metrics.json      RMSE、MAE和R²
reports/stage2/test_predictions.csv       测试集逐行预测与误差
reports/stage2/stage2_summary.md           可用于论文的阶段素材
```

### 7.2 运行与验证

```powershell
conda activate movie
python scripts/verify_stage1.py
python scripts/run_stage2.py
python scripts/verify_stage2.py
```

重复运行主程序会依据相同数据、分层方法和随机种子生成相同划分与指标。`split_manifest.json` 中记录原始评分文件和两个划分文件的SHA-256；生成时间允许变化，但核心实验结果必须保持一致。

### 7.3 第2阶段常见问题

| 问题 | 解决方法 |
|---|---|
| 找不到原始CSV | 先运行`scripts/download_movielens.py`和`scripts/verify_stage1.py` |
| Python解释器错误 | 在PyCharm中选择`movie`环境的`python.exe` |
| CSV字段不一致 | 不要手动改动`data/raw`；重新运行第1阶段转换程序 |
| 训练集与测试集重叠 | 删除手动生成的划分并运行`scripts/run_stage2.py`重建 |
| 全局均值与报告不一致 | 确认均值只从`train_ratings.csv`计算，禁止使用全量评分 |
| 测试集出现未见电影 | 这是随机划分的冷启动边界，保留并查看清单统计，不要偷偷移动记录 |
| 指标出现NaN | 检查评分、预测是否有限，并确认测试集非空 |
| 模型无法加载 | 使用相同环境中的joblib重新运行主程序保存模型 |
| 重复运行后划分变化 | 检查是否仍使用随机种子42、按rating分层，并且原始CSV哈希未变化 |

## 8. 第3阶段：特征工程与集成学习

第3阶段严格复用第2阶段的80000条训练记录和20000条测试记录。只在训练集内部使用 `KFold(n_splits=5, shuffle=True, random_state=42)` 比较随机森林、AdaBoost和XGBoost，并按平均CV RMSE选择最佳模型。测试集不参与特征参数学习、调参或模型选择。

### 8.1 无泄漏特征工程

最终Pipeline产生54个数值特征，主要包括：

- 用户年龄、性别独热和职业独热；
- 电影类型多标签特征；
- 上映年、评分年/月/星期/小时和影片年龄；
- 用户与电影平滑评分均值、对数评分次数和均值差。

训练行的用户/电影目标统计采用逐行留一计算，当前评分不会进入自己的统计特征。每个CV折只用折内训练部分学习类别、上映年中位数、全局均值和用户/电影映射。验证集及测试集中的未见ID回退到当前训练均值，计数置为0。

`rating`、`interaction_id`、标题、IMDb链接、邮编以及原始数值ID不进入模型特征。

### 8.2 固定模型配置

- 随机森林：200棵树，最大深度18，叶节点最少2个样本，`max_features=0.7`。
- AdaBoost：200个最大深度6的回归树，学习率0.05，平方损失。
- XGBoost：400棵树，最大深度6，学习率0.05，行列采样0.8，`tree_method=hist`。
- 所有随机种子均为42，预测统一裁剪到1至5分。

### 8.3 运行命令

```powershell
conda activate movie
python scripts/verify_stage2.py
python scripts/run_stage3.py
python scripts/verify_stage3.py
```

普通CPU环境下，AdaBoost的完整5折训练可能需要数分钟。为避免内存和线程争用，外层CV顺序运行，随机森林和XGBoost仅在模型内部并行。

### 8.4 输出文件

```text
configs/stage3_models.json                 固定模型与CV配置
src/movie_rating/features.py               留一目标统计特征工程
src/movie_rating/ensemble.py               裁剪回归器、模型和CV工具
models/stage3/*.joblib                     三个模型及最佳模型
data/processed/feature_schema.json         特征名称、分组与排除字段
reports/stage3/cv_fold_results.csv         15条折级结果
reports/stage3/cv_summary.csv              CV均值、标准差和排名
reports/stage3/test_metrics.*              固定测试集指标
reports/stage3/test_predictions.csv        三个模型逐行预测
reports/stage3/feature_importance.csv       原生特征重要性
reports/stage3/*.html                      Plotly对比图
reports/stage3/stage3_summary.md            论文阶段素材
```

### 8.5 常见问题

| 问题 | 解决方法 |
|---|---|
| 固定划分文件或哈希异常 | 先运行`scripts/verify_stage2.py`，不要自行重新划分 |
| 特征误含评分目标 | 确保传入Pipeline前已从X中移除`rating` |
| 目标统计特征泄漏 | 训练转换必须使用`fit_transform`中的逐行留一逻辑 |
| 未见用户或电影产生NaN | 回退到训练部分全局均值并将计数置0 |
| 类别列数量不一致 | 类别仅在`fit`中学习，`transform`按已学习列重建 |
| AdaBoost参数报错 | 当前版本应使用`estimator=`，不要使用废弃参数名 |
| XGBoost导入或DLL失败 | 在`movie`环境重新安装兼容的xgboost并运行`pip check` |
| 内存或线程占用过高 | 保持外层CV为`n_jobs=1`，避免多个模型同时训练 |
| 测试结果被用于选模 | 只能按`cv_summary.csv`的平均CV RMSE选择模型 |
| 重复运行指标变化 | 检查随机种子、固定配置和训练/测试文件哈希 |

后续神经网络也必须复用相同固定测试集。集成学习与神经网络的最终结论应在第4、5阶段完成后给出。

## 9. 第4阶段：PyTorch Embedding + MLP

第4阶段严格复用第2阶段的80000/20000固定划分。仅从80000条训练记录中按评分分层划出72000条内部训练记录和8000条验证记录，使用验证RMSE早停并选择最佳epoch；随后在完整80000条训练数据上从头训练最终模型，固定测试集只在模型保存后评估一次。

### 9.1 输入与无泄漏策略

- 用户、电影、职业和性别使用Embedding，索引0保留给未知类别；
- 年龄、上映年、评分年和影片年龄使用训练集参数标准化；
- 评分月份、星期、小时使用正余弦周期编码；
- 使用19个MovieLens官方电影类型多热特征；
- 不使用用户/电影评分均值、评分次数、平滑均值或目标编码；
- `rating`仅作为标签，`interaction_id`仅用于追踪，标题、IMDb链接和邮编不进入模型。

网络隐藏层为128、64、32，每层使用Linear、LayerNorm、ReLU和Dropout，输出通过`1 + 4 × sigmoid(raw)`限制在1至5。训练使用MSELoss和AdamW，默认CPU、随机种子42、Windows DataLoader工作进程数为0。

### 9.2 训练与验证

```powershell
conda activate movie
python scripts/run_stage4.py
python scripts/verify_stage4.py
```

如果当前PowerShell无法识别Conda，可以在Anaconda Prompt执行；如果`conda run`出现中文编码问题，可以直接选择PyCharm所用的`movie`环境解释器并设置：

```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\XUSITIAN\.conda\envs\movie\python.exe scripts\run_stage4.py
C:\Users\XUSITIAN\.conda\envs\movie\python.exe scripts\verify_stage4.py
```

上面的绝对路径是当前电脑的示例；项目代码本身不依赖固定盘符。在PyCharm中打开“设置 → 项目 → Python解释器”，选择已有Conda环境中的`python.exe`，再直接运行`scripts/run_stage4.py`。

### 9.3 主要输出与加载

```text
configs/stage4_mlp.json                    MLP与训练配置
data/processed/stage4_split_manifest.json 内部划分和固定文件哈希
models/stage4/mlp_validation_best.pt      验证阶段最佳权重
models/stage4/mlp_final.pt                完整训练集最终权重
models/stage4/*_preprocessor.joblib       类别映射、填补值和标准化器
models/stage4/model_metadata.json         重建模型所需结构和词表大小
reports/stage4/training_history.csv       逐epoch训练历史
reports/stage4/test_predictions.csv       固定测试集逐行预测
reports/stage4/cold_start_metrics.csv      冷启动分组指标
reports/stage4/training_curves.html        可独立打开的Plotly训练曲线
reports/stage4/stage4_summary.md           论文阶段素材
```

后续网页可调用`movie_rating.neural.load_mlp_artifacts`加载三个最终产物，再调用`predict_ratings`完成单条或批量预测。模型产物默认被`.gitignore`忽略；部署Streamlit Cloud前需要选择Git可用的模型发布方式或调整忽略规则，不能遗漏运行必需的模型文件。

### 9.4 常见问题

| 问题 | 解决方法 |
|---|---|
| CUDA不可用 | 正常使用CPU，本阶段不要求CUDA |
| Windows训练卡住 | 保持`num_workers=0`并从带`__main__`入口的脚本运行 |
| Embedding索引越界 | 确保未知索引为0，词表大小包含未知项，并加载配套预处理器 |
| 损失出现NaN | 检查缺失值、标准化、学习率和输入张量，训练脚本已启用梯度裁剪 |
| 模型重载结果变化 | 调用`eval()`并使用与权重配套的最终预处理器和元数据 |
| 指标不如随机森林 | 如实保留结果；不得查看测试集调参，正式对比留到第5阶段 |
| 固定划分验证失败 | 不要修改`train_ratings.csv`或`test_ratings.csv`，先运行第2阶段验证 |

## 10. 第5阶段：集成学习与神经网络对比

第5阶段不重新训练模型，而是复用第2至4阶段固定产物，在相同的20000条测试记录上统一比较全局均值、随机森林、AdaBoost、XGBoost和MLP。所有预测通过`interaction_id`一对一合并，禁止按行位置直接拼接。

### 10.1 比较内容与公平性

- 统一复算RMSE、MAE、R²、相对基线改善率和排名；
- 使用2000次成对Bootstrap比较MLP与三种集成模型，差异定义为MLP指标减集成模型指标；
- 按真实评分、冷启动、训练集用户活跃度和训练集电影流行度分析误差；
- 树模型使用原生分裂重要性，MLP使用8000条内部验证集上的分组排列重要性；
- 在同一CPU上进行1次预热和5次正式加载/端到端推理计时。

树模型重要性与MLP排列重要性的定义不同，只能比较各模型内部排名和信息类型，不能直接比较绝对数值，也不能解释为因果影响。固定测试集只用于最终对比和工程建议，不用于重新训练或调参。

### 10.2 运行和验证

```powershell
conda activate movie
python scripts/run_stage5.py
python scripts/verify_stage5.py
python -m pip check
```

如果PowerShell的`conda run`出现中文编码问题，先执行：

```powershell
$env:PYTHONIOENCODING="utf-8"
```

然后直接调用PyCharm当前选择的`movie`环境解释器。

### 10.3 输出文件

```text
configs/stage5_comparison.json                 Bootstrap、排列和计时配置
reports/stage5/unified_test_predictions.csv   五模型统一预测表
reports/stage5/metrics_comparison.*           测试指标和排名
reports/stage5/paired_bootstrap_summary.*     成对Bootstrap区间
reports/stage5/segment_metrics.csv            分组误差
reports/stage5/residual_summary.csv           残差统计
reports/stage5/*importance*.csv               两类特征重要性
reports/stage5/efficiency_benchmark.csv        本机CPU效率
reports/stage5/model_recommendation.json       第6阶段工程建议
reports/stage5/*.html                         五个Plotly交互图
reports/stage5/paper_results_material.md       论文结果素材
reports/stage5/stage5_summary.md               阶段总结
```

第6阶段应读取`model_recommendation.json`确定默认模型，同时保留全局均值、随机森林、XGBoost和MLP供模型实验室展示。该推荐已经使用固定测试集结果，后续不能继续使用同一测试集调参。

### 10.4 常见问题

| 问题 | 解决方法 |
|---|---|
| 不同阶段预测顺序不同 | 必须用`interaction_id`一对一合并并恢复固定测试集顺序 |
| 指标有极小差异 | 检查CSV浮点序列化；约`1e-8`误差可接受 |
| MLP验证权重无法加载 | 使用验证版预处理器的实际词表大小重建模型 |
| 排列重要性出现负值 | 保留原始值，仅在正归一化时截断为0 |
| 空分组R²报错 | 样本不足或真实评分无方差时保存为空并说明 |
| Bootstrap较慢 | 使用NumPy分批或循环，但不得少于2000次 |
| 效率结果波动 | 先预热并报告5次正式计时的均值和标准差 |
| 两类重要性看似矛盾 | 方法定义和评估数据不同，不比较绝对值 |

## 11. 第6阶段：Streamlit多页面电影评分与推荐系统

应用入口为`app.py`，使用`st.Page`与`st.navigation`注册首页、电影库、评分预测、个性化推荐、我的观影和模型分析。侧边栏通过统一用户编号切换预测与推荐结果。

### 11.1 主要功能

- 首页：统一电影规模、模型指标、系统状态、中文搜索和混合推荐预览；
- 电影库：合并300部中国电影和1,324部具有中文译名的其他国家电影，统一按片名、电影类型和年份检索、筛选与评分；
- 评分预测：自动按影片来源选择对应模型，显示中文模型名称、多模型评分和预测依据；
- 个性化推荐：中国电影与其他国家电影混合排序，并为每部电影提供至少两条理由；
- 我的观影：区分固定训练期历史与网页SQLite记录，支持编辑、二次确认删除和内存CSV下载；
- 模型分析：展示模型指标、运行效率、自助法检验、特征重要性、分段误差和即时多模型预测。

推荐只用固定训练集计算用户画像和电影热度，不读取固定测试集真实评分：

```text
综合分 = 0.85 × MLP预测 + 0.10 × 类型偏好 + 0.05 × 贝叶斯平滑口碑
```

候选集会排除训练期已评价、网页已看、不感兴趣和点踩的电影。推荐理由来自实际MLP分数、匹配的正偏好类型和训练期热度，历史较少时会明确说明降级依据。

### 11.2 初始化、验证和启动

```powershell
conda activate movie
python scripts/init_app_db.py
python -m unittest discover -s tests -p "test_stage6*.py"
python scripts/smoke_test_streamlit.py
python scripts/verify_stage6.py
python -m streamlit run app.py
```

网页默认地址为`http://localhost:8501`。如果端口被占用，可执行：

```powershell
python -m streamlit run app.py --server.port 8502
```

静态数据和报告使用`st.cache_data`，模型使用`st.cache_resource`。集成模型只有在多模型比较时才延迟加载。SQLite连接不缓存，每次操作使用短连接和参数化SQL。

本地反馈保存在`data/app/movie_app.db`，该文件已忽略提交，可删除后重新运行`python scripts/init_app_db.py`生成。数据库记录包含用户/电影、想看或已看、个人评分、喜欢/中立/不喜欢/不感兴趣、备注、保存时预测分和时间。

### 11.3 常见问题

| 问题 | 解决方法 |
|---|---|
| PowerShell找不到Conda | 改用Anaconda Prompt，或运行`conda init powershell`后重开终端 |
| 模型文件缺失 | 运行`scripts/verify_stage4.py`和`scripts/verify_stage5.py`检查前置产物 |
| 模型反复加载 | 确认模型加载函数仍使用`st.cache_resource`且路径参数稳定 |
| 保存反馈后推荐未变化 | 重新生成推荐；最终推荐不做长期缓存，数据库记录会在下次运行时读取 |
| SQLite提示锁定 | 关闭重复启动的网页进程；应用已使用短连接、WAL和`busy_timeout` |
| 8501端口占用 | 使用`--server.port 8502`或先关闭旧Streamlit进程 |
| 网页预测时间特征异常 | 不得用当前时间；系统使用用户训练期最新时间戳或训练集最大时间戳 |

Streamlit Community Cloud文件系统是临时性的；在线反馈在重启、休眠或重新部署后可能丢失。
# 第7阶段：公开部署、论文与答辩材料

本阶段把第6阶段网页整理为可部署包。公开仓库**不包含** MovieLens 原始CSV、训练/测试逐行数据和SQLite反馈库；应用首次启动时从 [GroupLens官方地址](https://files.grouplens.org/datasets/movielens/ml-100k.zip) 下载ZIP，核对MD5并生成数据。这样既能复现实验，也避免未经许可重新分发MovieLens数据。

## 本地与云端启动

```powershell
conda activate movie
python -m pip install -r requirements.txt
streamlit run app.py
python scripts/verify_stage7.py
```

Streamlit Community Cloud入口为 `app.py`，公开应用为 <https://mccr78dsncofymbfxhdwfo.streamlit.app/>。根目录只保留一个 `requirements.txt`，Conda复现实验文件位于 `conda/environment.yml`。首次云端冷启动需要访问 `files.grouplens.org`；若下载失败，应查看云端日志并重试，不要使用来源不明的镜像。

本地实验严格使用Python 3.10。2026-10-01部署时Community Cloud界面只提供Python 3.11至3.14，因此线上选择3.11，并额外在干净Python 3.11环境完成依赖安装、四模型推理、推荐、六页面AppTest和健康检查。

随机森林模型约38 MB，通过Git LFS跟踪。推送前运行 `git lfs install` 和 `git lfs track models/stage3/random_forest.joblib`，并确认远程仓库实际包含LFS对象。模型和数据读取使用Streamlit缓存。

用户观影与反馈保存在 `data/app/movie_app.db`。该数据库仅用于本地/课程演示，Streamlit Cloud重新部署、休眠恢复或实例迁移时可能丢失；真实生产应用应改用带认证的外部持久数据库。

论文与最终答辩材料见 `docs/`，兼容旧目录的答辩草稿见 `defense/`，部署证据、真实云端截图与验收结果见 `reports/stage7/`。仓库地址、在线地址和已验收提交记录在 `configs/stage7_deployment.json`。

## 中文电影数据库增强

系统包含两类边界清晰的中文电影资料：

1. `data/catalog/movie_localizations.csv` 为1,324部MovieLens影片提供中文片名和常用别名。中文字段只参与搜索与显示，影片仍使用原`movie_id`，因此不会改变模型输入、训练集、测试集或论文指标。
2. `data/catalog/chinese_movies.csv` 提供48部中国电影基础目录，覆盖1934年至2024年。它们可按中英文片名、简介、地区、类型和年份检索，也可以加入想看、标记已看和记录个人评分。

应用启动时会把两张CSV幂等同步到SQLite的`movie_localizations`、`chinese_movies`和`catalog_interactions`表。统一电影库当前包含1,624部电影；300部中国电影与1,324部其他国家电影共用“动作、冒险、喜剧、剧情”等中文类型筛选，不再按地区作为主要搜索条件。独立扩展库中没有MovieLens ID的影片会明确显示“冷启动”，不会输出没有训练依据的模型预测。

需要重新生成中文片名扩展和中国电影类型时运行：

```powershell
conda activate movie
python scripts/expand_movie_catalog.py
```

脚本通过MovieLens 25M的IMDb映射关联影片，并从Wikidata取得中文片名；已有人工校对译名优先保留。它支持重复执行，使用`--force`可重新查询Wikidata。MovieLens中唯一未提供类型的《路边野餐》以可审计覆盖项补为“剧情”。

继续扩充时可直接追加`chinese_movies.csv`，要求`catalog_id`唯一、年份为整数，并使用`|`分隔多个地区和类型。修改后运行：

```powershell
conda run -n movie python -m unittest tests.test_chinese_catalog -v
conda run -n movie python -m unittest discover -s tests -p "test_stage6*.py" -v
conda run -n movie python scripts/test_clean_deployment.py
```

基础目录的简介为课程项目原创概述，资料链接逐条保留在`source_url`；该目录用于检索和演示，不宣称完整收录全部华语电影，也不作为既有评分模型的训练数据。

### Youku-mPLUG候选线索导入

[Youku-mPLUG](https://github.com/X-PLUG/Youku-mPLUG) 是大规模中文视频—文本预训练数据集，官方类别中与电影有关的是“电影剪辑”和“电影周边（预告/杂谈）”。这些记录是视频标题和片段，不是带有上映年份、导演、类型及稳定影片ID的规范电影数据库，因此系统不会把它们直接写入正式中国电影目录，也不会把视频文件纳入仓库或Streamlit部署包。

如已按官方说明从[ModelScope数据页](https://modelscope.cn/datasets/modelscope/Youku-AliceMind/summary)取得分类标注，可只提取电影相关标题作为人工核验线索：

```powershell
conda activate movie
python scripts/import_youku_mplug_candidates.py `
  data/external/youku_mplug/classification_train.csv
python -m unittest tests.test_youku_mplug_import -v
```

脚本兼容CSV、JSONL和JSON，内置官方两个电影相关类别的编号映射；也可用`--classname`显式传入官方`classname.json`做一致性检查。输出为`data/imports/youku_mplug_movie_candidates.csv`及摘要JSON。候选表默认状态为`pending`，并预留规范中文片名、年份、地区、类型和简介字段。必须核对来源、确认记录对应电影正片并补齐事实字段，才能人工追加到`data/catalog/chinese_movies.csv`；短视频标题、预告、混剪和杂谈不得直接作为电影条目。`--force`会覆盖已有候选及人工审核内容，只应在明确需要重新生成时使用。

本项目只使用用户自行取得的标注文件，不下载或再分发Youku视频。Youku-mPLUG仓库代码采用Apache-2.0许可证，但视频、文本和具体数据使用仍应遵守官方数据页条款及内容权利要求；代码许可证不能替代对数据内容的授权。

## 中国电影真实评分数据库与独立训练

Youku-mPLUG不包含规范电影评分，因此评分扩展实验改用可复现的真实数据链路：MovieLens 25M提供匿名用户评分和IMDb映射，Wikidata提供CC0的出品地区、原始语言、中英文片名与上映年份。影片必须同时满足“出品地区属于中国大陆、香港、台湾、澳门或民国时期”以及“原始语言属于中文、官话、粤语、吴语或闽南语”等条件，避免把仅在中国参与制作的英语片误判为中国电影。

构建和训练命令：

```powershell
conda activate movie
python scripts/build_chinese_ratings_dataset.py
python scripts/train_chinese_models.py
python scripts/verify_chinese_training.py
```

构建脚本从GroupLens官方地址下载`ml-25m.zip`并核对MD5；网络不可用时可手动下载后使用：

```powershell
python scripts/build_chinese_ratings_dataset.py --archive D:\Downloads\ml-25m.zip
```

迭代过滤要求每位用户至少评价5部保留影片、每部影片至少有10条保留评分。实际生成55,483条评分、5,973名用户和300部中国电影。对每位用户按时间排序，最后一次评分作为测试集、倒数第二次作为验证集，其余为训练集；测试集固定为5,973条。

| 模型 | 测试RMSE | 测试MAE | R² |
|---|---:|---:|---:|
| XGBoost | 0.785562 | 0.581466 | 0.316829 |
| 随机森林 | 0.789642 | 0.583603 | 0.309714 |
| PyTorch MLP | 0.794466 | 0.596052 | 0.301255 |
| 全局均值 | 0.954616 | 0.732787 | -0.008847 |

训练模型位于`models/chinese/`，实验报告位于`reports/chinese_training/`。网页“电影库”将300部中国电影与1,324部具有中文译名的其他国家电影合并展示，所有影片按“冒险、喜剧”等题材统一筛选，并按影片来源自动选择评分模型；“个性化推荐”会在所选题材内混排两类影片并输出中文推荐理由。模型使用`st.cache_resource`加载。

MovieLens 25M原始包、评分子集和原始用户/影片ID均被`.gitignore`排除，不在公开仓库中再分发。公开的`data/catalog/chinese_rated_movies.csv`只包含Wikidata CC0字段以及本项目生成的内部模型索引。MovieLens 25M仅用于非商业课程研究，使用者须阅读GroupLens官方README、遵守不得擅自再分发等许可条件并在论文中致谢。
