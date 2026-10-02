# 第7阶段部署与交付摘要

## 已完成

- 根目录部署入口为 `app.py`，依赖锁定在唯一 `requirements.txt`，Conda复现文件迁移至 `conda/environment.yml`。
- 云端首次启动可从GroupLens官方地址下载MovieLens 100K，核对MD5、安全解压、转换CSV并复现固定80/20划分。
- 公开版本排除原始数据、划分CSV、SQLite、密钥和逐行测试预测；随机森林使用Git LFS。
- 不含CSV和数据库的临时部署副本已完成真实官方联网冷启动、MLP推理、推荐、五页面AppTest和Streamlit健康检查。
- 全新的 `.venv-stage7` 仅按 `requirements.txt` 安装后，通过Python 3.10干净部署验证。
- 用户已确认云端使用平台当前可选的Python 3.11；独立3.11环境按同一依赖文件通过完整兼容验证。
- 论文初稿、图表索引、参考文献、答辩PPT大纲、6至8分钟讲稿、演示脚本、问答和清单均已生成。

## 已验证关键结果

- 数据：100000条评分、943名用户、1682部电影。
- 固定测试集：20000条；训练集：80000条。
- 默认MLP：RMSE 0.929017、MAE 0.733321、R² 0.318865。
- 干净部署实际预测：用户1对电影1/2/3预测约4.220526、3.054736、3.011897。
- Top-5推荐：候选1074部，每部至少两条理由。

## 公开交付

- GitHub公开仓库：<https://github.com/zeyanx/movie-rating-prediction>
- Streamlit公开应用：<https://mccr78dsncofymbfxhdwfo.streamlit.app/>
- 已在线检查首页、电影详情、个性化推荐、我的观影和模型实验室；五张真实截图保存在 `reports/stage7/screenshots/`。
- 在线写入“想看”反馈后可立即读回。Community Cloud的SQLite为临时存储，休眠、重启或重新部署后记录可能清空。
- 部署中发现Windows与Linux默认CSV换行符不同会导致固定SHA-256不一致，现已显式使用CRLF输出并在提交`4cb26a8`修复、复验通过。
