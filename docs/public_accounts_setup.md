# 公开账户与持久化推荐配置

系统使用Supabase Auth处理邮箱注册和登录，使用PostgreSQL保存个人观影数据，并通过行级安全策略（RLS）保证每个账户只能访问自己的记录。Streamlit只接收公开项目URL和匿名密钥，不保存用户明文密码，也不需要在仓库中加入`service_role`密钥。

## 1. 创建Supabase项目

1. 登录 <https://supabase.com/dashboard> 并创建项目。
2. 在SQL Editor中打开并完整执行仓库的`supabase/schema.sql`。
3. 在Authentication → Providers中启用Email。
4. 建议保留邮箱确认；在Authentication → URL Configuration中把Site URL设为公开Streamlit地址：`https://mccr78dsncofymbfxhdwfo.streamlit.app/`。
5. 在Project Settings → API Keys中复制Project URL和publishable/anon key。不得把`service_role`密钥用于本网页或提交到Git。

## 2. 配置Streamlit Cloud Secrets

打开Streamlit Cloud工作区中的应用设置，在Secrets中保存：

```toml
[supabase]
url = "https://你的项目编号.supabase.co"
anon_key = "你的publishable或anon key"
```

保存后重启应用。真实值只能放在Streamlit Secrets或本地`.streamlit/secrets.toml`，仓库仅提供`secrets.example.toml`。

## 3. 验收

1. 使用一个新邮箱注册并完成邮件确认。
2. 登录后选择至少一种偏好类型并保存。
3. 在推荐页对一部电影选择“喜欢”或“不感兴趣”。
4. 退出登录，再次登录，确认“我的观影”仍能看到同一条记录。
5. 再注册第二个账户，确认看不到第一个账户的数据。

第5项用于验证RLS隔离，不能只检查页面是否隐藏数据。

## 4. 本地运行

复制示例文件并填入同一项目的公开配置：

```powershell
Copy-Item .streamlit/secrets.example.toml .streamlit/secrets.toml
streamlit run app.py
```

如果没有配置Supabase，应用会自动进入原有本地演示模式，继续使用SQLite和MovieLens匿名用户切换，方便离线开发与自动化测试。
