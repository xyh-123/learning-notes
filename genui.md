后端
---

依据Django框架搭建

### 安装

Linux系统下运行，Windows系统某些包不支持

```
git clone --branch dev/master git@github.com:martin-sicho/genui.git
conda env create -f environment.yml

#激活环境
source activate genui（上一步新建的环境名）
#安装其他依赖
pip install -r requirements.txt

```

### 启动应用服务器

debug-启用调试的配置。用于开发中的本地部署。

staging-配置用于部署在远程服务器上，但是必须调试成功。

production-在生产中部署

通过`genui.settings.base` 和 `genui.settings.genuibase` 设置模块可以创建自己的配置模块。新模块可以从这两个模块继承。查看 `genui.settings.debug` 的源代码，可以了解如何使用这些基本模块配置新项目。

```python
#激活环境
conda activate genui 

#进入src目录
cd src/

#使用debug模式
export DJANGO_SETTINGS_MODULE=genui.settings.debug configuration

#初始化数据库
python manage.py migrate 

#安装 genui 扩展模块
python manage.py genuisetup 

#运行开发服务器
python manage.py runserver # run the development server
```

第一次启动服务器，需要运行 `migrate` 和 `genuisetup` 命令。`genuisetup` 命令检查当前安装的 GenUI 扩展并运行它们的设置方法。每次安装或更新扩展时都应该运行此命令。

如果一切顺利，现在应该可以从输出中显示的端口从 localhost 访问后端应用程序。如果服务器在端口8000上运行，你可以通过 http://localhost:8000/API/服务器来验证这一点，它将显示后端 REST API 文档。

依赖于 PostgreSQL 数据库进行数据存储。配置文件 `genui.settings.debug` 假定数据库服务器对本地主机上的应用程序可用: 5432 端口。

对于其他配置，您应该使用 **POSTGRES_DB**、**POSTGRES_USER**、 **POSTGRES_PASSWORD** 和 **POSTGRES_HOST** 环境变量来告诉应用程序要查找哪个数据库以及要使用哪个凭据。查看 `genui.settings.stage` 和 `genui.settings.prod` 的源文件了解详情。

一切顺利应该能看到下面这个界面

![img](assess/J_CSGSZ0SA1BR21%7B%7BS49H1O.png)