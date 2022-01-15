常用代码块
===

连接自己的云端

```
from google.colab import drive
drive.mount('/content/drive')
```



报错
===

cd: too many arguments

进入文件夹的文件名有空格，需要在文件名两端加引号

```
!cd drive/MyDrive/Colab\ Notebooks/"Molecular Generation"
```

用%cd命令强制跳到文件路径

```
%cd /content/drive/My Drive
```

 更改代码运行路径在Colab中**cd命令是无效的**，切换工作目录使用chdir函数。

```text
import os
print(os.getcwd())
os.chdir("/content/drive/My Drive/Colab Notebooks")

# path = "/content/gdrive/My Drive/sample"
# os.chdir(path)
# 返回上一级目录
os.chdir('../')
print(os.getcwd())
```

如何从github中克隆项目到colab
---

连接云端

```
from google.colab import drive
drive.mount('/content/drive')
import os
```

用%cd命令强制跳到你要clone文件的那个路径，否则默认的是根路径不好管理

```
%cd /content/drive/My Drive
```

创建文件夹

```
!mkdir filename
```

进入创建的这个文件夹

```
# %cd './filename'
os.chdir('./filename')
```

**这两种方式都可以，用!cd是不行的**

克隆项目

```
!git clone https://github.com/xxxxxx
```

Colab安装RDkit
---

