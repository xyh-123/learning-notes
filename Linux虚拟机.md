虚拟机右上角网络图标没了且连不上网？解决办法
---

**重新找回右上角网络图标**

```
sudo service network-manager stop
sudo rm /var/lib/NetworkManager/NetworkManager.state
sudo service network-manager start

```

重新配置

1.编辑

/etc/NetworkManager/NetworkManager.conf  将其中的managed=false改为managed=true

```
sudo vi  /etc/NetworkManager/NetworkManager.conf
```

2.重启

```
sudo service network-manager restart
```

安装nginx
---

```
# 切换至root用户
sudo su root
apt-get install nginx
```

![在这里插入图片描述](assess/20181031202757348.gif)

```
nginx -v
```

![image-20220304150636670](assess/image-20220304150636670.png)

启动nginx

```
service nginx start
```

常用命令
---

```
#清屏命令
clear
```

解压tar.gz文件

```
 #解压到当前
 tar  -zxvf  fenci.py.tar.gz 
 #解压到指定文件夹
 tar  -zxvf  fenci.py.tar.gz  -C  pythontab/
```

tar文件

```
#解压
tar xvf FileName.tar
#压缩
tar cvf FileName.tarsu
```

解压.gz文件

```
gunzip -d pythontab.gz
```

Anaconda使用
---

```
 #出现图形化界面
 anaconda-navigator
```

### 解决进入环境后，命令行前面没有`(base)`标识的问题

在最后一行添加以下内容，其中的路径根据自己的安装路径填写

anaconda3的路径`/home/xyh/anaconda3`

```
sudo gedit ~/.bashrc
export PATH="//home/xyh/anaconda3/bin:$PATH"
```

```
sudo gedit /etc/profile
export PATH="//home/xyh/anaconda3/bin:$PATH"
```

```
source /etc/profile 
```

### 为安装的Pycharm建立快捷方式，创建文件

```
cd /usr/share/applications
sudo gedit pycharm.desktop
```

编辑这个文件，添加以下内容，根据自己pycharm的路径

```
[Desktop Entry]
Version=1.0
Type=Application
Name=Pycharm
Icon=//home/xyh/software/pycharm/bin/pycharm.png
Exec=sh /home/xyh/software/pycharm/bin/pycharm.sh
MimeType=application/x-py;
Name[en_US]=pycharm
```



requirements.txt中的git源安装问题
---

![image-20220305094402717](assess/image-20220305094402717.png)git源是下载到anaconda3/envs/环境名/lib/python3.7/site-packages中

![img](assess/PU0I%60ES@Y4%7BX_$V727JYT%5DN.png)

如果安装失败，可以去github上下载指定的文件夹然后放到该目录

![image-20220305094828985](assess/image-20220305094828985.png)