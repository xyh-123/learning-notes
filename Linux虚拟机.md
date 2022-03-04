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

