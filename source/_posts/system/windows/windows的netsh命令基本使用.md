---
title: "Windows中netsh命令的基本使用"
date: 2018-08-17T20:42:04+08:00
draft: false
tags: ["Windows","命令"]
series: []
categories: ["Windows"]
img: "/thumb/6.jpg"
summary: "记录Windows的netsh命令的基本使用"
---
## 本机端口转换
>`netsh interface portproxy add v4tov4 listenport=3309 listenaddress=0.0.0.0 connectport=3389 connectaddress=127.0.0.1`

>通过`netstat -ano | findstr :3309`查看端口是否启动
## 防火墙设置
>`netsh advfirewall firewall add rule name=”forwarded_RDPport_3309” protocol=TCP dir=in localip=127.0.0.1  localport=3389 action=allow`

## 查看全部代理
>`netsh interface portproxy show all`

## dump显示代理
>`netsh interface portproxy dump`

## 删除指定代理
>`netsh interface portproxy delete v4tov4 listenport=3309 listenaddress=0.0.0.0`

## 删除全部代理
>`netsh interface portproxy reset`


