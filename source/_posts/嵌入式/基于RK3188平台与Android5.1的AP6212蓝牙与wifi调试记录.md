---
title: 基于RK3188平台与Android5.1的AP6212蓝牙与wifi调试记录
date: 2017-08-30 11:21:46
tags:
	[RK3188,Wifi,蓝牙,Android5.1]
categories:
	- 嵌入式
copyright:
	yitalkabe
---

&emsp;&emsp;最近调试了一款基于RK3188与Android5.1的板子，蓝牙和wifi的芯片采用的是AP6212，AP6212文档在[**这里**](相关文档\AP6212_V1.1_09022014.pdf)。存在的问题是Wifi和蓝牙都不能扫描设备。
<!--more-->
### 1. AP6212管脚图

<div align=center>
	![AP6212管脚.jpg](AP6212管脚.jpg)
</div>

&emsp;&emsp;管脚10和管脚11是26M晶振的IN和OUT，要求26M晶振的频偏是10ppm。24管脚是32.768KHz晶振的输入脚，如果管脚22（VDD）的幅度为1.8v，则要求24管脚的幅度是1.7v-1.8v，如果管脚22（VDD）的幅度为3.3v，则要求24管脚的幅度是3v-3.3v。通过调整管脚22之前的分压电阻来调整幅度满足要求。通过以上操作可以使得wifi搜的信号，并能连接网络。如果wifi不能工作尝试换一下其他26M晶振。

### 2. 更新蓝牙固件
&emsp;&emsp;经过如上操作，可以使得wifi正常工作，并且蓝牙也能扫描出附近的蓝牙设备，并且能和蓝牙2.0配对，但是不能连接BLE。问题解决办法：联系AP6212的供应商，提供新的AP6212蓝牙固件。对Android5.1系统进行蓝牙固件更新。
#### 2.1 固件更新方法
1. 安装adb调试工具，配置adb.exe的环境变量
2. 将Android5设备与电脑连接
3. 通过`adb root`获得root权限，执行`adb remount`
4. push更新固件`adb push bcm43438a0.hcd /vendor/firmware/bcm43438a0.hcd`
5. 重新启动安卓设备，测试蓝牙连接
**固件位置要查看android源码进行查找，可以通过`locate bt_vendor.conf`对配置文件进行定位，比如`FwPatchFilePath = /vendor/firmware/`**