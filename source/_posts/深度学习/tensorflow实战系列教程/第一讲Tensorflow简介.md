---
title: Tensorflow系列教程1-简介
categories: [TensorFlow系列教程]
tags: [Tensorflow]
typora-root-url: 第一讲Tensorflow简介
typora-copy-images-to: 第一讲Tensorflow简介
---

​    

<!--more-->

# Tensorflow系列教程1-简介

## Tensorflow是什么

Tensorflow是Google的一个开源软件库。
- 采用数据流图，用于数值计算
    - 节点用于处理数据
    - 线表示节点间的输入输出关系
    - 线上运输张量
    - 节点被分配到各种计算设备上运行
- 支持多种平台——GPU、CPU、移动设备
- 最初用于深度学习，变得越来越通用、

Tensorflow的特点
- 高度的灵活性
- 真正的可移植性
- 产品和科研结合
- 自动求微分
- 多语言支持
- 性能最优化


## Tensorflow版本变迁

- 2015年11月
> TensorFlow宣布开源并发布首次版本
- 2015年12月
> 支持CPU，支持Python3.3（v0.6)
- 2016年4月
> 支持分布式(v0.8)
- 2016年11月
> 支持Windows(v0.11)
- 2017年2月
> 性能改进，API稳定性(v1.0)4
- 2017年4月
> Keras集成(v1.1)
- 2017年8月
> 高级API,预算估计器，更多模型，初始TPU支持(v1.3)
- 2017年11月
> Eager execution和Tensorflow Lite(v1.5)
- 2018年3月
> 推出TF Hub,TensorFlow.js,TF扩展库TensorFlow Extend(TFX)
- 2018年5月
> 新入门内容，Cloud TPU模块与管道(v1.6)
- 2018年6月
> 新的分布式策略API，概率编程工具TensorFlow Probability(v1.8)
- 2019年8月
> Cloud Big Table集成(v1.10)
- 2019年10月
> 侧重于可用性的API改进(v1.12)
- 2019年10月
> Tensorflow v2.0发布

### Tensorflow1.0-主要特性

- XLA-Acclerate Linear Algebra
    - 专门针对线性计算的优化器，可以使得TF计算的更快。
    - 提升训练速度58倍
    - 具有可移植性，可以在移动设备上运行
- 引入更高级别的API-tf.layers/tf.metrics/tf.losses/tf.keras
- TensorFlow调试器
- 支持Docker镜像，引入TensorFlow serving服务
- TensorFlow1.0的架构

![tfv1.0架构图.jpg](tfv1.0架构图.jpg)

### TensorFlow2.0 -主要特性

- 使用tf.keras和eager mode进行更加简单的模型构建，tf也主要推广这两个feature

- 鲁棒的跨平台模型部署

  - Tensorflow服务
    - 直接通过HTTP/REST或GRPC/协议缓冲区
    - TensorFlow Lite-可以部署在Android、IOS和嵌入式设备上
    - TensorFlow.js-在javascript中部署模型
    - 其他语言的部署

- 强大的研究实验，使得研究实验更加方便快捷

  - Keras功能API和子类API，运行创建复杂的拓扑结构
  - 自定义的训练逻辑，使用tf.GradientTape和tf.custom_gradien进行更细粒度的控制
  - 底层API自始至终可以与高层结合使用，完全的可定制，使模型更加灵活
  - 高级扩展：Ragged Tensors、Tensor2Tensor等

- 清楚不推荐使用的API和减少重复来简化API

- TensorFlow2.0的架构

  - 相比1.0版本，2.0版本去掉了layers，目的是使用户使用更高层的api
  - 相比1.0版本，2.0版本添加了deployment模块。

  ![image-20200212182157130](image-20200212182157130.png)

- tf2.0开发流程
  - 使用tf.data加载数据
  - 使用tf.keras构建模型，也可以用premade estimator来验证模型
    - 使用tensorflow hub进行迁移学习
  - 使用eager mode进行运行和测试
  - 使用分发策略进行分布式训练
  - 导出到SavedModel
  - 使用TensorFlow Saver、Tensorflow Lite、Tensorflow.js 部署模型

## Tensorflow vs PyTorch

### 入门时间

- TensorFlow1.*
  - 静态图
  - 学习额外概念
    - 图、会话、变量、占位符等
  - 写样板代码
- TensorFlow2.0
  - 动态图
  - Eager mode避免1.0缺点，直接集成在python中

- Pytorch
  - 动态图
  - Numpy的扩展，直接集成在python中

### 图创建和调试

![image-20200212210157272](image-20200212210157272.png)

- Tensorflow1.x
  
  - 静态图优点
    - 效率高
  - 静态图缺点
    - 难以调试，学习tfdbg调试
  
- TensorFlow2.0和Pytorch
  
  - 动态图优点
    - 调试熔容易
    - python自带调试工具
  
- TF1.x实现一个功能的代码
  ```python
	import tensorflow as tf
  print(tf.__version__)
  
  x = tf.Variable(0.)
y = tf.Variable(0.)
  print(x)
  print(y)
  
  # x = x + y
add_op = x.assign(x + y)
  # y = y / 2
  div_op = y.assign(y / 2)
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
      for iter in range(50):
          sess.run(add_op)
          sess.run(div_op)
      print(x.eval()) # sess.eval(x)
  ```
  
- pytorch实现一个功能

  ```python
  import torch
  print(torch.__version__)

  x = torch.Tensor([0.])
  y = torch.Tensor([1.])
  for iter in range(50):
  	x = x + y 
  	y = y / 2
  print(x)
  ```

- TF2.0实现一个功能的代码

  ```python
  import tensorflow as tf
  # tf.enable_eager_execution() # 在1.x中可以调用本语句开启动态图
  print(tf.__version__)
  
  x = tf.constant(0.)
  y = tf.constant(1.)
  
  for iter in range(50):
      x = x + y
      y = y / 2
  print(x.numpy())
  ```

- 纯python

  ```python
  x = 0
  y = 0 
  for iter in range(50):
  	x = x + y
  	y = y / 2
  print(x)
  ```

### 全面性

- Pytorch缺少
  - 沿维翻转张量(np.flip,np.flipud,np.fliplr)
  - 检查无穷与非数值张量(np.is_nan,np.is_inf)
  - 快速傅里叶变换(np.fft)
- 随着时间的变化，越来越接近

### 序列化与部署

- TensorFlow支持更加广泛
  - 图保存为protocol buffer
  - 跨语言
  - 跨平台
- Pytorch支持比较简单

## Tensorflow的环境配置

### 本地配置

- Virtualenv安装
  - www.tensorflow.org/install/pip
- conda安装
  - ``conda install tenserflow-gpu``

### 云端配置

- 为什么要在云端配置
  - 规格统一，节省自己的机器
  - 有直接配置好的环境镜像
- 云环境
  - Google Cloud配置 - 送300刀免费体验
  - Amazon

