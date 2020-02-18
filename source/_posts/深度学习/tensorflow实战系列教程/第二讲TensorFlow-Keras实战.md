---
title: Tensorflow系列教程2-Keras实战
categories: [TensorFlow系列教程]
tags: [Tensorflow]
typora-root-url: 第二讲TensorFlow-Keras实战
typora-copy-images-to: 第二讲TensorFlow-Keras实战
---

​    

<!--more-->

# Tensorflow系列教程2-Keras实战

##  理论部分

### keras简介

- keras的自身特点
  - 基于python的高级神经网络的API，不是一个完整的库
  - Francois Chollet于2014-2015年编写Keras
  - 以Tensorflow、CNTK或者Theano为后端运行，keras必须有后端才能运行
    - 后端可以切换，现在多用于tensorflow
  - 极方便于快速实验，帮助用户以最少的时间验证自己的想法

- Tensorflow-keras简介
  - Tensorflow对keras API规范的实现
  - 相对于以tensorflow为后端的keras，TensorFlow-keras于Tensorflow结合更加紧密
  - 实现在tf.keras空间下

- Tf-keras与Keras联系
  - 基于同一套API
    - keras程序可以通过导入方式轻松转为tf.keras程序
    - 反之可能不成立，因此tf.keras有其他特性
  - 相同的Json和HDF5模型序列化格式和语义
- Tf-keras与Keras区别
  - tf.keras全面支持eager mode
    - 只是使用keras.Sequential和keras.Model时没有影响
    - 自定义Model内部运算逻辑时候会有影响
      - tf底层API可以使用keras的model.fit等抽象
      - 适用于研究人员
    - tf.keras支持基于tf.data的模型训练
    - tf.keras支持TPU训练
    - tf.keras支持tf.distribution中的分布式策略
    - 其他特性
      - tf.keras可以与tensorflow中的estimator集成
      - tf.keras可以保存为SavedModel
  - 如何选择
    - 如果想使用tf.keras的任何一特性，那么选tf.keras
    - 如果后端互换性很重要，那么选择keras
    - 如果都不重要，随便选择即可

### 分类问题，回归问题、目标函数

- 分类问题预测的是类别，模型输出的是概率分布

  - 三分类问题的输出例子：[0.2,0.7,0.1]

- 回归问题预测的是值，模型的输出是一个实数值

- 为什么需要目标函数

  - 参数是逐步调整的
  - 目标函数真可以衡量模型的好坏

- 分类问题的目标函数

  - 需要衡量目标类别和当前预测的差距

    - 三分类问题的输出例子：[0.2,0.7,0.1]
    - 三分类真是类别：2->one_hot->[0,0,1]
    - one-hot编码，把正整数变成向量的表达
      - 生成一个长度不小于正整数的向量，只有正整数的位置处为1，其余部分为0

  - 平方差损失

    $\frac{1}{n}\sum\frac{1}{2}(y-Model(x))^2$

  - 交叉熵损失

    $\frac{1}{n}\sum y\ln(Model(x))$

- 回归问题的目标函数

  - 预测值与真实值的差距

  - 平方差损失

    $\frac{1}{n}\sum\frac{1}{2}(y-Model(x))^2$

  - 绝对值损失

    $\frac{1}{n}\sum\|y-Model(x)\|$

### 激活函数、批归一化、Dropout

- 激活函数

![image-20200214134919467](image-20200214134919467.png)

- 归一化

  - z-score归一化

    就是将样本数据变成均值为0方差为1的数据形式

  - min-max归一化

    (元素-最小值)/(最大值-最小值)

- 批归一化

  - 在每一层的激活值上都做归一化，使得网络的训练效果更好

- 归一化为什么会有效

  - 如下图，$\theta1$与$\theta2$的数据范围是不一样的，数据等高线(目标函数值)看起来像是椭圆，椭圆求法向量时候不是指向圆心，导致训练的规矩比较曲折。

  ![image-20200214135800129](image-20200214135800129.png)

- dropout

  在训练集上效果好，在测试集上效果不好。是因为模型参数过多，导致模型过拟合，记住了样本，泛化能力不强。

  ![image-20200214135927142](image-20200214135927142.png)



### Wide & Deep模型

- 简介
  - 2016年发布，用于分类问题和回归问题
  - 应用到了Google Play的应用推荐上

- 特征

  - 稀疏特征

    - 离散值特征

      - 一个x离散值特征用one-hot编码就是稀疏特征

      - 举例：专业={计算机，数学，其他}

        ​			词表={人工智能，你，我，他，....}

    - 叉乘

      - 两个离散值特征的组合

        {(计算机，你)，(计算机，我)，(计算机,人工智能)}

      - 叉乘之后
        - 稀疏特征做叉乘获取共现信息
        - 实现记忆效果

    - 稀疏特征的优缺点

      - 优点	
        - 有效，广泛应用于工业界
      - 缺点
        - 需要人工设计
        - 可能过拟合，所有特征都叉乘，相当于记住每一个样本
        - 泛化能力差，没出现就不会起效果

  - 密集特征

    - 向量表达
      - 词表={人工智能，你，我，他}
      - 他=[0.3.0.2.0.6,n维向量]
    - Word2Vec工具
      - （男-女）距离等于 (国王-王后)的距离
    - 密集特征的优缺点
      - 优点
        - 带有语义信息，不同向量之间有相关性
        - 兼容没有出现过的特征组合
        - 更少人工参与
      - 缺点：
        - 过度泛化，推荐不怎么相关的产品

  - 模型结构

    ![image-20200215192712627](image-20200215192712627.png)

    ![image-20200215192907210](image-20200215192907210.png)

    ![image-20200215193415767](image-20200215193415767.png)

### 超参数搜索

- 什么是超参数

  - 在神经网络有很多训练过程中不变的参数
    - 网络结构参数：几层，每层宽度，每层激活函数
    - 训练参数：batch_size，学习率，学习衰减算法等

- 为什么要搜索超参数

  - 人工去调试耗费人力

- 搜索策略

  - 网格搜索

    - 最优参数不在搜索空间中

    ![image-20200215213416470](image-20200215213416470.png)

  - 随机搜索

    - 搜索空间比网格搜索大很多

    ![image-20200215213506344](image-20200215213506344.png)

  - 遗传算法搜索

    ![image-20200215213654324](image-20200215213654324.png)

  - 启发式搜索

    ![image-20200215213930433](image-20200215213930433.png)

## 实战部分

### Keras搭建分类模型

- 读取数据

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os, sys, time
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf.keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
# 数据归一化
# x_valid = x_valid.astype('float32') / 255
# x_train = x_train.astype('float32') / 255
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_valid = scaler.fit_transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_train = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


def show_single_image(ima_arr):
    plt.imshow(ima_arr, cmap="binary")
    plt.show()


def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)

    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(x_data[index], cmap="binary",
                       interpolation="nearest")
            plt.axis("off")
            plt.title(class_names[y_data[index]])
    plt.show()


class_names = ['T-shrit', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot']
show_single_image(x_train[0])
show_imgs(3, 5, x_train, y_train, class_names)
```

![image-20200213133528926](image-20200213133528926.png)

![image-20200213134458326](image-20200213134458326.png)

```python
import tensorflow.keras as keras

# tf.keras.models.Sequential()
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# model输出不是one_hot编码时候使用sparse_categorical_crossentropy
# model输出是one_hot编码时候使用categorical_crossentropy
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
print(model.layers)
print(model.summary())

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))

```

不同的归一化方式结果对比（不同的优化器optimizer也可能对结果是有影响的，比如笔者实验室时候使用adam优化器准确率可以达到91%）

```bash
# x_valid = x_valid.astype('float32') / 255
# x_train = x_train.astype('float32') / 255
accuracy: 0.8798 - val_loss: 0.3495 - val_accuracy: 0.8732

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_valid = scaler.fit_transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
# x_train = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
accuracy: 0.9115 - val_loss: 0.3059 - val_accuracy: 0.8880
```

```python
def plot_learning_curves(_history):
    pd.DataFrame(_history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)
```

![image-20200214103719112](image-20200214103719112.png)

```python
# 进行测试集测试
model.evaluate(x_test, y_test)
```

```python
 loss: 0.1844 - accuracy: 0.8869
```

### Keras回调函数

```python
# Tensorboard，earlystopping，ModelCheckPoint
logdir = 'callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    keras.callbacks.EarlyStopping(min_delta=1e-3,
                                  patience=5)
]
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_valid, y_valid),
    callbacks=callbacks
)
```

- 可以运行tensorboard

```
tensorboard --logdir=callbacks
```

### Keras搭建回归模型

```python
import os
import sys
from typing import List, Sized

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# 获取数据
housing: sklearn.utils.Bunch = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)
import pprint

pprint.pprint(housing.data[0:5])
pprint.pprint(housing.target[0:5])

# 默认3:1划分
x_train, x_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=7, test_size=0.25
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, random_state=1
)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_valid.shape, y_valid.shape)

# 归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_valid_scaler = scaler.transform(x_valid)
x_test_scaler = scaler.transform(x_test)

# 搭建模型
model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]))
model.add(keras.layers.Dense(1))
model.summary()
model.compile(loss="mean_squared_error",
              optimizer="sgd")
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=5, min_delta=1e-3
    )
]
history = model.fit(x_train_scaler, y_train,
                    validation_data=(x_valid_scaler, y_valid),
                    epochs=100,
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

# 测试
test_res = model.evaluate(x_test_scaler, y_test)
print(test_res)
```

![image-20200215203640734](image-20200215203640734.png)

### Keras搭建深度神经网络

```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
```

![image-20200215164314893](image-20200215164314893.png)

- 前三次迭代没有明显的变化原因：
  - 参数众多，训练不充分
  - 梯度消失

- 添加批归一化

  ```python
  for _ in range(20):
      model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.BatchNormalization())
  ```
  
- 添加dropout

  ```python
  for _ in range(20):
      model.add(keras.layers.Dense(100, activation="selu"))
      model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.AlphaDropout(0.5))
  model.add(keras.layers.Dense(10, activation="softmax"))
  ```

  - AlphaDropout
    - 均值和方差不变
    - 激活值的归一化性质不变
    - 一般使用AlphaDropout

### Keras实现wide&deep模型

> 使用：
>
> ​	子类API
>
> ​	函数式API
>
> ​	多输入与多输出

- 对回归模型的代码通过子类API进行修改

```python
class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        """定义模型的层次"""
        self.hidden1_layer = keras.layers.Dense(30, activation="relu")
        self.hidden2_layer = keras.layers.Dense(30, activation="relu")
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """完成模型的正向计算"""
        hidden1 = self.hidden1_layer(inputs)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([inputs, hidden2])
        output = self.output_layer(concat)
        return output


model = WideDeepModel()
model.build(input_shape=(None, 8))
```

![image-20200215205717102](image-20200215205717102.png)

- 对回归模型的代码通过函数式进行修改

```python
# 使用函数式API
input = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)

concat = keras.layers.concatenate([input, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input], outputs=output)
print(model.summary())
```

![image-20200215204509528](image-20200215204509528.png)

- 多输入方式

  ```python
  # 多输入
  input_wide = keras.layers.Input(shape=[5])
  input_deep = keras.layers.Input(shape=[6])
  hidden1 = keras.layers.Dense(30, activation="relu")(input_deep)
  hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
  concat = keras.layers.concatenate([input_wide, hidden2])
  output = keras.layers.Dense(1)(concat)
  model = keras.models.Model(inputs=[input_wide, input_deep],
                             outputs=[output])
  x_train_scaler_wide = x_train_scaler[:, : 5]
  x_train_scaler_deep = x_train_scaler[:, 2:]
  x_valid_scaler_wide = x_valid_scaler[:, : 5]
  x_valid_scaler_deep = x_valid_scaler[:, 2:]
  x_test_scaler_wide = x_test_scaler[:, : 5]
  x_test_scaler_deep = x_test_scaler[:, 2:]
  
  model.compile(loss="mean_squared_error",
                optimizer="sgd")
  callbacks = [
      keras.callbacks.EarlyStopping(
          patience=5, min_delta=1e-3
      )
  ]
  history = model.fit([x_train_scaler_wide, x_train_scaler_deep], y_train,
                      validation_data=([x_valid_scaler_wide, x_valid_scaler_deep], 					  y_valid),
                      epochs=100,
                      # callbacks=callbacks
                      )
  ```

  ![image-20200215211552014](image-20200215211552014.png)

- 多输出模式

  ```python
  input_wide = keras.layers.Input(shape=[5])
  input_deep = keras.layers.Input(shape=[6])
  hidden1 = keras.layers.Dense(30, activation="relu")(input_deep)
  hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
  concat = keras.layers.concatenate([input_wide, hidden2])
  output = keras.layers.Dense(1)(concat)
  output2 = keras.layers.Dense(1)(hidden2)
  
  model = keras.models.Model(inputs=[input_wide, input_deep],
                             outputs=[output, output2])
  
  ```

  ![image-20200215212622330](image-20200215212622330.png)

### Keras与scikit-learn实现超参数搜索

- 使用scikit实现超参数搜索

  ```python
  import os
  import sys
  from typing import List, Sized
  
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import tensorflow.keras as keras
  from sklearn.preprocessing import StandardScaler
  import tensorflow as tf
  import sklearn
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import fetch_california_housing
  
  # 获取数据
  housing: sklearn.utils.Bunch = fetch_california_housing()
  
  
  # 默认3:1划分
  x_train, x_test, y_train, y_test = train_test_split(
      housing.data, housing.target, random_state=7, test_size=0.25
  )
  x_train, x_valid, y_train, y_valid = train_test_split(
      x_train, y_train, random_state=1
  )
  
  # 归一化
  from sklearn.preprocessing import StandardScaler
  
  scaler = StandardScaler()
  x_train_scaler = scaler.fit_transform(x_train)
  x_valid_scaler = scaler.transform(x_valid)
  x_test_scaler = scaler.transform(x_test)
  
  
  # 随机参数搜索
  # 1 转化为sklearn的model
  # 2 定义参数集合
  # 3 搜索参数
  
  # 定义模型
  def build_model(hidden_layers=1,
                  layer_size=30,
                  lr=3e-3):
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(layer_size, activation="relu", input_shape=x_train.shape[1:]))
      for _ in range(hidden_layers - 1):
          model.add(keras.layers.Dense(layer_size, activation="relu"))
      model.add(keras.layers.Dense(1))
      optimizer = keras.optimizers.SGD(lr)
      model.compile(loss="mse", optimizer=optimizer)
      return model
  
  
  callbacks = [
      keras.callbacks.EarlyStopping(
          patience=5, min_delta=1e-3
      )
  ]
  # 将模型转化为sklearn形式
  sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)
  # history = sklearn_model.fit(x_train_scaler,
  #                             y_train,
  #                             epochs=20,
  #                             validation_data=(x_valid_scaler, y_valid),
  #                             callbacks=callbacks)
  
  
  # 定义搜索空间参数集合
  from scipy.stats import reciprocal
  
  param_distribution = {
      "hidden_layers": [1, 2, 3, 4],
      "layer_size": np.arange(1, 100),
      "lr": reciprocal(1e-4, 1e-2)
  }
  
  # 搜索参数
  from sklearn.model_selection import RandomizedSearchCV
  
  random_search_cv = RandomizedSearchCV(sklearn_model,
                                        param_distribution,
                                        n_iter=10, n_jobs=1)
  random_search_cv.fit(x_train_scaler,
                       y_train,
                       epochs=20,
                       validation_data=(x_valid_scaler, y_valid),
                       callbacks=callbacks)
  print(random_search_cv.best_params_)
  print(random_search_cv.best_score_)
  print(random_search_cv.best_estimator_)
  model = random_search_cv.best_estimator_.model
  test_res = model.evaluate(x_test_scaler, y_test)
  print(test_res)
  ```

  ```bash
  {'hidden_layers': 4}
  -0.6898686443393751
  <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x00000234F1D52508>
  ```

  



