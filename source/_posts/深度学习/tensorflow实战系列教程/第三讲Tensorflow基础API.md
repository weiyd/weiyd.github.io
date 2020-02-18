---
title: Tensorflow系列教程3-Tensorflow基础API
categories: [TensorFlow系列教程]
tags: [Tensorflow]
typora-root-url: 第三讲Tensorflow基础API
typora-copy-images-to: 第三讲Tensorflow基础API
---

![](/bg.jpg)   

<!--more-->

# Tensorflow系列教程3-Tensorflow基础API

## 基础数据类型

- tf.constant

  常量操作

  ```python
  t= tf.constant([[1.,2.,3.],[4.,5.,6.]])
  print(t)
  print(t[:,1:])#  第一列之后的数据
  print(t[:,1]) # 第二列数据
  print(t[...,1]) # 第二列数据
  ```

  > 在1.0中定义的变量是不能直接被打印的，需要配合session run打印。在2.0中是可以直接打印的，是因为eager execution是默认打开的。

  ```bash
  tf.Tensor(
  [[1. 2. 3.]
   [4. 5. 6.]], shape=(2, 3), dtype=float32)
  tf.Tensor(
  [[2. 3.]
   [5. 6.]], shape=(2, 2), dtype=float32)
  tf.Tensor([2. 5.], shape=(2,), dtype=float32)
  tf.Tensor([2. 5.], shape=(2,), dtype=float32)
  ```

  算子操作

  ```python
  print(t+10)
  print(tf.square(t))
  print(t @ tf.transpose(t))
  ```

  ```bash
  tf.Tensor(
  [[11. 12. 13.]
   [14. 15. 16.]], shape=(2, 3), dtype=float32)
  tf.Tensor(
  [[ 1.  4.  9.]
   [16. 25. 36.]], shape=(2, 3), dtype=float32)
  tf.Tensor(
  [[14. 32.]
   [32. 77.]], shape=(2, 2), dtype=float32)
  ```

  与numpy的转换

  ```python
  import numpy as np
  # tf->numpy
  print(t.numpy())
  print(type(t.numpy()))
  
  # numpy->tf
  print(np.squre(t))
  np_t = np.array([[1.,2.,3.],[4.,5.,6.]])
  print(tf.constant(np_t))
  ```

  ```bash
  [[1. 2. 3.]
   [4. 5. 6.]]
  <class 'numpy.ndarray'>
  [[ 1.  4.  9.]
   [16. 25. 36.]]
  tf.Tensor(
  [[1. 2. 3.]
   [4. 5. 6.]], shape=(2, 3), dtype=float64)
  ```

  tf的0维数据

  > 在tf中称之为scalars

  ```python
  t = tf.constant(2.718)
  print(t.numpy)
  print(t.shape)
  ```

  ```bash
  2.718
  ()
  ```

- tf.string

  ```python
  t = tf.constant("cafe")
  print(t)
  print(tf.strings.length(t))
  print(tf.strings.length(t,unit="UTF8_CHAR"))
  print(tf.strings.unicode_decode(t,"UTF8"))
  
  t = tf.constant(["cafe","caffee","咖啡"])
  print(tf.strings.length(t))
  print(tf.strings.length(t,unit="UTF8_CHAR"))
  print(tf.strings.unicode_decode(t,"UTF8"))
  ```

  ```bash
  tf.Tensor(b'cafe', shape=(), dtype=string)
  tf.Tensor(4, shape=(), dtype=int32)
  tf.Tensor(4, shape=(), dtype=int32)
  tf.Tensor([ 99  97 102 101], shape=(4,), dtype=int32)
  
  tf.Tensor([4 6 6], shape=(3,), dtype=int32)
  tf.Tensor([4 6 2], shape=(3,), dtype=int32)
  <tf.RaggedTensor [[99, 97, 102, 101], [99, 97, 102, 102, 101, 101], [21654, 21857]]>
  ```

  > RaggedTensor是不完整的N维矩阵，如上每一行的数据个数是不一样的，是tf2.0新加的一个功能。

- tf.RaggedTensor

  ```python
  r = tf.ragged.constant([[11,12],[21,22,23],[],[41]])
  print(r)
  print(r[1])
  print(r[1:])
  
  # 拼接
  r2 = tf.ragged.constant([[51,52],[],[71]])
  print(tf.concat([r,r2],axis=0))
  r3 = tf.ragged.constant([[13,14],[],[71],[99]])
  print(tf.concat([r,r3],axis=1))
  
  print(r.to_tensor())
  ```

  ```bash
  <tf.RaggedTensor [[11, 12], [21, 22, 23], [], [41]]>
  tf.Tensor([21 22 23], shape=(3,), dtype=int32)
  <tf.RaggedTensor [[21, 22, 23], [], [41]]>
  <tf.RaggedTensor [[11, 12], [21, 22, 23], [], [41], [51, 52], [], [71]]>
  <tf.RaggedTensor [[11, 12, 13, 14], [21, 22, 23], [71], [41, 99]]>
  ```

  转为tensor

  ```pytnon
  print(r.to_tensor())
  ```

  ```bash
  tf.Tensor(
  [[11 12  0]
   [21 22 23]
   [ 0  0  0]
   [41  0  0]], shape=(4, 3), dtype=int32)
  ```

  > 不对其的地方用0填充

- tf.SparseTensor

  > 大部分是0，只有少数是非0。只要把非0值的大小和位置记录下来就可以。

  ```python4
  # indices是要排好序的，否则在执行to_dense()会报错
  s = tf.SparseTensor(indices = [[0,1],[1,0],[2,3]],
                     values=[1.,2.,3.],
                     dense_shape=[3,4])
  print(s)
  print(tf.sparse.to_dense(s))
  ```

  ```bash
  SparseTensor(indices=tf.Tensor(
  [[0 1]
   [1 0]
   [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
  tf.Tensor(
  [[0. 1. 0. 0.]
   [2. 0. 0. 0.]
   [0. 0. 0. 3.]], shape=(3, 4), dtype=float32)
  ```

  sparsetensor的操作

  ```python
  # 乘法操作
  s2 = s * 2
  print(s2)
  print(tf.sparse.to_dense(s2))
  
  s4 = tf.constant([[10,20],[30,40],[50,60],[70,80]],dtype=tf.float32)
  # sparsetensor与densetensor相乘得到densetensor
  print(tf.sparse.sparse_dense_matmul(s,s4))
  
  # 加法操作
  try:
      s3 = s + 1
  except TypeError as ex:
      print(ex)
  ```

  ```bash
  # 乘法操作
  SparseTensor(indices=tf.Tensor(
  [[0 1]
   [1 0]
   [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([2. 4. 6.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
  tf.Tensor(
  [[0. 2. 0. 0.]
   [4. 0. 0. 0.]
   [0. 0. 0. 6.]], shape=(3, 4), dtype=float32)
   
   tf.Tensor(
  [[ 30.  40.]
   [ 20.  40.]
   [210. 240.]], shape=(3, 2), dtype=float32)
   # 加法操作
   unsupported operand type(s) for +: 'SparseTensor' and 'int'
  ```

  indices没有排序情况(sparsetensor常见的坑)

  ```python
  s5 = tf.SparseTensor(indices = [[0,2],[0,1],[2,3]],
                     values=[1.,2.,3.],
                     dense_shape=[3,4])
  print(s5)
  print(tf.sparse.to_dense(s5)) # 会报错InvalidArgumentError
  
  s6 = tf.sparse.reorder(s5) # 对s5的indices重新排序
  print(tf.sparse.to_dense(s6)) # 不会报错InvalidArgumentError
  ```

  ```bash
  SparseTensor(indices=tf.Tensor(
  [[0 2]
   [0 1]
   [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
   InvalidArgumentError: indices[1] = [0,1] is out of order [Op:SparseToDense]
   
   tf.Tensor(
  [[0. 2. 1. 0.]
   [0. 0. 0. 0.]
   [0. 0. 0. 3.]], shape=(3, 4), dtype=float32)
  ```

- tf.Variable

  ```python
  v = tf.Variable([[1,2,3],[4,5,6]],dtype=tf.float32)
  print(v)
  print(v.value())
  print(v.numpy())
  
  # 重新赋值
  v.assign(2*v)
  v[1,2].assign(43)
  v[1].assign([7,8,9])
  ```

  ```bash
  <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
  array([[1., 2., 3.],
         [4., 5., 6.]], dtype=float32)>
  tf.Tensor(
  [[1. 2. 3.]
   [4. 5. 6.]], shape=(2, 3), dtype=float32)
  [[1. 2. 3.]
   [4. 5. 6.]]
   
   # 重新赋值(只能用assign函数，用=是不可以的)
   <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
  array([[ 2.,  4.,  6.],
         [ 8., 10., 12.]], dtype=float32)>
  
  <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
  array([[ 2.,  4.,  6.],
         [ 8., 10., 43.]], dtype=float32)>
  
  <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
  array([[2., 4., 6.],
         [7., 8., 9.]], dtype=float32)>
  ```

## 自定义损失函数

> 在房价回归预测的程序中使用自定义损失函数

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

# 搭建模型
model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]))
model.add(keras.layers.Dense(1))
model.summary()


def customized_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


model.compile(loss=customized_loss,
              optimizer="sgd",
              metrics=['accuracy'，'mse']
              )
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=5, min_delta=1e-3
    )
]
history = model.fit(x_train_scaler, y_train,
                    validation_data=(x_valid_scaler, y_valid),
                    epochs=100,
                    callbacks=callbacks)

```

> ETA: 0s - loss: 0.3770 - accuracy: 0.0029 - mse: 0.3770 
>
> 可见自定义的loss和mse计算的结果是一致的

## 自定义Layer

```python
# 自定义激活层 tf.nn.softplus: log(1+e^-x)
customized_softplus = tf.keras.layers.Lambda(lambda x: tf.nn.softplus(x), name="ActiveLayer")
print(customized_softplus([-10., -5., 0., 5., 10.]))

# 自定义Denselayer
class CustomizedDenlayer(keras.layers.Layer):
    def __init__(self, unit, activation=None, **kwargs):
        self.units = unit
        self.activation = keras.layers.Layer(activation)
        super(CustomizedDenlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """构建所需要的参数"""
        self.kernel = self.add_weight(name="kernel",
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(CustomizedDenlayer, self).build(input_shape)

    def call(self, x, **kwargs):
        """完成正向计算"""
        return self.activation(x @ self.kernel + self.bias)


# 搭建模型
model = keras.models.Sequential()
model.add(CustomizedDenlayer(30, activation="relu", input_shape=x_train.shape[1:]))
model.add(CustomizedDenlayer(1))
model.add(customized_softplus)
# 相当于下面两种方式
# 方式1
# model.add(keras.layers.Dense(1, activation="softplus"))
# 方式2
# model.add(keras.layers.Dense(1))
# model.add(keras.layers.Activation("softplus"))
print(model.summary())
```

## tf.function的使用

- 将普通的python代码转为图结构

  ```python
  # @tf.function # 可以使用修饰函数
  def scaled_elu(z, scale=1.0, alpha=1.0):
      # z >=0 ? scale *z : scale * alpha * tf.nn.elu(z)
      is_positive = tf.greater_equal(z, 0.0)
      # 满足条件的替换成第一个参数 不满足条件的替换程第二个参数
      return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))
  
  print(scaled_elu(tf.constant(-3.)))
  print(scaled_elu(tf.constant([-3., -2.5])))
  
  # 将python语法函数转为tf的函数,优势是快
  scaled_elu_tf = tf.function(scaled_elu)
  print(scaled_elu_tf(tf.constant(-3.)))
  print(scaled_elu_tf(tf.constant([3., -2.5])))
  ```

- 对function的参数做限定

  ```python
  @tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
  def cube(z):
      return tf.pow(z, 3)
  
  print(cube(tf.constant([1, 2, 3])))
  print(cube(tf.constant([1., 2., 3.])))
  
  ```

  ```bash
  tf.Tensor([ 1  8 27], shape=(3,), dtype=int32)
  
  ValueError: Python inputs incompatible with input_signature:
    inputs: (
      tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32))
    input_signature: (
      TensorSpec(shape=(None,), dtype=tf.int32, name='x'))
  ```

## 自定义求导

```python
def f(x):
    return 3.0 * x ** 2 + 2.0 * x - 1

# 求导
def approximate_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)

print(approximate_derivative(f, 1))

def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)

# 求偏导
def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x: g(x, x1), x2, eps)
    return dg_x1, dg_x2

print(approximate_gradient(g, 2, 3))
```

```python
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)
# tape只能用一次
dz_x1 = tape.gradient(z, x1)
print(dz_x1) # tf.Tensor(9.0, shape=(), dtype=float32)
try:
    dz_x2 = tape.gradient(z, x2)
except RuntimeError as ex:
    # GradientTape.gradient can only be called once on non-persistent tapes.
    print(ex)

# 解决上面的问题
with tf.GradientTape(persistent=True) as tape:
    z = g(x1, x2)
dz_x1 = tape.gradient(z, x1) # tf.Tensor(9.0, shape=(), dtype=float32)
dz_x2 = tape.gradient(z, x2)# tf.Tensor(42.0, shape=(), dtype=float32)
del tape # 需要手动删除tape，因为配置persistent=True后系统不会自动回收

# 可以直接求x1和x2的偏导
with tf.GradientTape() as tape:
    z = g(x1, x2)
dz_x1x2 = tape.gradient(z,[x1,x2])
print(dz_x1x2) 
# [<tf.Tensor: id=46, shape=(), dtype=float32, numpy=9.0>,
# <tf.Tensor: id=52, shape=(), dtype=float32, numpy=42.0>]

```

对常数求导

```python
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
dz_x1 = tape.gradient(z, x1)
dz_x2 = tape.gradient(z, x2)
print(dz_x1)
print(dz_x2)

with tf.GradientTape() as tape:
    z = g(x1, x2)
dz_x1x2 = tape.gradient(z, [x1, x2])
print(dz_x1x2) # [None, None]

# 输出None，当要观察constant的梯度时候可以用tf.watch()方法
with tf.GradientTape() as tape:
    tape.watch(x1) # 可以关注constant的梯度
    tape.watch(x2) # 可以关注constant的梯度
    z = g(x1, x2)
dz_x1x2 = tape.gradient(z, [x1, x2])
print(dz_x1x2)
# [<tf.Tensor: id=46, shape=(), dtype=float32, numpy=9.0>,
# <tf.Tensor: id=52, shape=(), dtype=float32, numpy=42.0>]
```

对多函数求导

```python
x1 = tf.Variable(5.0, name="x1")
x2 = tf.Variable(5.0, name="x2")
with tf.GradientTape() as tape:
    z1 = 3 * x1 + 4 * x2
    z2 = x1 ** 2 + x2 ** 6
print(tape.gradient([z1, z2], [x1, x2]))
# 输出是[z1对x1的偏导+z2对x1的偏导. z1对x2的偏导+z2对x2的偏导]
# [<tf.Tensor: id=45, shape=(), dtype=float32, numpy=13.0>, 
# <tf.Tensor: id=46, shape=(), dtype=float32, numpy=18754.0>]

```

求二阶导数

```python
def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)

x1 = tf.Variable(2.0)
y1 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as out_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, y1)
    inner_grads = inner_tape.gradient(z, [x1, y1])
outer_grads = [out_tape.gradient(inner_grad, [x1, y1]) for inner_grad in inner_grads]
print(inner_grads)
print(outer_grads)
del inner_tape
del out_tape

# [[None, <tf.Tensor: id=83, shape=(), dtype=float32, numpy=6.0>], [<tf.Tensor: id=94, shape=(), dtype=float32, numpy=6.0>, <tf.Tensor: id=92, shape=(), dtype=float32, numpy=14.0>]]
```

反向传播算法实现

```python
def f(x):
    return 3.0 * x ** 2 + 2.0 * x - 1


learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx)

print(x)
```

```
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-0.3333333>
```

## 自定义优化器

```python
learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    # x.assign_sub(learning_rate * dz_dx)
    optimizer.apply_gradients([(dz_dx, x)])

print(x)
```

```python
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-0.3333333>
```

## 手动实现神经网络用于房价回归预测

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

# 搭建模型
model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]))
model.add(keras.layers.Dense(1))
model.summary()
model.compile(loss="mean_squared_error",optimizer="sgd")

# metric的使用
metric = keras.metrics.MeanSquaredError()
print(metric([5.], [2.]))
print(metric([1.], [0.]))
print(metric.result())  # 累加数据

metric.reset_states()
print(metric([1.], [3.]))
print(metric.result())  # 清除之前的累加

# 开始训练
epochs = 100
batch_size = 32
steps_per_epoch = len(x_train_scaler) // batch_size
optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquaredError()

# 根据每一次batch的数据
def random_batch(x, y, batch_size=32):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]

# 开始训练
for epoch in range(epochs):
    metric.reset_states()# 每次epoch都清除metric
    # 开始batch遍历
    for step in range(steps_per_epoch):
        # 获取batch数据
        x_batch, y_batch = random_batch(x_train_scaler, y_train, batch_size=batch_size)
        
        with tf.GradientTape() as tape:
            y_pred = model(x_batch) # 正向传播
            loss = tf.reduce_mean(keras.losses.mean_squared_error(y_batch, y_pred))# 获得loss
            metric(y_batch, y_pred)# 记录loss(事先定义的mse)
        # 反向传播
        grads = tape.gradient(loss, model.variables)
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradients(grads_and_vars)
        # 输出每一次的batch结果
        print("\rEpoch", epoch, "train mse:", metric.result(), end='')
        
    # 输出每一次epoch的结果
    y_valid_pred = model(x_valid_scaler)
    valid_loss = tf.reduce_mean(keras.losses.mean_squared_error(y_valid_pred, y_valid))
    print("\r", epoch, "valid mse:", valid_loss, end='')
```

