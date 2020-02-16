---
title: plotly数据可视化入门
date: 2020-02-07 16:30:22
tags: [数据可视化]
categories: [python]
typora-copy-images-to: plotly入门
typora-root-url: plotly入门
---

![bg](/bg.jpg)

<!--more-->

# plotly数据可视化入门

## 安装

``conda install plotly``

## 包导入

```python
import plotly
import pandas as pd
import plotly.graph_objects as go
```

## 数据导入

```python
data = pd.read_csv('data/datasets/nz_weather.csv')
data.head()
```

## 画图

### 曲线图和散点图

- 曲线图

    ```python
    # 画线和画点都用Scatter
    line = go.Scatter(x= data['DATE'],y= data['Auckland'],name="Auckland")
    line2 = go.Scatter(x= data['DATE'],y= data['Wellington'],name="Wellington")
    # 创建一张图
    fig = go.Figure([line,line2])
    fig.update_layout(
        title="新西兰天气",
        xaxis_title="Date",
        yaxis_title="Weather"
    )
    fig.show()
    ```

![image-20200216120950509](image-20200216120950509.png)

- 散点图

  ```python
  import plotly.graph_objects as go
  import pandas as pd
  
  data = pd.read_csv("data/datasets/iris.csv")
  
  data.groupby("Name").count().index
  name_to_color = {
      "Iris-setosa": 0,
      "Iris-versicolor": 1,
      "Iris-virginica": 2
  }
  data["color"] = data["Name"].map(name_to_color)
  
  points = go.Scatter(
      x=data["SepalLength"],
      y=data["SepalWidth"],
      mode="markers",
      marker={
          "color": data["color"]
      }
  )
  fig = go.Figure(points)
  fig.show()
  ```
  
  ![image-20200216121903720](image-20200216121903720.png)

### 条形图

```python
data_2010 = data[(data["DATE"]>="2010-01") & (data["DATE"]<"2011-01")]
bar1 = go.Bar(
    x=data_2010["DATE"],
    y=data_2010["Auckland"],
    text=data_2010["Auckland"],
    textposition="outside",
    name="Auckland"
)
bar2 = go.Bar(
    x=data_2010["DATE"],
    y=data_2010["Wellington"],
    text=data_2010["Wellington"],
    textposition="outside",
    name="Wellington"
)
fig = go.Figure([bar1,bar2])
fig.show()
```

![image-20200216121127543](image-20200216121127543.png)

### 直方图Histogram

```python
hist = go.Histogram(
    x=data['Auckland'],
    xbins={"size":10}
)
go.Figure(hist)
fig = go.Figure([hist])
fig.update_layout(
    bargap=0.1
)
fig.show()

```

![image-20200216121345988](image-20200216121345988.png)

### 3D图

```python
import plotly.graph_objects as go
import pandas as pd

data = pd.read_csv("data/datasets/3d-line1.csv")

line =  go.Scatter3d(x=data['x'],y=data['y'],z=data['z'])
fig = go.Figure(line)
fig.show()
```

![image-20200216122407669](image-20200216122407669.png)

```python
import plotly.graph_objects as go
import pandas as pd

data = pd.read_csv("data/datasets/3d-line1.csv")
data.head()

line = go.Scatter3d(
    x=data['x'],
    y=data['y'],
    z=data['z'],
    mode="markers",
    marker={
        "size": 5
    }
)
fig = go.Figure(line)
fig.show()

line = go.Scatter3d(x=data['x'], y=data['y'], z=data['z'])
fig = go.Figure(line)
fig.show()
```

![image-20200216122531369](/image-20200216122531369.png)

### Surfacer

```python
import numpy as np

x = np.arange(-5, 6)
y = np.arange(-5, 6)
xv, yv = np.meshgrid(x, y)
z = xv ** 2 + yv ** 2

surface = go.Surface(x=xv, y=yv, z=z)
fig = go.Figure(surface)
fig.show()
```



![image-20200216123859439](image-20200216123859439.png)



## express模块

### 散点图

```python
# 为了不用自己构建映射字典，plotly又添加了express模块
import plotly.express as px
fig = px.scatter(data,
                 x ="SepalLength",
                 y="SepalWidth",
                 color='Name'
                 )
fig.show()

```

![image-20200216122254623](image-20200216122254623.png)