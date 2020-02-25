---
title: Python连接MSSQL
date: 2018-04-17 18:58:45
tags: [Python,SQLSERVER,MSSQL]
categories: [Python,数据库,ds]
copyright: yitakabe
---
## 写在前面
&emsp;&emsp;最近需要做一些爬虫的工作，将爬取的内容需要存到SQL-SERVER数据库中。爬虫准备采用Python的Scrapy框架，因此将爬虫结果存进SQL-SERVER数据库自然是需要用Python来进行处理。Python有一个专门操作SQLSERVER的库叫做pymssql。
<!--more-->
## 安装
1. 下载pymssql
&emsp;&emsp;从[**这里**](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pymssql)选取合适的版本进行下载。可能下载的速度有点慢，[**这里**](.\pymssql-2.1.4.dev5-cp36-cp36m-win_amd64.whl)提供了python3.6_x64版本的whl文件。

2. 安装pymssql
&emsp;&emsp;通过命令`pip install pymssql-2.1.4.dev5-cp36-cp36m-win_amd64.whl`进行安装

## pymssql的使用
``` python
import pymssql

conn = pymssql.connect(host='127.0.0.1', user='sa', password='1', database='mytest')
cur = conn.cursor()

cur.execute("insert into [dbo].[tabel] (name) values ('张三')")
conn.commit()  # 如果update/delete/insert记得要conn.commit()

cur.execute('select top 5 * from [dbo].[tabel]')
print(cur.fetchall())

cur.close()
conn.close()

```

