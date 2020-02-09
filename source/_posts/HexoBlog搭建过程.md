---

title: Hexo博客搭建过程
date: 2020-02-07 16:30:22
tags: []
categories: []

---

# 配置环境

## 安装Node js

## 安装hexo

``npm install -g hexo-cli``

# 使用Hexo

## 创建blog

``hexo init``

## 新建文章

``hexo new post 第一篇文章``

### 编辑文章

```markdown
---
title: 第一篇文章
date: 2020-02-07 16:30:22
tags: [第一个tag]
categories: [第一个分类]
---

# 第一节
## 第一小节
xxxxx
# 第二节
## 第二小节
xxxxx
```



## 生成

``hexo g``

## 启动本地测试服务器

``hexo s``

# 修改主题

## 下载主题

``git clone https://github.com/iissnan/hexo-theme-next themes/next``

## 配置主题

配置文件：``_config.yml``

修改：``theme: next``

## 配置menu

### 配置config文件

配置文件：``_config.yml``

修改：

```
menu:
  home: /|| home
  about: /about/|| user
  tags: /tags/|| tags
  categories: /categories/|| th
  archives: /archives/|| archive
```

### 添加文件夹

- 添加tags

  - ``hexo new page tags``

  - 在``source/``下生成``tags``文件

  - 修改``index.md``

    ```markdown
    title: 标签
    type: "tags"
    ```

    

- 添加categories

  - ``hexo new page categories``

  - 在``source/``下生成``categories``文件

  - 修改``index.md``

    ```markdown
    title: 分类
    type: "categories"
    ```

- 添加about

  - ``hexo new page about``

  - 在``source/``下生成``about``文件

  - 修改``index.md``

    ```markdown
    layout: "about"
    title: "About"
    comments: true
    
    自我介绍
    ```

## 配置语言

将根目录下的``_config.yml``下``language: en``修改为``language: zh-Hans ``

# 添加搜索功能

## 安装搜索插件

``npm install hexo-generator-searchdb --save``

## 修改配置文件

在根目录下的/theme/next/_config.yml文件中添加配置

```bash
search:
  path: search.xml
  field: post
  format: html
  limit: 10000
```

在根目录下的/theme/next/_config.yml文件中搜索`local_search`，将`enable`改为`true`

```
local_search:
  enable: true
```

# 添加文章插入图片功能

## 安装插件

``npm install https://github.com/CodeFalling/hexo-asset-image --save``

## 修改配置文件

修改根目录下的``_config.yml``中的``post_asset_folder: false``改为``post_asset_folder: true``

# 部署

## 安装插件

``npm install hexo-deployer-git --save``

## 配置根目录的配置文件

```bash
deploy:
  type: git
  repository: https://github.com/weiyd/weiyd.github.io.git
  branch: master
```

