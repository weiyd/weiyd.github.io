---
title: Vue2.0 学习笔记
date:  2018-1-7 02:27:59
categories: 
    - 前端
tags: 
    - 前端 
    - VueJS
keywords:  
description: 
    本文主要看《Vue.js权威指南》做的笔记。书中的实例是基于Vue1.0编写的，现在版本是2.0+，相比改动了不少东西，因此在阅读此书过程中，用Vue2实现一遍Vuw的内部指令，在此做一下笔记。
---
## Vue.js2.0 笔记
### 内部指令
#### v-if
``` HTML
<body class="native">
<div id="example">
    <p v-if="greeting">Hello</p>
</div>
</body>
<script>
    var exampleVM = new Vue({
        el: "#example",
        data: {
            greeting: false
        }
    })
</script>
```

#### v-show
``` HTML
<body>
<div id="example">
    <p v-show="greeting">Hello</p>
</div>
</body>
<script>
    var exampleVM = new Vue({
        el: "#example",
        data: {
            greeting: true
        }
    })
</script>
```

#### v-if和v-show的比较
&emsp;&emsp;在切换v-if时候，Vue.js有一个局部编译和卸载过程，因为v-if中模板可能包括数据绑定或子组件，v-id是惰性的，如果初始渲染条件为假，什么也做不做，在条件第一次变为真时，才开始进行局部编译（编译会被缓存起来）。<br>
&emsp;&emsp;相比之下，v-show元素始终被编译并保留，只是简单地基于CSS切换。<br>
&emsp;&emsp;一般来说，v-if具有更高的切换消耗，而v-show有更高的初始渲染消耗。**因此需要频繁的切换，使用v-show比较好。如果运行时条件不大可能变化，使用v-if较好。**<br>
#### v-else
``` HTML
<body>
<div id="example">
    <p v-if="greeting">greeting is ok</p>
    <p v-else="greeting">greeting is not ok</p>
</div>
</body>
<script>
    var exampleVM = new Vue({
        el: "#example",
        data: {
            greeting: false
        }
    })
</script>
```
#### v-model
&emsp;&emsp;v-model指令用来在input、select、text、checkbox、radio等表单控件元素上创建双向数据绑定。根据控件类型v-model自动选择正确的方法更新元素。
``` HTML
<body id="example">
<form>
    姓名：
    <input type="text" v-model="data.name" placeholder="">
    <br/>
    性别：
    <input type="radio" id="man" value="One" v-model="data.sex">
    <label for="man">男</label>
    <input type="radio" id="woman" value="Two" v-model="data.sex">
    <label for="woman">女</label>
    <br/>
    兴趣：
    <input type="checkbox" id="book" value="book" v-model="data.interest">
    <label for="book">阅读</label>
    <input type="checkbox" id="swim" value="swim" v-model="data.interest">
    <label for="swim">游泳</label>
    <input type="checkbox" id="game" value="game" v-model="data.interest">
    <label for="game">游戏</label>
    <input type="checkbox" id="song" value="song" v-model="data.interest">
    <label for="song">唱歌</label>
    <br/>
    身份：
    <select v-model="data.identity">
        <option value="teacher" selected>教师</option>
        <option value="doctor">医生</option>
        <option value="lawyer">律师</option>
    </select>
    <br/>
</form>
</body>
<script>
    var exampleVM = new Vue({
        el: 'example',
        data: {
            name: '',
            sex: '',
            interest: [],
            identity: ''
        }
    })
</script>
```
注1：v-model在input事件中同步输入框的值和数据，如果要伴随时间改变而不是同步改变的话，可以参考如下代码：
``` HTML
<body>
<div id="example">
    <input v-model.lazy="name">
    {{name}}
</div>
</body>
<script>
    var a = new Vue({
        el: "#example",
        data: {
            name: "输入框外点击鼠标显示输入内容"
        }
    })
</script>
```
注2：v-moel可以对输入框的内容进行trim操作，具体如下代码：
``` HTML
<body>
<div id="example">
    <input v-model.trim="name">
    {{name}}
</div>
</body>
<script>
    var a = new Vue({
        el: "#example",
        data: {
            name: "  前后都有空格  "
        }
    })
</script>
```
#### v-for
当使用v-for时，v-model不再生效。`<input v-for="str in strings" v-model="str">`
``` HTML
<body>
<div id="example">
    <li v-for="(item,index) in items">
        序号：{{index}}  工号：{{item.id}} 姓名：{{item.msg}}
    </li>
</div>
</body>
<script>
    var a = new Vue({
        el: "#example",
        data: {
            items: [
                {
                    id: '001',
                    msg: "张三"
                },
                {
                    id:'002',
                    msg: "李四"
                },
            ]
        }
    })
</script>
```
#### v-html
``` HTML
<body>
<div id="example" v-html="html">
    {{html}}
</div>
</body>
<script>
    var a = new Vue({
        el: "#example",
        data: {
            html:"<li>这是一个innerHtml</li>"
        }
    })
</script>
```
注：不建议在网站上直接动态渲染任意HTML片段，这样很容易导致XSS攻击。