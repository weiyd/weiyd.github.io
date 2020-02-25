---
title: Docker镜像的基本操作
date:  2017-12-3 22:27:59
categories: 
    - Docker
tags: 
    - Docker
keywords:  Docker
description: 本文主要记录了笔者实践《Docker技术入门与实践》过程中做的一些笔记，笔记内容主要是Docker镜像的基本操作。
---
### 获取镜像
``` bash
sudo docker pull [server]/imagename[:tag]
```
如果不指定tag，会默认使用`latest`标签,默认服务器是`registry.hub.docker.com`,也可以使用其他注册服务器进行pull，例如DockerPool社区的镜像源`dl.dockerpool.com`下载最新的镜像。例如:
``` bash
sudo docker pull dl.dockerpool.com/ubuntu:14.04
```
### 查看镜像信息
``` bash
sudo docker images
>>
REPOSITORY       TAG       IMAGE_ID        CREATED          SIZE
ubuntu           14.04     20c44cd7596f    2 weeks ago      123 MB
```
### 为镜像打TAG
``` bash
sudo docker tag ubuntu:14.04 ubuntu:mytag
```
tag标签是指向了同一个镜像，只是镜像的别名而已，标签起到了引用和快捷方式的作用。通过`sudo docker images`查看镜像：
``` bash
REPOSITORY       TAG       IMAGE_ID        CREATED          SIZE
ubuntu           14.04     20c44cd7596f    2 weeks ago      123 MB
ubuntu           mytag     20c44cd7596f    2 weeks ago      123 MB
```
可以通过`sudo docker inspect REPOSITORY:TagName`或者`sudo docker inspect REPOSITORY:Image_ID`查看镜像的详细信息。
``` bash
[
    {
        "Id": "sha256:20c44cd7596ff4807aef84273c99588d22749e2a7e15a7545ac96347baa65eda",
        "RepoTags": [
            "ubuntu:14.04",
            "ubuntu:latest",
            "ubuntu:mytag"
        ],
        "RepoDigests": [
            "ubuntu@sha256:7c67a2206d3c04703e5c23518707bdd4916c057562dd51c74b99b2ba26af0f79"
        ],
        "Parent": "",
        "Comment": "",
        "Created": "2017-11-17T21:59:25.014645802Z",
        "Container": "e5f1a9df75b86a5d803eaf6f3fed6a0f8ef5fbf15a6c5039df087e4348ed8171",
        "ContainerConfig": {
            "Hostname": "e5f1a9df75b8",
            "Domainname": "",
            "User": "",
            "AttachStdin": false,
            "AttachStdout": false,
            "AttachStderr": false,
            "Tty": false,
            "OpenStdin": false,
            "StdinOnce": false,
            "Env": [
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ],
            "Cmd": [
                "/bin/sh",
                "-c",
                "#(nop) ",
                "CMD [\"/bin/bash\"]"
            ],
            "ArgsEscaped": true,
            "Image": "sha256:b5771e7d8dcc594b886dbdd6a9c3de60d45252ca657dfdff6e1d996728dfa2cd",
            "Volumes": null,
            "WorkingDir": "",
            "Entrypoint": null,
            "OnBuild": null,
            "Labels": {}
        },
        "DockerVersion": "17.06.2-ce",
        "Author": "",
        "Config": {
            "Hostname": "",
            "Domainname": "",
            "User": "",
            "AttachStdin": false,
            "AttachStdout": false,
            "AttachStderr": false,
            "Tty": false,
            "OpenStdin": false,
            "StdinOnce": false,
            "Env": [
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ],
            "Cmd": [
                "/bin/bash"
            ],
            "ArgsEscaped": true,
            "Image": "sha256:b5771e7d8dcc594b886dbdd6a9c3de60d45252ca657dfdff6e1d996728dfa2cd",
            "Volumes": null,
            "WorkingDir": "",
            "Entrypoint": null,
            "OnBuild": null,
            "Labels": null
        },
        "Architecture": "amd64",
        "Os": "linux",
        "Size": 122792927,
        "VirtualSize": 122792927,
        "GraphDriver": {
            "Name": "aufs",
            "Data": null
        },
        "RootFS": {
            "Type": "layers",
            "Layers": [
                "sha256:788ce2310e2fdbbf81fe21cbcc8a44da4cf648b0339b09c221abacb
                4cd5fd136",
                "sha256:aa4e47c4511638484cd5d95eadd7a8e4da307375ba31ff50d47aa9065dce01e0",
                "sha256:b3968bc26fbd527f214f895aeef940a6930c62d853fe8b12bd479f0b53518150",
                "sha256:c9748fbf541d3e043521e165b015d45825de33c00a8acb037443cfbd0cb5e677",
                "sha256:2f5b0990636a87f1557d64ba39808dcd64031328b2a159c5805115b8e725bbbc"
            ]
        }
    }
]
```
上面返回的JSON信息太多，如果只需要其中部分内容，可以使用`-f`参数来指定，例如获取Architecture信息：</br>
`sudo docker inspect -f {{".Architecture"}} ubuntu:14.04`</br>
返回内容：</br>
`amd64`</br>
### 删除镜像
``` bash
sudo docker rmi REPOSITORY:TagName
```
当镜像只剩下一个标签时候，使用此命令会彻底删除该镜像。
### 创建镜像 
创建镜像有三种方式：基于已有镜像的容器创建、基于本地模板导入、基于Dockerfile创建。



