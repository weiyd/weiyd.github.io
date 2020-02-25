---
title: "ssm系列教程2-Mybatis入门实践"
date: 2019-05-01T19:42:04+08:00
draft: false
tags: ["mybatis"]
series: ["SSM框架系列笔记"]
categories: ["Java"]
toc: true
img: "/thumb/2.jpg"
summary: "介绍Mybatis之实现sql查询"
typora-copy-images-to: ssm系列教程2-mybatis入门实践
typora-root-url: ssm系列教程2-mybatis入门实践
---

# Mybatis简介

​		MyBatis 是一款优秀的持久层框架，它支持定制化 SQL、存储过程以及高级映射。MyBatis 避免了几乎所有的 JDBC 代码和手动设置参数以及获取结果集。MyBatis 可以使用简单的 XML 或注解来配置和映射原生信息，将接口和 Java 的 POJOs(Plain Ordinary Java Object,普通的 Java对象)映射成数据库中的记录。



## 1 新建项目

- 选择模板

  <img src="./1567838352044.png" alt="1567838352044" style="zoom: 80%;"/>

- 配置组织名与项目名

  <img src="./1567838117120.png" alt="1567838117120" style="zoom:50%;" />

- 配置Maven（提前配置好的话可选默认）

  ![1567838165970](./1567838165970.png)

- 配置项目位置

  ![1567838195617](./1567838195617.png)

- Maven构建

  ![1567838594374](./1567838594374.png)

## 2 安装依赖

- maven搜索依赖包

  ![1567838710837](./1567838710837.png)

- 安装Mybatis依赖

  - ![1567838742365](./1567838742365.png)

  - ```
    <!-- https://mvnrepository.com/artifact/org.mybatis/mybatis -->
    <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis</artifactId>
        <version>3.5.2</version>
    </dependency>
    ```

  - 复制到pom文件

- 安装mysql依赖

  - 版本号选择与使用的数据保持一致

  - ![1567838823839](./1567838823839.png)

  - ```
    <!-- https://mvnrepository.com/artifact/mysql/mysql-connector-java -->
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <version>8.0.15</version>
    </dependency>
    ```

  - 复制到pom文件

- 检查是否导入

  ![1567839086182](./1567839086182.png)

## 3 配置映射

### 3.1 MybatisUtil类

- 在com.lovefit下新建util.MybatisUtil类

- ![1567839285122](./1567839285122.png)

- ```java
  package com.lovefit.util;
  
  import org.apache.ibatis.io.Resources;
  import org.apache.ibatis.session.SqlSession;
  import org.apache.ibatis.session.SqlSessionFactory;
  import org.apache.ibatis.session.SqlSessionFactoryBuilder;
  
  import java.io.IOException;
  import java.io.InputStream;
  
  public class MybatisUtil {
  
      private static SqlSessionFactory sqlSessionFactory;
  
      static {
          String resource="mybatis.cfg.xml";
          InputStream in =null;
          try {
              in = Resources.getResourceAsStream(resource);
              sqlSessionFactory = new SqlSessionFactoryBuilder().build(in);
          } catch (IOException e) {
              e.printStackTrace();
          }finally {
              if(in !=null){
                  try {
                      in.close();
                  } catch (IOException e) {
                      e.printStackTrace();
                  }
              }
          }
      }
  
      public static SqlSession getSession(){
          return sqlSessionFactory.openSession();
      }
  }
  ```

### 3.2 Pojo类

- 新建pojo.Girl类

  - ```java
    package com.lovefit.pojo;
    
    import java.util.Date;
    
    public class Girl {
    
        private long id;
    
        private String name;
    
        private String flower;
    
        private Date birthday;
    
    
        public long getId() {
            return id;
        }
    
        public void setId(long id) {
            this.id = id;
        }
    
        public String getName() {
            return name;
        }
    
        public void setName(String name) {
            this.name = name;
        }
    
        public String getFlower() {
            return flower;
        }
    
        public void setFlower(String flower) {
            this.flower = flower;
        }
    
        public Date getBirthday() {
            return birthday;
        }
    
        public void setBirthday(Date birthday) {
            this.birthday = birthday;
        }
    }
    
    ```

  - 需要封装字段

### 3.3 Mapper接口

- 在``src\main\java\com\lovefit``下新建mapper文件夹

- 在mapper文件夹下新建GirlMapper.java文件

  - ```java
    package com.lovefit.mapper;
    import com.lovefit.pojo.Girl;
    
    public interface GirlMapper {
    
        int insert(Girl g);

        Girl queryByID(int id);
    
    }
    ```

### 3.4 配置MybatisUtil类中的配置文件

- 新建resources文件夹与mybatis.cfg.xml

  - 在src/main/java新建resources文件夹

  - sources文件夹配置为source root 

  - 在sources文件夹下新建mybatis.cfg.xml文件

  - 在mybatis.cfg.xml输入以下内容

    - ```xml
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE configuration PUBLIC "-//mybatis.org//DTD Config 3.0//EN" "http://mybatis.org/dtd/mybatis-3-config.dtd">
      <configuration>
          <environments default="dev">
              <environment id="dev">
                  <transactionManager type="JDBC"></transactionManager>
                  <dataSource type="UNPOOLED">
                      <property name="url" value="jdbc:mysql://localhost:3306/mybatis?&amp;characterEncoding=utf8&amp;serverTimezone=GMT"/>
                      <property name="driver" value="com.mysql.cj.jdbc.Driver"/>
                      <property name="username" value="root"/>
                      <property name="password" value="root"/>
                  </dataSource>
              </environment>
          </environments>
      
          <mappers>
              <!--不要写. 要写斜杠-->
              <mapper resource="com/lovefit/mapper/GirlMapper.xml"/>
          </mappers>
      </configuration>
      ```

    - 其中url是连接字符串，使用mysql8以上的版本话需要配置时区

    - mappers是需要映射的Pojo

- 新建mapper文件夹

  - 在src/main/resources下新建com/lovefit/mapper文件夹

  - mapper文件夹下新建GirlMapper.xml文件

  - GirlMapper.xml文件输入以下内容

    - ```xml
      <?xml version="1.0" encoding="UTF-8" ?>
      <!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Config 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
      
      <mapper namespace="com.lovefit.mapper.GirlMapper">
      
          <insert id="insert">
              insert into girl (name,flower,birthday) values (#{name},#{flower},#{birthday})
          </insert>

          <select id="queryByID" resultType="com.lovefit.pojo.Girl">
              select * from girl where id = #{id}
          </select>
      
      </mapper>
      
      ```

  - namespace是需要映射的接口

### 3.5 测试

- ```java
  public class Test1 {
  
    @Test
    public void m1()
    {
        SqlSession sqlSession = MybatisUtil.getSession();

        GirlMapper mapper = sqlSession.getMapper(GirlMapper.class);

        Girl g = new Girl();
        g.setName("林心如");
        g.setFlower("霍建花");
        g.setBirthday(new Date());

        mapper.insert(g);
        sqlSession.commit();

        sqlSession.close();
    }

    @Test
    public void m2()
    {
        SqlSession sqlSession = MybatisUtil.getSession();

        GirlMapper mapper = sqlSession.getMapper(GirlMapper.class);
        Girl girl = mapper.queryByID(1);
  
        assert girl.getName().equals("林心如");
        sqlSession.close();
    }
  }
  ```

- ![1567840699575](./1567840699575.png)

### 3.6 数据库文件

  ```sql
/*
Navicat MySQL Data Transfer

Source Server         : 本地mysql
Source Server Version : 80015
Source Host           : localhost:3306
Source Database       : mybatis

Target Server Type    : MYSQL
Target Server Version : 80015
File Encoding         : 65001

Date: 2019-09-07 15:16:47
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for girl
-- ----------------------------
DROP TABLE IF EXISTS `girl`;
CREATE TABLE `girl` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) DEFAULT NULL,
  `flower` varchar(255) DEFAULT NULL,
  `birthday` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
  ```

