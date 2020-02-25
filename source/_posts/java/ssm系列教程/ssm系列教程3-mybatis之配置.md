---
title: "ssm系列教程3-Mybatis之配置"
date: 2019-05-01T20:42:04+08:00
draft: false
tags: ["mybatis"]
series: ["SSM框架系列笔记"]
categories: ["Java"]
toc: true
img: "/thumb/2.jpg"
summary: "介绍Mybatis的配置文件"
typora-copy-images-to: ssm系列教程3-mybatis之配置
typora-root-url: ssm系列教程3-mybatis之配置
---

# Mybatis的配置文件

​		在mybatis入门实践中介绍了MybatisUtil类，其中resource变量配置了mybatis的配置文件路径，如果只有名字没有路径的话，默认在项目的sources root目录下（本项目指定为``src/main/resources``）下查找。

- MybatisUtil类

  ```java
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

- mybatis.cfg.xml

  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE configuration PUBLIC "-//mybatis.org//DTD Config 3.0//EN" "http://mybatis.org/dtd/mybatis-3-config.dtd">
  <configuration>
  
      <!--配置文件位置-->
      <properties resource="jdbc.properties">
          <!--优先使用jdbc.properties里面的属性-->
          <!--dataSource的优先级最高-->
          <property name="username" value="root"/>
          <property name="password" value="root"/>
          <property name="driver" value="com.mysql.cj.jdbc.Driver"/>
          <property name="url" value="jdbc:mysql://localhost:3306/mybatis?&amp;characterEncoding=utf8&amp;serverTimezone=GMT"/>
      </properties>
      <settings>
          <!--缓存开启-->
          <setting name="cacheEnabled" value="true"/>
          <setting name="lazyLoadingEnabled" value="true"/>
          <setting name="multipleResultSetsEnabled" value="true"/>
          <setting name="useColumnLabel" value="true"/>
          <setting name="useGeneratedKeys" value="false"/>
          <setting name="autoMappingBehavior" value="PARTIAL"/>
          <setting name="autoMappingUnknownColumnBehavior" value="WARNING"/>
          <setting name="defaultExecutorType" value="SIMPLE"/>
          <setting name="defaultStatementTimeout" value="25"/>
          <setting name="defaultFetchSize" value="100"/>
          <setting name="safeRowBoundsEnabled" value="false"/>
          <!--下划线风格转换为驼峰-->
          <setting name="mapUnderscoreToCamelCase" value="false"/>
          <setting name="localCacheScope" value="SESSION"/>
          <setting name="jdbcTypeForNull" value="OTHER"/>
          <setting name="lazyLoadTriggerMethods" value="equals,clone,hashCode,toString"/>
          <setting name="logImpl" value="LOG4J"/>
      </settings>
  
  
      <!--类的别名-->
      <typeAliases>
          <!--注册一个简写的类名，可以再其他mapper文件中被引用-->
          <!--不推荐使用,有前缀很清楚-->
          <!--<typeAlias type="com.lovefit.pojo.Girl" alias="girl"></typeAlias>-->
          <!--直接注册整个包，该包之下的所有类都生效，默认规则为简写类名-->
          <!--<package name="com.lovefit.pojo"/>-->
      </typeAliases>
  
      <!--数据类型转换-->
      <!--<typeHandlers>-->
      <!--</typeHandlers>-->
  
      <!--Mybatis的属性配置-->
  
      <environments default="dev">
          <environment id="dev">
              <transactionManager type="JDBC"></transactionManager>
              <dataSource type="UNPOOLED">
                  <!--<property name="url" value="jdbc:mysql://localhost:3306/mybatis?&amp;characterEncoding=utf8&amp;serverTimezone=GMT"/>-->
                  <property name="url" value="${url}"/>
                  <property name="driver" value="${driver}"/>
                  <property name="username" value="${username}"/>
                  <property name="password" value="${password}"/>
              </dataSource>
          </environment>
      </environments>
  
      <mappers>
          <!--不要写. 要写斜杠-->
          <!--第一种：通过类路径引入xml文件-->
          <!--<mapper resource="com/lovefit/mapper/GirlMapper.xml"/>-->
          
          <!--第二种：通过URL 路径协议 引入xml文件-->
          <!--<mapper url="file:\\C:/Users/WYD181/Desktop/SSM_Study/mybatis002/src/main/resources/com/lovefit/mapper/GirlMapper.xml"/>-->
          
          <!--第三种：通过类的接口全限定名引入，必须保持我们的接口和mapper.xml在同包之下-->
          <!--<mapper class="com.lovefit.mapper.GirlMapper"></mapper>-->
  
          <!--第四种：引入一个包的方式，以后只要是在这个包下新建Mapper，不需要重新引入-->
          <package name="com.lovefit.mapper"/>
      </mappers>
  </configuration>
  ```

## 1 properties

​		``properties``属性可以让mybatis配置文件读取mybatis配置文件需要的属性值（方式：``<property name="url" value="${url}"/>``），便于动态的修改mybatis的配置文件。

- 用法1

  - ```xml
    <properties>
        <property name="username" value="root"/>
        <property name="password" value="root"/>
        <property name="driver" value="com.mysql.cj.jdbc.Driver"/>
        <property name="url" value="jdbc:mysql://localhost:3306/mybatis?&amp;characterEncoding=utf8&amp;serverTimezone=GMT"/>
    </properties>
    ```

- 用法2

  - 配置``properties``的resource字段，resource是单独配置属性值的properties文件。

  - ```properties
    username = root
    password=root
    driver=com.mysql.cj.jdbc.Driver
    url=jdbc:mysql://localhost:3306/mybatis?&characterEncoding=utf8&serverTimezone=GMT
    ```

  - ```xml
    <properties resource="jdbc.properties">
    </properties>
    ```

- 优先级比较

  - properties resource> properties value

## 2 settings

mybatis的设置配置，常用的配置如下（具体可以查看

[官网]: http://www.mybatis.org/mybatis-3/zh/configuration.html#settings

）：

```xml
<settings>
    <!--缓存开启-->
    <setting name="cacheEnabled" value="true"/>
    <setting name="lazyLoadingEnabled" value="true"/>
    <setting name="multipleResultSetsEnabled" value="true"/>
    <setting name="useColumnLabel" value="true"/>
    <setting name="useGeneratedKeys" value="false"/>
    <setting name="autoMappingBehavior" value="PARTIAL"/>
    <setting name="autoMappingUnknownColumnBehavior" value="WARNING"/>
    <setting name="defaultExecutorType" value="SIMPLE"/>
    <setting name="defaultStatementTimeout" value="25"/>
    <setting name="defaultFetchSize" value="100"/>
    <setting name="safeRowBoundsEnabled" value="false"/>
    <!--下划线风格转换为驼峰-->
    <setting name="mapUnderscoreToCamelCase" value="false"/>
    <setting name="localCacheScope" value="SESSION"/>
    <setting name="jdbcTypeForNull" value="OTHER"/>
    <setting name="lazyLoadTriggerMethods" value="equals,clone,hashCode,toString"/>
    <setting name="logImpl" value="LOG4J"/>
</settings>
```



## 3 typeAliases

typeAliases是给PoJo类起一个别名，方便mapper.xml调用不用写类的全名路径。

- 用法1

  - ``<typeAlias type="com.lovefit.pojo.Girl" alias="girl_alias"></typeAlias>``

  - 在GirlMapper.xml中引用

    - ```xml
      <mapper namespace="com.lovefit.mapper.GirlMapper">
          <select id="queryByID" resultType="girl_alias">
              select * from girl where id = #{id}
          </select>
      </mapper>
      ```

- 用法2

  - ``<package name="com.lovefit.pojo"/>``

  - 在GirlMapper.xml中引用,其中resultType直接指定com.lovefit.pojo的类名即可（不区分大小写）。

    - ```xml
      <mapper namespace="com.lovefit.mapper.GirlMapper">
          <select id="queryByID" resultType="girl">
              select * from girl where id = #{id}
          </select>
      </mapper>
      ```

## 4 environments

environments是配置mybatis的运行环境基本信息的，可以配置多种环境。

- 需要的基本字段
  - url：连接字符串
  - driver：驱动
  - username：用户名
  - password：密码
- 配置文件可以直接配置基本字段的值也可以通过引用的方式来配置。建议使用引用的方式进行配置。

```xml
 <environment id="dev">
     <transactionManager type="JDBC"></transactionManager>
     <dataSource type="UNPOOLED">
         <!--<property name="url" value="jdbc:mysql://localhost:3306/mybatis?&amp;characterEncoding=utf8&amp;serverTimezone=GMT"/>-->
         <property name="url" value="${url}"/>
         <property name="driver" value="${driver}"/>
         <property name="username" value="${username}"/>
         <property name="password" value="${password}"/>
     </dataSource>
 </environment>
```

## 5 mappers

mappers是指定各个Pojo对应的mapper文件的位置。

- 配置方法1：

  - ```<mapper resource="com/lovefit/mapper/GirlMapper.xml"/>```
  - 配置的路径为resource下面的具体mapper.xml文件，有多少个文件就需要添加多少个配置。

- 配置方法2：

  - ```xml
    <mapper url="file:\\C:/Users/WYD181/Desktop/SSM_Study/mybatis002/src/main/resources/com/lovefit/mapper/GirlMapper.xml"/>
    ```

  - 配置路径为各个mapper文件的URL路径协议地址，不方便使用，因为项目在不同电脑的路径不一定相同。

- 配置方法3：

  - ``<mapper class="com.lovefit.mapper.GirlMapper"></mapper>``
  - 通过类的接口引入，必须保证``GirlMapper.java``路径和``GirlMapper.xml``路径是对应的。比如类路径是``src/main/java/com/lovefit/mapper/GirlMapper``那么要求GirlMapper.xml路径是``src/main/resource/com/lovefit/mapper/GirlMapper``

- 配置方法4：

  - ``<package name="com.lovefit.mapper"/>``
  - 配置方法3需要将所有配置文件手动配置，方法4是自动将方法三中的对应mapper文件夹配置一下就可以，这样子可以减少配置复杂度。

