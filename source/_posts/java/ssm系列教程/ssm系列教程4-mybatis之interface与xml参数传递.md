---
title: "ssm系列教程4-Mybatis之Interface与Xml参数传递"
date: 2019-05-01T22:42:04+08:00
draft: false
tags: ["mybatis"]
series: ["SSM框架系列笔记"]
categories: ["Java"]
toc: true
img: "/thumb/2.jpg"
summary: "介绍MapperClass的接口与MapperXml中sql的参数传递"
typora-copy-images-to: ssm系列教程4-mybatis之interface与xml参数传递
typora-root-url: ssm系列教程4-mybatis之interface与xml参数传递
---

# 1 单复杂数据类型

## 1.1 mapper-java

``int insert(Girl g);`` 传参形式单个复杂类型

## 1.2 mapper-xml

```xml
<insert id="insert">
    insert into girl (name,flower,birthday) values (#{name},#{flower},#{birthday})
</insert>
```

> 其中name，flower，birthday都是Girl的属性名称。

## 1.3 测试

```java
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
```



# 2 单基本数据类型

## 2.1 mapper-java

``Girl queryByID(int id);`` 传参形式，单个基本类型

## 2.2 mapper-xml

```xml
<select id="queryByID" resultType="Girl">   
    select * from girl where id = #{id} # id可以为任意
</select>
```

## 2.3 测试

```java
@Test
public void m2()
{
    SqlSession sqlSession = MybatisUtil.getSession();

    GirlMapper mapper = sqlSession.getMapper(GirlMapper.class);

    Girl girl = mapper.queryByID(4);

    assert girl.getName().equals("林心如");
    
    sqlSession.close();
}
```

# 3 多基本数据类型

## 3.1 mapper-java

```
Girl queryByNameFlower(String name,String flower);
```

## 3.2 mapper-xml

```xml
<select id="queryByNameFlower" resultType="com.lovefit.pojo.Girl">
    <!--select * from girl where name = #{param1} and flower = #{param2}-->
    select * from girl where name = #{arg0} and flower = #{arg1}
</select>
```

参数无论使用arg0、arg1还是param1、param2代码可读性都是很差的。

## 3.3 使用注解配置参数名

````java
 Girl queryByNameFlower(@Param("name") String name,@Param("flower") String flower);
````

```xml
<select id="queryByNameFlower" resultType="com.lovefit.pojo.Girl">
	select * from girl where name = #{name} and flower = #{flower}
</select>
```

## 3.4 测试

```java
@Test
public void m4()
{
    SqlSession sqlSession = MybatisUtil.getSession();

    GirlMapper mapper = sqlSession.getMapper(GirlMapper.class);

    Girl girl = mapper.queryByNameFlower("林心如","茉莉花");

    assert girl.getName().equals("林心如");
    sqlSession.close();
}
```

# 4 单Map数据类型

## 4.1 mapper-java

```java
List<Girl> queryByNameFlower1(Map<String,Object> map);
```

## 4.2 mapper-xml

```xml
<select id="queryByNameFlower1" resultType="com.lovefit.pojo.Girl">
    select * from girl where name = #{name} and flower = #{flower}
</select>
```

## 4.3 测试

```java
@Test
public void m5()
{
    SqlSession sqlSession = MybatisUtil.getSession();

    GirlMapper mapper = sqlSession.getMapper(GirlMapper.class);

    Girl girl = new Girl();
    Map<String,Object> map = new HashMap<>();
    map.put("name","林心如");
    map.put("flower","霍建花");

    List<Girl> girl2 = mapper.queryByNameFlower1(map);
    assert girl2.getName().equals("林心如");
    sqlSession.close();
}
```

# 5多复杂数据类型

## 5.1 mapper-java

```
List<Girl> queryByAB(@Param("a") A a,@Param("b") B b);
```

```java
public class A {
    private String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
```

```java
public class B {
    private String flower;

    public String getFlower() {
        return flower;
    }

    public void setFlower(String flower) {
        this.flower = flower;
    }
}
```

## 5.2 mapper-xml

```
<select id="queryByAB" resultType="com.lovefit.pojo.Girl">
    #select * from girl where name = #{param1.name} and flower = #{param2.flower}
    select * from girl where name = #{a.name} and flower = #{b.flower}
</select>
```

## 5.3 测试

```java
@Test
public void m6()
{
    SqlSession sqlSession = MybatisUtil.getSession();

    GirlMapper mapper = sqlSession.getMapper(GirlMapper.class);

    A a = new A();
    a.setName("林心如");

    B b = new B();
    b.setFlower("霍建花");

    List<Girl> girl = mapper.queryByAB(a,b);
    assert girl.get(0).getName().equals("林心如");
    sqlSession.close();
}
```

