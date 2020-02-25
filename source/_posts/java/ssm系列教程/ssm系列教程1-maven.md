---
title: "ssm系列教程1-Maven基本使用"
date: 2019-05-01T18:42:04+08:00
draft: false
tags: ["maven"]
series: ["SSM框架系列笔记"]
categories: ["Java"]
toc: true
img: "/thumb/1.jpg"
summary: "介绍Maven的基本使用"

typora-copy-images-to: ssm系列教程1-maven
typora-root-url: ssm系列教程1-maven
---

# 1. Maven

## 1.1 什么是Maven

​		由Apache公司推出的一个管理我们项目的一个工具，ant演变过来的。是一个自动化构建工具。



## 1.2 pom

project object model，工程对象模型，pom.xml。

### 1.2.1 pom文件配置

```xml
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>com.lovefit</groupId>
  <artifactId>hello</artifactId>
  <version>1.0-SNAPSHOT</version>

  <name>hello</name>
  <!-- FIXME change it to the project's website -->
  <url>http://www.example.com</url>

  <properties>
    <!--属性的定义-->
    <!--项目构建使用的字符编码-->
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <!--编译源代码的版本号-->
    <maven.compiler.source>1.8</maven.compiler.source>
    <!--目标代码版本号-->
    <maven.compiler.target>1.8</maven.compiler.target>
  </properties>

  <!--依赖集合-->
  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.11</version>
      <scope>test</scope>
    </dependency>
  </dependencies>

  <build>
    <pluginManagement><!-- lock down plugins versions to avoid using Maven defaults (may be moved to parent pom) -->
      <plugins>
        <!-- clean lifecycle, see https://maven.apache.org/ref/current/maven-core/lifecycles.html#clean_Lifecycle -->
        <plugin>
          <artifactId>maven-clean-plugin</artifactId>
          <version>3.1.0</version>
        </plugin>
        <!-- default lifecycle, jar packaging: see https://maven.apache.org/ref/current/maven-core/default-bindings.html#Plugin_bindings_for_jar_packaging -->
        <plugin>
          <artifactId>maven-resources-plugin</artifactId>
          <version>3.0.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-compiler-plugin</artifactId>
          <version>3.8.0</version>
        </plugin>
        <plugin>
          <artifactId>maven-surefire-plugin</artifactId>
          <version>2.22.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-jar-plugin</artifactId>
          <version>3.0.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-install-plugin</artifactId>
          <version>2.5.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-deploy-plugin</artifactId>
          <version>2.8.2</version>
        </plugin>
        <!-- site lifecycle, see https://maven.apache.org/ref/current/maven-core/lifecycles.html#site_Lifecycle -->
        <plugin>
          <artifactId>maven-site-plugin</artifactId>
          <version>3.7.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-project-info-reports-plugin</artifactId>
          <version>3.0.0</version>
        </plugin>
      </plugins>
    </pluginManagement>
  </build>
</project>
```

## 2.1 安装与配置

### 2.1.1 配置setting

- 配置文件

  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  
  <!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
  -->
  
  <!--
   | This is the configuration file for Maven. It can be specified at two levels:
   |
   |  1. User Level. This settings.xml file provides configuration for a single user,
   |                 and is normally provided in ${user.home}/.m2/settings.xml.
   |
   |                 NOTE: This location can be overridden with the CLI option:
   |
   |                 -s /path/to/user/settings.xml
   |
   |  2. Global Level. This settings.xml file provides configuration for all Maven
   |                 users on a machine (assuming they're all using the same Maven
   |                 installation). It's normally provided in
   |                 ${maven.conf}/settings.xml.
   |
   |                 NOTE: This location can be overridden with the CLI option:
   |
   |                 -gs /path/to/global/settings.xml
   |
   | The sections in this sample file are intended to give you a running start at
   | getting the most out of your Maven installation. Where appropriate, the default
   | values (values used when the setting is not specified) are provided.
   |
   |-->
  <settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 http://maven.apache.org/xsd/settings-1.0.0.xsd">
    <!-- localRepository
     | The path to the local repository maven will use to store artifacts.
     |
     | Default: ${user.home}/.m2/repository
    <localRepository>/path/to/local/repo</localRepository>
    -->
  
    <!-- interactiveMode
     | This will determine whether maven prompts you when it needs input. If set to false,
     | maven will use a sensible default value, perhaps based on some other setting, for
     | the parameter in question.
     |
     | Default: true
    <interactiveMode>true</interactiveMode>
    -->
  
    <!-- offline
     | Determines whether maven should attempt to connect to the network when executing a build.
     | This will have an effect on artifact downloads, artifact deployment, and others.
     |
     | Default: false
    <offline>false</offline>
    -->
  
    <!-- pluginGroups
     | This is a list of additional group identifiers that will be searched when resolving plugins by their prefix, i.e.
     | when invoking a command line like "mvn prefix:goal". Maven will automatically add the group identifiers
     | "org.apache.maven.plugins" and "org.codehaus.mojo" if these are not already contained in the list.
     |-->
    <pluginGroups>
      <!-- pluginGroup
       | Specifies a further group identifier to use for plugin lookup.
      <pluginGroup>com.your.plugins</pluginGroup>
      -->
    </pluginGroups>
  
    <!-- proxies
     | This is a list of proxies which can be used on this machine to connect to the network.
     | Unless otherwise specified (by system property or command-line switch), the first proxy
     | specification in this list marked as active will be used.
     |-->
    <proxies>
      <!-- proxy
       | Specification for one proxy, to be used in connecting to the network.
       |
      <proxy>
        <id>optional</id>
        <active>true</active>
        <protocol>http</protocol>
        <username>proxyuser</username>
        <password>proxypass</password>
        <host>proxy.host.net</host>
        <port>80</port>
        <nonProxyHosts>local.net|some.host.com</nonProxyHosts>
      </proxy>
      -->
    </proxies>
  
    <!-- servers
     | This is a list of authentication profiles, keyed by the server-id used within the system.
     | Authentication profiles can be used whenever maven must make a connection to a remote server.
     |-->
    <servers>
      <!-- server
       | Specifies the authentication information to use when connecting to a particular server, identified by
       | a unique name within the system (referred to by the 'id' attribute below).
       |
       | NOTE: You should either specify username/password OR privateKey/passphrase, since these pairings are
       |       used together.
       |
      <server>
        <id>deploymentRepo</id>
        <username>repouser</username>
        <password>repopwd</password>
      </server>
      -->
  
      <!-- Another sample, using keys to authenticate.
      <server>
        <id>siteServer</id>
        <privateKey>/path/to/private/key</privateKey>
        <passphrase>optional; leave empty if not used.</passphrase>
      </server>
      -->
    </servers>
  
    <!-- mirrors
     | This is a list of mirrors to be used in downloading artifacts from remote repositories.
     |
     | It works like this: a POM may declare a repository to use in resolving certain artifacts.
     | However, this repository may have problems with heavy traffic at times, so people have mirrored
     | it to several places.
     |
     | That repository definition will have a unique id, so we can create a mirror reference for that
     | repository, to be used as an alternate download site. The mirror site will be the preferred
     | server for that repository.
     |-->
    <mirrors>
      <!-- mirror
       | Specifies a repository mirror site to use instead of a given repository. The repository that
       | this mirror serves has an ID that matches the mirrorOf element of this mirror. IDs are used
       | for inheritance and direct lookup purposes, and must be unique across the set of mirrors.
       |
      <mirror>
        <id>mirrorId</id>
        <mirrorOf>repositoryId</mirrorOf>
        <name>Human Readable Name for this Mirror.</name>
        <url>http://my.repository.com/repo/path</url>
      </mirror>
       -->
    </mirrors>
  
    <!-- profiles
     | This is a list of profiles which can be activated in a variety of ways, and which can modify
     | the build process. Profiles provided in the settings.xml are intended to provide local machine-
     | specific paths and repository locations which allow the build to work in the local environment.
     |
     | For example, if you have an integration testing plugin - like cactus - that needs to know where
     | your Tomcat instance is installed, you can provide a variable here such that the variable is
     | dereferenced during the build process to configure the cactus plugin.
     |
     | As noted above, profiles can be activated in a variety of ways. One way - the activeProfiles
     | section of this document (settings.xml) - will be discussed later. Another way essentially
     | relies on the detection of a system property, either matching a particular value for the property,
     | or merely testing its existence. Profiles can also be activated by JDK version prefix, where a
     | value of '1.4' might activate a profile when the build is executed on a JDK version of '1.4.2_07'.
     | Finally, the list of active profiles can be specified directly from the command line.
     |
     | NOTE: For profiles defined in the settings.xml, you are restricted to specifying only artifact
     |       repositories, plugin repositories, and free-form properties to be used as configuration
     |       variables for plugins in the POM.
     |
     |-->
    <profiles>
      <!-- profile
       | Specifies a set of introductions to the build process, to be activated using one or more of the
       | mechanisms described above. For inheritance purposes, and to activate profiles via <activatedProfiles/>
       | or the command line, profiles have to have an ID that is unique.
       |
       | An encouraged best practice for profile identification is to use a consistent naming convention
       | for profiles, such as 'env-dev', 'env-test', 'env-production', 'user-jdcasey', 'user-brett', etc.
       | This will make it more intuitive to understand what the set of introduced profiles is attempting
       | to accomplish, particularly when you only have a list of profile id's for debug.
       |
       | This profile example uses the JDK version to trigger activation, and provides a JDK-specific repo.
      <profile>
        <id>jdk-1.4</id>
  
        <activation>
          <jdk>1.4</jdk>
        </activation>
  
        <repositories>
          <repository>
            <id>jdk14</id>
            <name>Repository for JDK 1.4 builds</name>
            <url>http://www.myhost.com/maven/jdk14</url>
            <layout>default</layout>
            <snapshotPolicy>always</snapshotPolicy>
          </repository>
        </repositories>
      </profile>
      -->
  
      <!--
       | Here is another profile, activated by the system property 'target-env' with a value of 'dev',
       | which provides a specific path to the Tomcat instance. To use this, your plugin configuration
       | might hypothetically look like:
       |
       | ...
       | <plugin>
       |   <groupId>org.myco.myplugins</groupId>
       |   <artifactId>myplugin</artifactId>
       |
       |   <configuration>
       |     <tomcatLocation>${tomcatPath}</tomcatLocation>
       |   </configuration>
       | </plugin>
       | ...
       |
       | NOTE: If you just wanted to inject this configuration whenever someone set 'target-env' to
       |       anything, you could just leave off the <value/> inside the activation-property.
       |
      <profile>
        <id>env-dev</id>
  
        <activation>
          <property>
            <name>target-env</name>
            <value>dev</value>
          </property>
        </activation>
  
        <properties>
          <tomcatPath>/path/to/tomcat/instance</tomcatPath>
        </properties>
      </profile>
      -->
    </profiles>
  
    <!-- activeProfiles
     | List of profiles that are active for all builds.
     |
    <activeProfiles>
      <activeProfile>alwaysActiveProfile</activeProfile>
      <activeProfile>anotherAlwaysActiveProfile</activeProfile>
    </activeProfiles>
    -->
  </settings>
  
  ```

- 手动安装Maven需要配置setting的jar包存放位置。

  ``<localRepository>D:/reposi</localRepository>``

- 配置中央环境，换言之更换源地址。

  ```xml
  <mirror>
        <id>nexus-aliyun</id>
        <mirrorOf>*</mirrorOf>
        <name>Nexus aliyun</name>
        <url>http://maven.aliyun.com/nexus/content/groups/public</url>
  </mirror>
  ```

## 3.1 IDEA新建项目

### 3.1.1 IDEA的配置

- 配置setting的Maven路径与配置文件路径

  <img src="./1567820129249.png" alt="1567820129249" style="zoom:50%;" />

  - 实际maven新建项目时候 会确认maven的版本

    <img src="./1567821066994.png" alt="1567821066994" style="zoom: 50%;" />

- IDEA 常用的骨架

  - ``org.apache.maven.archetypes:maven-archtype-site-quickstart``
  - ``org.apache.maven.archetypes:maven-archtype-site-webapp``

### 3.1.2 maven构建

``` java
C:\MySoftWare\Java\jdk1.8.0_144\bin\java.exe -Dmaven.multiModuleProjectDirectory=C:\Users\WYD181\AppData\Local\Temp\archetypetmp "-Dmaven.home=C:\ad\JetBrains\IntelliJ IDEA 2019.2.1\plugins\maven\lib\maven3" "-Dclassworlds.conf=C:\ad\JetBrains\IntelliJ IDEA 2019.2.1\plugins\maven\lib\maven3\bin\m2.conf" "-Dmaven.ext.class.path=C:\ad\JetBrains\IntelliJ IDEA 2019.2.1\plugins\maven\lib\maven-event-listener.jar" -Dfile.encoding=UTF-8 -classpath "C:\ad\JetBrains\IntelliJ IDEA 2019.2.1\plugins\maven\lib\maven3\boot\plexus-classworlds-2.6.0.jar" org.codehaus.classworlds.Launcher -Didea.version2019.2.1 -DinteractiveMode=false -DgroupId=com.lovefit -DartifactId=hello -Dversion=1.0-SNAPSHOT -DarchetypeGroupId=org.apache.maven.archetypes -DarchetypeArtifactId=maven-archetype-quickstart -DarchetypeVersion=RELEASE org.apache.maven.plugins:maven-archetype-plugin:RELEASE:generate
[INFO] Scanning for projects...
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/plugins/maven-archetype-plugin/maven-metadata.xml
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/plugins/maven-archetype-plugin/maven-metadata.xml (918 B at 392 B/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/plugins/maven-archetype-plugin/3.1.2/maven-archetype-plugin-3.1.2.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/plugins/maven-archetype-plugin/3.1.2/maven-archetype-plugin-3.1.2.pom (11 kB at 26 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/maven-archetype/3.1.2/maven-archetype-3.1.2.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/maven-archetype/3.1.2/maven-archetype-3.1.2.pom (12 kB at 29 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/maven-parent/33/maven-parent-33.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/maven-parent/33/maven-parent-33.pom (44 kB at 72 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/apache/21/apache-21.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/apache/21/apache-21.pom (17 kB at 43 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/plugins/maven-archetype-plugin/3.1.2/maven-archetype-plugin-3.1.2.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/plugins/maven-archetype-plugin/3.1.2/maven-archetype-plugin-3.1.2.jar (97 kB at 48 kB/s)
[INFO] 
[INFO] ------------------< org.apache.maven:standalone-pom >-------------------
[INFO] Building Maven Stub Project (No POM) 1
[INFO] --------------------------------[ pom ]---------------------------------
[INFO] 
[INFO] >>> maven-archetype-plugin:3.1.2:generate (default-cli) > generate-sources @ standalone-pom >>>
[INFO] 
[INFO] <<< maven-archetype-plugin:3.1.2:generate (default-cli) < generate-sources @ standalone-pom <<<
[INFO] 
[INFO] 
[INFO] --- maven-archetype-plugin:3.1.2:generate (default-cli) @ standalone-pom ---
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-catalog/3.1.2/archetype-catalog-3.1.2.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-catalog/3.1.2/archetype-catalog-3.1.2.pom (2.0 kB at 1.6 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-models/3.1.2/archetype-models-3.1.2.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-models/3.1.2/archetype-models-3.1.2.pom (2.8 kB at 2.7 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-utils/3.2.0/plexus-utils-3.2.0.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-utils/3.2.0/plexus-utils-3.2.0.pom (4.8 kB at 8.1 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus/5.1/plexus-5.1.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus/5.1/plexus-5.1.pom (23 kB at 19 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-descriptor/3.1.2/archetype-descriptor-3.1.2.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-descriptor/3.1.2/archetype-descriptor-3.1.2.pom (2.0 kB at 4.8 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-common/3.1.2/archetype-common-3.1.2.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-common/3.1.2/archetype-common-3.1.2.pom (18 kB at 22 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/groovy/groovy/2.4.16/groovy-2.4.16.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/groovy/groovy/2.4.16/groovy-2.4.16.pom (19 kB at 24 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/ivy/ivy/2.4.0/ivy-2.4.0.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/ivy/ivy/2.4.0/ivy-2.4.0.pom (6.3 kB at 7.9 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/net/sourceforge/jchardet/jchardet/1.0/jchardet-1.0.pom
Downloaded from central: https://repo.maven.apache.org/maven2/net/sourceforge/jchardet/jchardet/1.0/jchardet-1.0.pom (1.3 kB at 3.2 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-component-annotations/1.7.1/plexus-component-annotations-1.7.1.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-component-annotations/1.7.1/plexus-component-annotations-1.7.1.pom (770 B at 2.0 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-containers/1.7.1/plexus-containers-1.7.1.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-containers/1.7.1/plexus-containers-1.7.1.pom (5.0 kB at 8.1 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-invoker/3.0.1/maven-invoker-3.0.1.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-invoker/3.0.1/maven-invoker-3.0.1.pom (4.9 kB at 12 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-components/31/maven-shared-components-31.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-components/31/maven-shared-components-31.pom (5.1 kB at 12 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/maven-parent/31/maven-parent-31.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/maven-parent/31/maven-parent-31.pom (43 kB at 32 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/apache/19/apache-19.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/apache/19/apache-19.pom (15 kB at 27 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-utils/3.2.1/maven-shared-utils-3.2.1.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-utils/3.2.1/maven-shared-utils-3.2.1.pom (5.6 kB at 14 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-components/30/maven-shared-components-30.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-components/30/maven-shared-components-30.pom (4.6 kB at 5.7 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/maven-parent/30/maven-parent-30.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/maven-parent/30/maven-parent-30.pom (41 kB at 18 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-artifact-transfer/0.11.0/maven-artifact-transfer-0.11.0.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-artifact-transfer/0.11.0/maven-artifact-transfer-0.11.0.pom (11 kB at 11 kB/s)
[IJ]-1-ARTIFACT_DOWNLOADING-[IJ]-path=-[IJ]-artifactCoord=org.apache.maven.shared:maven-shared-components:pom:33-[IJ]-error=
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-components/33/maven-shared-components-33.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-components/33/maven-shared-components-33.pom (5.1 kB at 9.1 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-common-artifact-filters/3.0.1/maven-common-artifact-filters-3.0.1.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-common-artifact-filters/3.0.1/maven-common-artifact-filters-3.0.1.pom (4.8 kB at 12 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-utils/3.1.0/maven-shared-utils-3.1.0.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-utils/3.1.0/maven-shared-utils-3.1.0.pom (5.0 kB at 12 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/commons-codec/commons-codec/1.11/commons-codec-1.11.pom
Downloaded from central: https://repo.maven.apache.org/maven2/commons-codec/commons-codec/1.11/commons-codec-1.11.pom (14 kB at 25 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/commons/commons-parent/42/commons-parent-42.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/commons/commons-parent/42/commons-parent-42.pom (68 kB at 20 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/slf4j/slf4j-api/1.7.5/slf4j-api-1.7.5.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/slf4j/slf4j-api/1.7.5/slf4j-api-1.7.5.pom (2.7 kB at 6.4 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/slf4j/slf4j-parent/1.7.5/slf4j-parent-1.7.5.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/slf4j/slf4j-parent/1.7.5/slf4j-parent-1.7.5.pom (12 kB at 21 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/wagon/wagon-provider-api/3.3.3/wagon-provider-api-3.3.3.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/wagon/wagon-provider-api/3.3.3/wagon-provider-api-3.3.3.pom (1.9 kB at 5.0 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/wagon/wagon/3.3.3/wagon-3.3.3.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/wagon/wagon/3.3.3/wagon-3.3.3.pom (21 kB at 27 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-interactivity-api/1.0-alpha-6/plexus-interactivity-api-1.0-alpha-6.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-interactivity-api/1.0-alpha-6/plexus-interactivity-api-1.0-alpha-6.pom (726 B at 1.9 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-interactivity/1.0-alpha-6/plexus-interactivity-1.0-alpha-6.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-interactivity/1.0-alpha-6/plexus-interactivity-1.0-alpha-6.pom (1.1 kB at 2.6 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-components/1.1.9/plexus-components-1.1.9.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-components/1.1.9/plexus-components-1.1.9.pom (2.4 kB at 6.1 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-script-interpreter/1.0/maven-script-interpreter-1.0.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-script-interpreter/1.0/maven-script-interpreter-1.0.pom (3.8 kB at 9.1 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/groovy/groovy/1.8.3/groovy-1.8.3.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/groovy/groovy/1.8.3/groovy-1.8.3.pom (32 kB at 40 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/antlr/antlr/2.7.7/antlr-2.7.7.pom
Downloaded from central: https://repo.maven.apache.org/maven2/antlr/antlr/2.7.7/antlr-2.7.7.pom (632 B at 1.6 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/ant/ant/1.8.1/ant-1.8.1.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/ant/ant/1.8.1/ant-1.8.1.pom (8.8 kB at 11 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/ant/ant-parent/1.8.1/ant-parent-1.8.1.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/ant/ant-parent/1.8.1/ant-parent-1.8.1.pom (4.3 kB at 10.0 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-descriptor/3.1.2/archetype-descriptor-3.1.2.jar
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/ivy/ivy/2.4.0/ivy-2.4.0.jar
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/groovy/groovy/2.4.16/groovy-2.4.16.jar
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-catalog/3.1.2/archetype-catalog-3.1.2.jar
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-common/3.1.2/archetype-common-3.1.2.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-catalog/3.1.2/archetype-catalog-3.1.2.jar (19 kB at 20 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/net/sourceforge/jchardet/jchardet/1.0/jchardet-1.0.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-descriptor/3.1.2/archetype-descriptor-3.1.2.jar (24 kB at 23 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-component-annotations/1.7.1/plexus-component-annotations-1.7.1.jar
Downloaded from central: https://repo.maven.apache.org/maven2/net/sourceforge/jchardet/jchardet/1.0/jchardet-1.0.jar (27 kB at 20 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/wagon/wagon-provider-api/3.3.3/wagon-provider-api-3.3.3.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-component-annotations/1.7.1/plexus-component-annotations-1.7.1.jar (4.3 kB at 3.0 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-utils/3.2.0/plexus-utils-3.2.0.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetype/archetype-common/3.1.2/archetype-common-3.1.2.jar (185 kB at 127 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-interactivity-api/1.0-alpha-6/plexus-interactivity-api-1.0-alpha-6.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-interactivity-api/1.0-alpha-6/plexus-interactivity-api-1.0-alpha-6.jar (12 kB at 6.5 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-invoker/3.0.1/maven-invoker-3.0.1.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/wagon/wagon-provider-api/3.3.3/wagon-provider-api-3.3.3.jar (56 kB at 30 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-utils/3.2.1/maven-shared-utils-3.2.1.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-invoker/3.0.1/maven-invoker-3.0.1.jar (33 kB at 15 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-artifact-transfer/0.11.0/maven-artifact-transfer-0.11.0.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-artifact-transfer/0.11.0/maven-artifact-transfer-0.11.0.jar (128 kB at 46 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-common-artifact-filters/3.0.1/maven-common-artifact-filters-3.0.1.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-shared-utils/3.2.1/maven-shared-utils-3.2.1.jar (167 kB at 58 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/commons-codec/commons-codec/1.11/commons-codec-1.11.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-common-artifact-filters/3.0.1/maven-common-artifact-filters-3.0.1.jar (61 kB at 18 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/slf4j/slf4j-api/1.7.5/slf4j-api-1.7.5.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/slf4j/slf4j-api/1.7.5/slf4j-api-1.7.5.jar (26 kB at 7.0 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-script-interpreter/1.0/maven-script-interpreter-1.0.jar
Downloaded from central: https://repo.maven.apache.org/maven2/commons-codec/commons-codec/1.11/commons-codec-1.11.jar (335 kB at 85 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/ant/ant/1.8.1/ant-1.8.1.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/shared/maven-script-interpreter/1.0/maven-script-interpreter-1.0.jar (21 kB at 5.0 kB/s)
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/plexus/plexus-utils/3.2.0/plexus-utils-3.2.0.jar (263 kB at 57 kB/s)
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/ant/ant/1.8.1/ant-1.8.1.jar (1.5 MB at 102 kB/s)
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/ivy/ivy/2.4.0/ivy-2.4.0.jar (1.3 MB at 62 kB/s)
Downloaded from central: https://repo.maven.apache.org/maven2/org/codehaus/groovy/groovy/2.4.16/groovy-2.4.16.jar (4.7 MB at 111 kB/s)
[INFO] Generating project in Batch mode
[INFO] Archetype repository not defined. Using the one from [org.apache.maven.archetypes:maven-archetype-quickstart:1.4] found in catalog remote
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetypes/maven-archetype-quickstart/maven-metadata.xml
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetypes/maven-archetype-quickstart/maven-metadata.xml (589 B at 1.5 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetypes/maven-archetype-quickstart/1.4/maven-archetype-quickstart-1.4.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetypes/maven-archetype-quickstart/1.4/maven-archetype-quickstart-1.4.pom (1.6 kB at 4.2 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetypes/maven-archetype-bundles/1.4/maven-archetype-bundles-1.4.pom
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetypes/maven-archetype-bundles/1.4/maven-archetype-bundles-1.4.pom (4.5 kB at 12 kB/s)
Downloading from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetypes/maven-archetype-quickstart/1.4/maven-archetype-quickstart-1.4.jar
Downloaded from central: https://repo.maven.apache.org/maven2/org/apache/maven/archetypes/maven-archetype-quickstart/1.4/maven-archetype-quickstart-1.4.jar (7.1 kB at 19 kB/s)
[INFO] ----------------------------------------------------------------------------
[INFO] Using following parameters for creating project from Archetype: maven-archetype-quickstart:RELEASE
[INFO] ----------------------------------------------------------------------------
[INFO] Parameter: groupId, Value: com.lovefit
[INFO] Parameter: artifactId, Value: hello
[INFO] Parameter: version, Value: 1.0-SNAPSHOT
[INFO] Parameter: package, Value: com.lovefit
[INFO] Parameter: packageInPathFormat, Value: com/lovefit
[INFO] Parameter: package, Value: com.lovefit
[INFO] Parameter: version, Value: 1.0-SNAPSHOT
[INFO] Parameter: groupId, Value: com.lovefit
[INFO] Parameter: artifactId, Value: hello
[INFO] Project created from Archetype in dir: C:\Users\WYD181\AppData\Local\Temp\archetypetmp\hello
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  02:53 min
[INFO] Finished at: 2019-09-07T09:57:12+08:00
[INFO] ------------------------------------------------------------------------

```

- 构建完成的目录

  ![1567821522189](./1567821522189.png)



