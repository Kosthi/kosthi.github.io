+++
title = 'Build Doris on MacBook M1'
date = '2025-03-09T20:28:46+08:00'
draft = true
tags = ["Doris", "MacBook", "M1"]
categories = ["Build"]
author = "Koschei"
+++

## 安装环境依赖

```shell
brew install automake autoconf libtool pkg-config texinfo coreutils gnu-getopt \
python@3 cmake ninja ccache bison byacc gettext wget pcre maven llvm@16 openjdk@17 npm
```

*Doris master 目前只支持 jdk17 版本*

需要设置的环境变量

```shell
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH=$JAVA_HOME/bin:$PATH
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
export PATH="/opt/homebrew/opt/texinfo/bin:$PATH"
```

## 拉取自己的代码

1. 拉取代码

    ```shell
    cd ~
    mkdir DorisDev
    cd DorisDev
    git clone https://github.com/GitHubID/doris.git
    ```

2. 设置环境变量
   
    ```shell
    export DORIS_HOME=~/DorisDev/doris
    export PATH=$DORIS_HOME/bin:$PATH
    ```

## 下载 Doris 编译依赖

1. [Apache Doris Third Party Prebuilt](https://github.com/apache/doris-thirdparty/releases/tag/automation)页面有所有第三方库的源码，可以直接下载[doris-thirdparty-source.tgz](https://github.com/apache/doris-thirdparty/releases/download/automation/doris-thirdparty-source.tgz)获得。

2. 可以在[Apache Doris Third Party Prebuilt](https://github.com/apache/doris-thirdparty/releases/tag/automation)页面直接下载预编译好的第三方库，省去编译第三方库的过程，参考下面的命令。
    
    ```shell
    cd thirdparty
    rm -rf installed

    # Intel 芯片
    curl -L https://github.com/apache/doris-thirdparty/releases/download/automation/doris-thirdparty-prebuilt-darwin-x86_64.tar.xz \
        -o - | tar -Jxf -

    # Apple Silicon 芯片
    curl -L https://github.com/apache/doris-thirdparty/releases/download/automation/doris-thirdparty-prebuilt-darwin-arm64.tar.xz \
        -o - | tar -Jxf -

    # 保证 protoc 和 thrift 能够正常运行
    cd installed/bin

    ./protoc --version
    ./thrift --version
    ```
3. 运行`protoc`和`thrift`的时候可能会遇到**无法打开，因为无法验证开发者**的问题，可以到前往`安全性与隐私`。点按`通用`面板中的`仍要打开`按钮，以确认打算打开该二进制。参考[https://support.apple.com/zh-cn/HT202491](https://support.apple.com/zh-cn/HT202491)。

## 修改系统最大文件句柄数

```shell
# bash
echo 'ulimit -n 65536' >>~/.bashrc
    
# zsh
echo 'ulimit -n 65536' >>~/.zshrc
```

## 编译 Doris

```shell
cd $DORIS_HOME
sh build.sh
```

## 编译过程中可能会遇到高版本的 Node.js 导致的错误

M1 目前没遇到过。
opensslErrorStack: ['error:03000086:digital envelope routines::initialization error']
library: 'digital envelope routines'
reason: 'unsupported'
code: 'ERR_OSSL_EVP_UNSUPPORTED'
以下命令解决问题。参考[https://stackoverflow.com/questions/74726224/opensslerrorstack-error03000086digital-envelope-routinesinitialization-e](https://stackoverflow.com/questions/74726224/opensslerrorstack-error03000086digital-envelope-routinesinitialization-e)

```shell
#指示Node.js使用旧版的OpenSSL提供程序
export NODE_OPTIONS=--openssl-legacy-provider
```

## 编译过程中可能会遇到 jni.h not found

Mac 的 JDK 在更深一层目录。

```shell
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH=$JAVA_HOME/bin:$PATH
```

## 配置 Debug 环境

```shell
# 将编译好的包cp出来
cp -r output ../doris-run

# 配置FE/BE的conf，用默认的也行吧
1、IP、目录
2、BE 额外配置 min_file_descriptor_number = 10000
```

## 开始用 IDE 进行 Debug

参考官方OK

[CLion Mac 调试 BE](https://doris.apache.org/zh-CN/community/developer-guide/mac-dev/be-clion-dev)

[IntelliJ IDEA Mac 调试 FE](https://doris.apache.org/zh-CN/community/developer-guide/mac-dev/fe-idea-dev)
