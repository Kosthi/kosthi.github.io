+++
title = 'Build Doris on MacBook M1'
date = '2025-03-09T20:28:46+08:00'
draft = false
tags = ["Doris", "MacBook", "M1"]
categories = ["Build"]
author = "Koschei"
+++

## Install Environment Dependencies

```shell
brew install automake autoconf libtool pkg-config texinfo coreutils gnu-getopt \
python@3 cmake ninja ccache bison byacc gettext wget pcre maven llvm@16 openjdk@17 npm
```

*Doris master currently only supports JDK 17*

Environment variables that need to be set:

```shell
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH=$JAVA_HOME/bin:$PATH
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
export PATH="/opt/homebrew/opt/texinfo/bin:$PATH"
```

## Clone Your Code

1. Clone the repository

    ```shell
    cd ~
    mkdir DorisDev
    cd DorisDev
    git clone https://github.com/GitHubID/doris.git
    ```

2. Set environment variables
   
    ```shell
    export DORIS_HOME=~/DorisDev/doris
    export PATH=$DORIS_HOME/bin:$PATH
    ```

## Download Doris Build Dependencies

1. The [Apache Doris Third Party Prebuilt](https://github.com/apache/doris-thirdparty/releases/tag/automation) page has all third-party library sources, you can directly download [doris-thirdparty-source.tgz](https://github.com/apache/doris-thirdparty/releases/download/automation/doris-thirdparty-source.tgz).

2. You can directly download pre-built third-party libraries from the [Apache Doris Third Party Prebuilt](https://github.com/apache/doris-thirdparty/releases/tag/automation) page, saving the process of compiling third-party libraries. Refer to the commands below.
    
    ```shell
    cd thirdparty
    rm -rf installed

    # Intel chip
    curl -L https://github.com/apache/doris-thirdparty/releases/download/automation/doris-thirdparty-prebuilt-darwin-x86_64.tar.xz \
        -o - | tar -Jxf -

    # Apple Silicon chip
    curl -L https://github.com/apache/doris-thirdparty/releases/download/automation/doris-thirdparty-prebuilt-darwin-arm64.tar.xz \
        -o - | tar -Jxf -

    # Ensure protoc and thrift can run properly
    cd installed/bin

    ./protoc --version
    ./thrift --version
    ```
3. When running `protoc` and `thrift`, you may encounter the **"cannot be opened because the developer cannot be verified"** issue. You can go to `Security & Privacy`. Click the `Open Anyway` button in the `General` panel to confirm you want to open the binary. Refer to [https://support.apple.com/HT202491](https://support.apple.com/HT202491).

## Modify System Maximum File Handle Limit

```shell
# bash
echo 'ulimit -n 65536' >>~/.bashrc
    
# zsh
echo 'ulimit -n 65536' >>~/.zshrc
```

## Build Doris

```shell
cd $DORIS_HOME
sh build.sh
```

## You May Encounter Errors Caused by High Version Node.js During Build

Haven't encountered this on M1.
opensslErrorStack: ['error:03000086:digital envelope routines::initialization error']
library: 'digital envelope routines'
reason: 'unsupported'
code: 'ERR_OSSL_EVP_UNSUPPORTED'
The following command solves the problem. Refer to [https://stackoverflow.com/questions/74726224/opensslerrorstack-error03000086digital-envelope-routinesinitialization-e](https://stackoverflow.com/questions/74726224/opensslerrorstack-error03000086digital-envelope-routinesinitialization-e)

```shell
# Tell Node.js to use legacy OpenSSL provider
export NODE_OPTIONS=--openssl-legacy-provider
```

## You May Encounter jni.h Not Found During Build

Mac's JDK is in a deeper directory.

```shell
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH=$JAVA_HOME/bin:$PATH
```

## Configure Debug Environment

```shell
# Copy the compiled package
cp -r output ../doris-run

# Configure FE/BE conf, default settings are fine
1. IP, directory
2. BE additional configuration: min_file_descriptor_number = 10000
```

## Start Debugging with IDE

Refer to the official documentation:

[CLion Mac Debug BE](https://doris.apache.org/community/developer-guide/mac-dev/be-clion-dev)

[IntelliJ IDEA Mac Debug FE](https://doris.apache.org/community/developer-guide/mac-dev/fe-idea-dev)
