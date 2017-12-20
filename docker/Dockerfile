FROM alpine:3.4
MAINTAINER Zhou Rui <zhourui@getqujing.com>

# use aliyun mirrors, thx to *GFW*
ENV PIP_CONFIG_FILE "/root/.pip/pip.conf"
RUN echo "http://mirrors.aliyun.com/alpine/v3.4/main" > /etc/apk/repositories \
    && echo "http://mirrors.aliyun.com/alpine/v3.4/community" >> /etc/apk/repositories \
    && echo "@edge http://mirrors.aliyun.com/alpine/edge/community" >> /etc/apk/repositories \
    && echo "@edge http://mirrors.aliyun.com/alpine/edge/testing" >> /etc/apk/repositories
RUN mkdir /root/.pip \
    && echo "[global]" > $PIP_CONFIG_FILE \
    && echo "index-url = http://mirrors.aliyun.com/pypi/simple" >> $PIP_CONFIG_FILE \
    && echo "trusted-host = mirrors.aliyun.com" >> $PIP_CONFIG_FILE
RUN apk update

ENV PYTHON_VERSION 3.5.2-r2
# python 3
RUN apk add python3="$PYTHON_VERSION" \
    && python3 -m ensurepip \
    && rm -r /usr/lib/python*/ensurepip \
    && pip3 install --upgrade pip setuptools

# Java environment from _/openjdk:alpine
# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

# add a simple script that can auto-detect the appropriate JAVA_HOME value
# based on whether the JDK or only the JRE is installed
RUN { \
        echo '#!/bin/sh'; \
        echo 'set -e'; \
        echo; \
        echo 'dirname "$(dirname "$(readlink -f "$(which javac || which java)")")"'; \
    } > /usr/local/bin/docker-java-home \
    && chmod +x /usr/local/bin/docker-java-home

ENV JAVA_HOME /usr/lib/jvm/java-1.8-openjdk/jre
ENV PATH $PATH:/usr/lib/jvm/java-1.8-openjdk/jre/bin:/usr/lib/jvm/java-1.8-openjdk/bin
ENV JAVA_VERSION 8u111
ENV JAVA_ALPINE_VERSION 8.111.14-r0

RUN set -x \
    && apk add openjdk8-jre="$JAVA_ALPINE_VERSION" \
    && [ "$JAVA_HOME" = "$(docker-java-home)" ]

RUN apk add libstdc++ libgfortran openblas-dev@edge rocksdb-dev@edge
RUN apk add --virtual build-deps g++ gfortran make musl-dev linux-headers python3-dev
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN pip install --no-cache-dir configobj==5.0.6
RUN pip install --no-cache-dir numpy==1.12.0
RUN pip install --no-cache-dir scipy==0.18.1
RUN pip install --no-cache-dir gensim==0.13.3
RUN pip install --no-cache-dir imbalanced-learn==0.1.8
RUN pip install --no-cache-dir nltk==3.2.1
RUN pip install --no-cache-dir pynlpir==0.4.6
RUN pip install --no-cache-dir regex==2017.01.17
RUN pip install --no-cache-dir aiohttp==1.2.0

RUN apk del --purge build-deps

# fuck Dr. Zhang!
COPY ./NLPIR.user /usr/lib/python3.5/site-packages/pynlpir/Data

# clear cache
RUN rm /var/cache/apk/*
