# (以wubuntu22.04 & python=3.10 & CUDA=11.8为例)：
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# 安装git
RUN apt-get update
# 设置工作目录（容器内的指定目录）
WORKDIR /app

### 安装python 3.10.8 和 pip
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    ca-certificates \
    && curl -O https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz \
    && tar -xzf Python-3.10.8.tgz \
    && cd Python-3.10.8 \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.10.8.tgz Python-3.10.8 \
    && ln -s /usr/local/bin/python3.10 /usr/local/bin/python \
    && ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -i https://mirrors.aliyun.com/pypi/simple/\
    && apt-get update && apt-get install -y libgl1-mesa-glx

# 克隆特定版本的代码
COPY . /app

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    




