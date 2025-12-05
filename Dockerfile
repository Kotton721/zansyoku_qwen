FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG PYVER=3.10.15

# 基本ツール & ライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates git vim \
    build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev libgdbm-dev libnss3-dev libffi-dev liblzma-dev tk-dev \
    libxml2-dev libxslt1-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    locales tzdata \
 && rm -rf /var/lib/apt/lists/*

# ロケール/タイムゾーン
RUN locale-gen ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1

# ==== Python 3.10.15 ソースビルド ====
RUN wget -q https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tar.xz \
 && tar xf Python-${PYVER}.tar.xz \
 && cd Python-${PYVER} \
 && ./configure --enable-optimizations --enable-loadable-sqlite-extensions \
 && make -j"$(nproc)" \
 && make install \
 && cd / && rm -rf Python-${PYVER} Python-${PYVER}.tar.xz \
 && ln -s /usr/local/bin/python3 /usr/local/bin/python \
 && python -V

# ==== pip 導入 ====
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python /tmp/get-pip.py \
 && pip --version \
 && rm /tmp/get-pip.py

# ==== Python パッケージ ====
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

# ルートでpip実行の警告を抑えたい場合は --root-user-action=ignore を付けてもOK
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

WORKDIR /workspace
CMD ["/bin/bash"]