# 使用私有 Isaac Sim 镜像作为基础镜像
FROM 192.168.5.54:5000/x86/nvidia/isaac-sim:5.1.0

# 设置环境变量避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color
ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACSIM_PYTHON_EXE=/isaac-sim/python.sh

# 切换到 root 用户确保有权限
USER root

# 配置华为 apt 源（国内加速，x86 架构）
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.huaweicloud.com|g' /etc/apt/sources.list.d/ubuntu.sources || \
    sed -i 's|http://security.ubuntu.com|https://mirrors.huaweicloud.com|g' /etc/apt/sources.list.d/ubuntu.sources || \
    (echo "deb https://mirrors.huaweicloud.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
     echo "deb https://mirrors.huaweicloud.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
     echo "deb https://mirrors.huaweicloud.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list)

# 配置阿里云 pip 源（国内加速）
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf && \
    echo "trusted-host = mirrors.aliyun.com" >> /root/.pip/pip.conf

# 更新 apt 并安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    ca-certificates \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 克隆 IsaacLab 仓库
RUN git clone https://github.com/isaac-sim/IsaacLab.git /workspace/IsaacLab

# 创建 Isaac Sim 符号链接
WORKDIR /workspace/IsaacLab
RUN ln -s ${ISAACSIM_PATH} _isaac_sim

# 安装 IsaacLab 及其扩展
RUN chmod +x /workspace/IsaacLab/isaaclab.sh && \
    ./isaaclab.sh --install

# 配置 Python 环境
RUN ${ISAACSIM_PYTHON_EXE} -m pip install --upgrade pip

# 设置环境变量
ENV PYTHONPATH=/workspace/IsaacLab
ENV PATH=/workspace/IsaacLab/scripts:${PATH}

# 清理缓存以减小镜像体积
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    ${ISAACSIM_PYTHON_EXE} -m pip cache purge

# 设置默认工作目录
WORKDIR /workspace/IsaacLab

# 验证安装（构建时注释掉，运行时可以验证）
# RUN ${ISAACSIM_PYTHON_EXE} scripts/tutorials/00_sim/create_empty.py --help

# 暴露端口（如果需要）
# EXPOSE 8888

# 设置容器启动命令
CMD ["/bin/bash"]
