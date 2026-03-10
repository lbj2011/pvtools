# ====================================================================
# 阶段 1: BUILDER - 编译和安装所有依赖 (包括 NumPy/SciPy/rdtools)
# 使用更完整的 Python 镜像确保所有编译工具可用
# ====================================================================
FROM python:3.11-slim AS builder

# 环境变量：避免生成 .pyc、强制非交互模式
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. 安装构建所需的系统依赖
# 注意：我们将 build-essential 放在这一阶段，最终镜像中不会包含它们。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    # 针对大型科学计算库 (如 NumPy/SciPy) 可能需要的运行时库
    libgomp1 \
    libblas3 \
    liblapack3 \
    # 清理APT缓存
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并安装依赖
COPY requirements.txt .

# 使用 --no-cache-dir 减少 pip 缓存，加速安装
# 这一步会编译 NumPy, rdtools 等所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# ====================================================================
# 阶段 2: RUNTIME - 最终运行环境
# 使用最精简的 slim 镜像来保持最小的 slug 体积
# ====================================================================
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 1. 复制在构建阶段安装的科学计算库（这是解决 'numpy._core.numeric' 问题的关键）
# 必须确保复制到正确的 Python 版本路径下
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 2. 复制可执行文件 (如 gunicorn)
COPY --from=builder /usr/local/bin /usr/local/bin

# 3. 复制项目代码
COPY . .

# 4. 定义启动命令 (Heroku 会自动注入 PORT 环境变量)
# 使用 0.0.0.0:$PORT 确保 Gunicorn 监听所有网络接口
# 假设您的应用入口文件为 index.py，其中定义了 server 变量
CMD gunicorn index:server --bind 0.0.0.0:${PORT:-8000} --timeout 120

# 提示：Heroku 默认使用 $PORT 变量，这里使用 ${PORT:-8000} 是为了本地测试方便