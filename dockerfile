FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Recebe os dados puros do PowerShell
ARG PROXY_USER
ARG PROXY_PASS
ARG PROXY_IP

# Passa para o ambiente de forma oculta
ENV P_USER=$PROXY_USER
ENV P_PASS=$PROXY_PASS
ENV P_IP=$PROXY_IP

# O Python instala tudo, apaga TUDO do OpenCV (inclusive rastros fantasmas) e instala o Headless do zero
RUN python -c "import os, urllib.parse, subprocess, sys; \
    u = urllib.parse.quote(os.environ.get('P_USER', ''), safe=''); \
    p = urllib.parse.quote(os.environ.get('P_PASS', ''), safe=''); \
    i = os.environ.get('P_IP', ''); \
    proxy = f'http://{u}:{p}@{i}:8080' if u and p else ''; \
    cmd1 = ['pip', 'install', '--no-cache-dir', '--proxy', proxy, '-r', 'requirements.txt'] if proxy else ['pip', 'install', '--no-cache-dir', '-r', 'requirements.txt']; \
    cmd2 = ['pip', 'uninstall', '-y', 'opencv-python', 'opencv-contrib-python', 'opencv-python-headless']; \
    cmd3 = ['pip', 'install', '--no-cache-dir', '--proxy', proxy, 'opencv-python-headless'] if proxy else ['pip', 'install', '--no-cache-dir', 'opencv-python-headless']; \
    res = subprocess.run(cmd1); \
    _ = [subprocess.run(c) for c in (cmd2, cmd3)] if res.returncode == 0 else None; \
    sys.exit(res.returncode)"

# Apaga as credenciais da memória
ENV P_USER=""
ENV P_PASS=""

COPY . .