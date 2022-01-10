FROM nvidia/cudagl:11.1-devel-ubuntu20.04

# env variables for tzdata install
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Vancouver

RUN apt-get update -y && \
    apt-get install -y \
        python3-pip \ 
        python3-dev \
        libcudnn8=8.0.5.39-1+cuda11.1 \
        libcudnn8-dev=8.0.5.39-1+cuda11.1 \
        libsm6 libxext6 libxrender-dev \
        freeglut3-dev \
        libfreetype6-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pytorch and pytorch scatter deps (need to match CUDA version)
RUN pip3 install --no-cache-dir \
    torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Make folder
RUN mkdir -p xray

# Install all other python deps
COPY [ "./setup.py",  "xray/setup.py" ]
RUN python3 -m pip install --no-cache-dir `python3 xray/setup.py --list-reqs`

# Install repo
COPY [ ".",  "xray/" ]
RUN python3 -m pip install --no-cache-dir xray/