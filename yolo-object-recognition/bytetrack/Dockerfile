FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Clone ByteTrack repository
RUN git clone https://github.com/ifzhang/ByteTrack.git
WORKDIR /workspace/ByteTrack

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.0.0 torchvision==0.15.1
RUN pip3 install -r requirements.txt
RUN pip3 install cython
RUN pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip3 install cython_bbox

# Install ByteTrack
RUN python3 setup.py develop

# Set up working directory
WORKDIR /workspace/ByteTrack

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video

# Default command
CMD ["/bin/bash"] 