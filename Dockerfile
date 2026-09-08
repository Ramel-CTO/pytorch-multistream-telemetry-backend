# Multi-stage build layer to keep the final edge image lightweight
FROM nvcr.io/nvidia/tensorrt:24.01-py3 AS builder

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install system compiler tools and native C++ dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and unpack native LibTorch C++ binaries (Pre-built with CUDA support)
RUN wget https://pytorch.org -O libtorch.zip \
    && unzip libtorch.zip -d /opt/ \
    && rm libtorch.zip

COPY . .

# Compile your high-performance C++ multistream engine
RUN mkdir build && cd build && \
    cmake -DCMAKE_PREFIX_PATH=/opt/libtorch .. && \
    make -j$(nproc)

# --- Runtime Execution Stage ---
FROM ubuntu:22.04

WORKDIR /app
# Pull only the compiled binary and necessary runtime dynamic libraries
COPY --from=builder /app/build/telemetry_engine /app/telemetry_engine
COPY --from=builder /usr/lib/x86_64-linux-gnu/libopencv_*.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /opt/libtorch/lib/ /opt/libtorch/lib/

ENV LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Open network port for asynchronous telemetry metadata streaming
EXPOSE 8080

ENTRYPOINT ["./telemetry_engine"]
