# ===== Optimized GPU Dockerfile for AudioProcessor =====
# This Dockerfile is optimized for maximum GPU performance with:
# - Multi-stage build for smaller image size
# - GPU-optimized PyTorch and CUDA libraries
# - Memory-efficient model loading
# - Optimized Python runtime
# - GPU monitoring and profiling tools

# ===== Stage 1: Build stage =====
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as builder

# Set environment variables for build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    cmake \
    ninja-build \
    pkg-config \
    libsndfile1-dev \
    libasound2-dev \
    libfftw3-dev \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel cython

# Install PyTorch with CUDA support (optimized for performance)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other GPU-optimized dependencies
RUN pip install --no-cache-dir \
    tensorflow[and-cuda]==2.13.0 \
    tensorflow-hub \
    transformers[torch] \
    accelerate \
    bitsandbytes \
    flash-attn \
    xformers \
    triton \
    nvidia-ml-py3 \
    pynvml

# Install audio processing libraries
RUN pip install --no-cache-dir \
    librosa \
    soundfile \
    pyloudnorm \
    pyannote.audio \
    laion-clap \
    webrtcvad \
    crepe \
    openai-whisper

# Install machine learning libraries
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    pyarrow \
    sentence-transformers \
    spacy

# Install other dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pydantic-settings \
    celery[redis] \
    redis \
    flower \
    boto3 \
    botocore \
    httpx \
    aiohttp \
    prometheus-client \
    structlog \
    psutil

# ===== Stage 2: Runtime stage =====
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CUDA_VISIBLE_DEVICES=0
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HOME=/app/.cache/huggingface

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    ffmpeg \
    libsndfile1 \
    libasound2 \
    libfftw3-3 \
    libopenblas0 \
    liblapack3 \
    libhdf5-103 \
    libjpeg8 \
    libpng16-16 \
    libtiff5 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    libx264-163 \
    libgtk-3-0 \
    libatlas3-base \
    curl \
    wget \
    htop \
    nvidia-smi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application user
RUN useradd --create-home --shell /bin/bash --uid 1000 audio_processor

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/
COPY *.py /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/.cache/transformers /app/.cache/torch /app/.cache/huggingface \
    && chown -R audio_processor:audio_processor /app

# Copy GPU optimization scripts
COPY install_pytorch_cuda.sh /app/
RUN chmod +x /app/install_pytorch_cuda.sh

# Create GPU monitoring script
RUN echo '#!/bin/bash\n\
echo "=== GPU Information ==="\n\
nvidia-smi\n\
echo "\n=== CUDA Version ==="\n\
nvcc --version\n\
echo "\n=== PyTorch CUDA Support ==="\n\
python -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\"); print(f\"GPU count: {torch.cuda.device_count()}\"); [print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\") for i in range(torch.cuda.device_count())]"\n\
echo "\n=== Memory Usage ==="\n\
python -c "import torch; print(f\"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\") if torch.cuda.is_available() else print(\"No GPU available\")"\n\
' > /app/gpu_info.sh && chmod +x /app/gpu_info.sh

# Create GPU optimization script
RUN echo '#!/bin/bash\n\
echo "=== Optimizing GPU Settings ==="\n\
export CUDA_LAUNCH_BLOCKING=0\n\
export CUDA_CACHE_DISABLE=0\n\
export CUDA_CACHE_MAXSIZE=2147483648\n\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True\n\
export OMP_NUM_THREADS=1\n\
export MKL_NUM_THREADS=1\n\
export NUMEXPR_NUM_THREADS=1\n\
export OPENBLAS_NUM_THREADS=1\n\
export VECLIB_MAXIMUM_THREADS=1\n\
export NUMBA_NUM_THREADS=1\n\
export CUDA_VISIBLE_DEVICES=0\n\
export TORCH_CUDNN_V8_API_ENABLED=1\n\
export TORCH_CUDNN_SDPA_ENABLED=1\n\
echo "GPU optimization environment variables set"\n\
' > /app/optimize_gpu.sh && chmod +x /app/optimize_gpu.sh

# Create health check script
RUN echo '#!/bin/bash\n\
# Check if GPU is available and responsive\n\
if command -v nvidia-smi &> /dev/null; then\n\
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | awk "{print \\$1}" | grep -q "^[0-9]" || exit 1\n\
fi\n\
# Check if API is responding\n\
curl -f http://localhost:8000/health || exit 1\n\
' > /app/health_check.sh && chmod +x /app/health_check.sh

# Set ownership
RUN chown -R audio_processor:audio_processor /app

# Switch to application user
USER audio_processor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/health_check.sh

# Expose ports
EXPOSE 8000 9090

# Default command with GPU optimization
CMD ["/bin/bash", "-c", "source /app/optimize_gpu.sh && uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1"]
