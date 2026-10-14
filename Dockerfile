FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/prism

# Copy model directory using environment variable or default
ARG MODEL_DIR=/models/bonsai
COPY ${MODEL_DIR}/Ternary-Bonsai-2-27B-PTQ1_0.gguf /opt/prism/

# Alternative: if model is not available locally, instructions are provided in README
ENV LD_LIBRARY_PATH=/opt/prism:/usr/local/cuda/lib64

ENTRYPOINT ["/opt/prism/llama-server"]
