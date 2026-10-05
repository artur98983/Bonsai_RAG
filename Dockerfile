FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/prism

COPY prism/llama-prism-b10709-9a9394a/ /opt/prism/

ENV LD_LIBRARY_PATH=/opt/prism:/usr/local/cuda/lib64

ENTRYPOINT ["/opt/prism/llama-server"]
