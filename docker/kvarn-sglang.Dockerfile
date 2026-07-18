FROM lmsysorg/sglang:v0.5.12.post1-cu130

# Install sglang-kernel (upstream main requires >=0.4.5)
RUN pip install "sglang-kernel>=0.4.5" --force-reinstall --no-deps 2>/dev/null || true

# Copy our entire sglang source tree (overlay on top of installed package)
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

WORKDIR /sgl-workspace/sglang