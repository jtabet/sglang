FROM lmsysorg/sglang:v0.5.12.post1-cu130

# Install sglang-kernel (required for some ops)
RUN pip install sglang-kernel --force-reinstall --no-deps 2>/dev/null || true

# Copy our entire sglang source tree (overlay on top of installed package)
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

WORKDIR /sgl-workspace/sglang