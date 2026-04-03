FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG POLYGEIST_REPO=https://github.com/llvm/Polygeist.git
ARG POLYGEIST_REF=main
ARG LLVM_PROJECTS=clang;mlir;lld
ARG BUILD_TYPE=Release
ARG JOBS=4

# Toolchain and build prerequisites.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    git \
    libedit-dev \
    libncurses-dev \
    libxml2-dev \
    lld \
    ninja-build \
    pybind11-dev \
    python3 \
    python3-dev \
    python3-numpy \
    python3-pybind11 \
    python3-pygments \
    python3-pip \
    python3-venv \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# Build Polygeist and the compatible LLVM/MLIR tree from source.
WORKDIR /opt
RUN git clone --depth 1 --branch ${POLYGEIST_REF} ${POLYGEIST_REPO} polygeist
WORKDIR /opt/polygeist
RUN set -eux; \
        git submodule sync --recursive; \
        for attempt in 1 2 3; do \
            if git submodule update --init --recursive --depth 1; then \
                break; \
            fi; \
            if [ "$attempt" -eq 3 ]; then \
                echo "Submodule fetch failed after retries"; \
                exit 1; \
            fi; \
            sleep 10; \
        done

WORKDIR /opt/polygeist/llvm-project
RUN cmake -S llvm -B build -G Ninja \
    -DLLVM_ENABLE_PROJECTS="${LLVM_PROJECTS}" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
RUN cmake --build build -- -j${JOBS}

WORKDIR /opt/polygeist
RUN cmake -S . -B build -G Ninja \
    -DMLIR_DIR=/opt/polygeist/llvm-project/build/lib/cmake/mlir \
    -DCLANG_DIR=/opt/polygeist/llvm-project/build/lib/cmake/clang \
    -DLLVM_DIR=/opt/polygeist/llvm-project/build/lib/cmake/llvm \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_ASSERTIONS=ON
RUN cmake --build build -- -j${JOBS}

# Install the POC package.
WORKDIR /workspace
COPY POC_initial_demo /workspace/POC_initial_demo
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -e /workspace/POC_initial_demo[dev]

# Runtime environment.
ENV PATH="/opt/polygeist/build/bin:/opt/polygeist/llvm-project/build/bin:${PATH}"
ENV PYTHONPATH="/opt/polygeist/llvm-project/build/tools/mlir/python_packages/mlir_core"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/smoke_test.sh /usr/local/bin/loophole-smoke-test
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/loophole-smoke-test

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
