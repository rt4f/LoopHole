# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_VERSION=17
ARG LLVM_GITHUB_TAG=llvmorg-17.0.6
ARG POLYGEIST_REF=llvm17bump
ARG BUILD_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    clang-${LLVM_VERSION} \
    libclang-${LLVM_VERSION}-dev \
    libmlir-${LLVM_VERSION}-dev \
    llvm-${LLVM_VERSION}-dev \
    llvm-${LLVM_VERSION}-tools \
    mlir-${LLVM_VERSION}-tools \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

ENV LLVM_PREFIX=/usr/lib/llvm-${LLVM_VERSION}
ENV PATH=${LLVM_PREFIX}/bin:${PATH}
ENV LD_LIBRARY_PATH=${LLVM_PREFIX}/lib:${LD_LIBRARY_PATH}

WORKDIR /opt/src

RUN git clone --depth 1 --branch ${POLYGEIST_REF} https://github.com/llvm/Polygeist.git /opt/src/polygeist

RUN git clone --depth 1 --filter=blob:none --sparse --branch ${LLVM_GITHUB_TAG} \
            https://github.com/llvm/llvm-project.git /opt/src/polygeist/llvm-project && \
        cd /opt/src/polygeist/llvm-project && \
        git sparse-checkout set \
            clang/include \
            clang/lib/CodeGen \
            clang/lib/Driver \
            clang/tools/driver && \
        mkdir -p /opt/src/polygeist/llvm-project/llvm

RUN python3 - <<'PY'
from pathlib import Path

path = Path('/opt/src/polygeist/lib/polygeist/Passes/CMakeLists.txt')
drop_tokens = (
    'ConvertParallelToGPU.cpp',
    'SerializeToCubin.cpp',
    'ParallelLoopDistribute.cpp',
    'ConvertPolygeistToLLVM.cpp',
    'MLIRGPUOps',
    'MLIRNVVMDialect',
)

drop_files = (
    'ConvertParallelToGPU.cpp',
    'SerializeToCubin.cpp',
    'ParallelLoopDistribute.cpp',
    'ConvertPolygeistToLLVM.cpp',
)

original = path.read_text(encoding='utf-8').splitlines()
filtered = [line for line in original if not any(token in line for token in drop_tokens)]
path.write_text('\n'.join(filtered) + '\n', encoding='utf-8')

for filename in drop_files:
    candidate = path.parent / filename
    if candidate.exists():
        candidate.unlink()

patched = path.read_text(encoding='utf-8')
for token in drop_tokens:
    if token in patched:
        raise RuntimeError(f'Failed to remove token: {token}')
PY

RUN python3 - <<'PY'
from pathlib import Path

path = Path('/opt/src/polygeist/tools/cgeist/CMakeLists.txt')
text = path.read_text(encoding='utf-8')
text = text.replace('  MLIRGPUOps\n', '')
text = text.replace('MLIRGPUOps\n', '')
path.write_text(text, encoding='utf-8')

if 'MLIRGPUOps' in path.read_text(encoding='utf-8'):
    raise RuntimeError('Failed to remove MLIRGPUOps from cgeist CMakeLists.txt')
PY

RUN python3 - <<'PY'
from pathlib import Path

path = Path('/opt/src/polygeist/lib/polygeist/Ops.cpp')
text = path.read_text(encoding='utf-8')
if '#include <set>' not in text:
    needle = '#include "mlir/IR/IntegerSet.h"\n'
    if needle not in text:
        raise RuntimeError('Could not find include insertion point in Ops.cpp')
    text = text.replace(needle, needle + '#include <set>\n', 1)
    path.write_text(text, encoding='utf-8')
PY

RUN python3 - <<'PY'
from pathlib import Path

path = Path('/opt/src/polygeist/tools/cgeist/driver.cc')
text = path.read_text(encoding='utf-8')
needle = '#include "clang/../../lib/Driver/ToolChains/Cuda.h"\n'
replacement = '#if POLYGEIST_ENABLE_CUDA\n#include "clang/../../lib/Driver/ToolChains/Cuda.h"\n#endif\n'
if needle in text:
    text = text.replace(needle, replacement, 1)

text = text.replace(
    '#include "mlir/Dialect/DLTI/DLTI.h"\n',
    '#include "mlir/Dialect/DLTI/DLTI.h"\n#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"\n',
)

text = text.replace('  polygeist::registerGpuSerializeToCubinPass();\n', '')
text = text.replace('  MLIRArgs.push_back("-opaque-pointers=0");\n', '')

text = text.replace(
    '  context.getOrLoadDialect<mlir::scf::SCFDialect>();\n',
    '  context.getOrLoadDialect<mlir::scf::SCFDialect>();\n  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();\n',
)

text = text.replace(
    '      } else if (ToCPU.size() != 0) {\n        optPM.addPass(polygeist::createCPUifyPass(ToCPU));\n      }\n',
    '      } else if (ToCPU.size() != 0) {\n        // CPUify pass is not available in this pared-down build.\n      }\n',
)

text = text.replace(
    '        pm3.addPass(polygeist::createConvertPolygeistToLLVMPass(\n            options, CStyleMemRef, /* onlyGpuModules */ false));\n',
    '',
)

path.write_text(text, encoding='utf-8')
PY

RUN python3 - <<'PY'
from pathlib import Path

clang_mlir = Path('/opt/src/polygeist/tools/cgeist/Lib/clang-mlir.cc')
text = clang_mlir.read_text(encoding='utf-8')

patches = (
    (
        'builder.create<LLVM::MemcpyOp>(loc, V, src, typeSize, volatileCpy);',
        'builder.create<LLVM::MemcpyOp>(loc, V, src, typeSize, false);',
    ),
    (
        'Clang->getInvocation().getCodeGenOpts().OpaquePointers = false;\n',
        '',
    ),
    (
        'Clang->getLangOpts().OpenMPIsDevice',
        'Clang->getLangOpts().OpenMPIsTargetDevice',
    ),
    (
        'getLLVMLinkageVarDefinition(VD, /*isConstant*/ false)',
        'getLLVMLinkageVarDefinition(VD)',
    ),
    (
        'getLLVMLinkageVarDefinition(VD,\n                                          /*isConstant*/ false)',
        'getLLVMLinkageVarDefinition(VD)',
    ),
    (
        'APInt::getAllOnesValue(',
        'APInt::getAllOnes(',
    ),
)

for old, new in patches:
    if old in text:
        text = text.replace(old, new)

for token in (
    'OpaquePointers = false',
    'OpenMPIsDevice',
    'getLLVMLinkageVarDefinition(VD, /*isConstant*/ false)',
):
    if token in text:
        raise RuntimeError(f'Failed to patch clang-mlir.cc token: {token}')

clang_mlir.write_text(text, encoding='utf-8')

cgcall = Path('/opt/src/polygeist/tools/cgeist/Lib/CGCall.cc')
cg_text = cgcall.read_text(encoding='utf-8')
cg_patches = (
    (
        'builder.create<LLVM::MemmoveOp>(loc, args[0], args[1], args[2],\n                                        args[3]);',
        'builder.create<LLVM::MemmoveOp>(loc, args[0], args[1], args[2], false);',
    ),
    (
        'builder.create<LLVM::MemmoveOp>(loc, args[0], args[1], args[2], args[3]);',
        'builder.create<LLVM::MemmoveOp>(loc, args[0], args[1], args[2], false);',
    ),
    (
        'builder.create<LLVM::MemsetOp>(loc, args[0], args[1], args[2], args[3]);',
        'builder.create<LLVM::MemsetOp>(loc, args[0], args[1], args[2], false);',
    ),
    (
        'builder.create<LLVM::MemcpyOp>(loc, args[0], args[1], args[2], args[3]);',
        'builder.create<LLVM::MemcpyOp>(loc, args[0], args[1], args[2], false);',
    ),
)

for old, new in cg_patches:
    if old in cg_text:
        cg_text = cg_text.replace(old, new)

cgcall.write_text(cg_text, encoding='utf-8')

cgstmt = Path('/opt/src/polygeist/tools/cgeist/Lib/CGStmt.cc')
stmt_text = cgstmt.read_text(encoding='utf-8')
old_stmt = '/*iterArgs=*/llvm::None'
new_stmt = '/*iterArgs=*/mlir::ValueRange{}'
if old_stmt in stmt_text:
    stmt_text = stmt_text.replace(old_stmt, new_stmt)
    cgstmt.write_text(stmt_text, encoding='utf-8')

type_utils = Path('/opt/src/polygeist/tools/cgeist/Lib/TypeUtils.cc')
tu_text = type_utils.read_text(encoding='utf-8')
old_tu = 'return PointerType::get(anonymize(PT->getPointerElementType()),\n                            PT->getAddressSpace());'
new_tu = 'return PointerType::get(T->getContext(), PT->getAddressSpace());'
if old_tu in tu_text:
    tu_text = tu_text.replace(old_tu, new_tu)
    type_utils.write_text(tu_text, encoding='utf-8')
PY

RUN cmake -S /opt/src/polygeist -B /opt/build/polygeist -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=${LLVM_PREFIX}/lib/cmake/llvm \
      -DMLIR_DIR=${LLVM_PREFIX}/lib/cmake/mlir \
            -DClang_DIR=${LLVM_PREFIX}/lib/cmake/clang && \
    cmake --build /opt/build/polygeist --target cgeist -j ${BUILD_THREADS}

RUN install -m 0755 /opt/build/polygeist/bin/cgeist /usr/local/bin/cgeist && \
        if [ -x /opt/build/polygeist/bin/cgeist++ ]; then \
            install -m 0755 /opt/build/polygeist/bin/cgeist++ /usr/local/bin/cgeist++; \
        else \
            ln -sf /usr/local/bin/cgeist /usr/local/bin/cgeist++; \
        fi

WORKDIR /workspace/POC_initial_demo

COPY requirements.txt /tmp/loophole-requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /tmp/loophole-requirements.txt

CMD ["/bin/bash"]
