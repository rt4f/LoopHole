# This file allows users to call find_package(LLD) and pick up our targets.



set(LLVM_VERSION 18.0.0)
find_package(LLVM ${LLVM_VERSION} EXACT REQUIRED CONFIG
             HINTS "C:/Users/Yasho/LoopHole/build/./lib/cmake/llvm")

set(LLD_EXPORTED_TARGETS "lldCommon;lld;lldCOFF;lldELF;lldMachO;lldMinGW;lldWasm")
set(LLD_CMAKE_DIR "C:/Users/Yasho/LoopHole/build/lib/cmake/lld")
set(LLD_INCLUDE_DIRS "C:/Users/Yasho/LoopHole/Polygeist/llvm-project/lld/include;C:/Users/Yasho/LoopHole/build/tools/lld/include")

# Provide all our library targets to users.
include("C:/Users/Yasho/LoopHole/build/lib/cmake/lld/LLDTargets.cmake")
