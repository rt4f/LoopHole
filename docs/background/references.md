# 06 — References and Resources

> Organized bibliography for the LoopHole project. Covers all cited works from `docs/background/problem-statement.md`, supplementary tools, and online resources gathered during research.

---

## 6.1 Core Program Lifting Tools (Primary SOTA)

### 6.1.1 mlirSynth

| Field | Value |
|---|---|
| **Title** | "Automatic Lifting of C Code to High-Level MLIR Dialects via Program Synthesis" |
| **Authors** | Alexander Brauckmann, Elizabeth Polgreen, Tobias Grosser, Elizabeth Polgreen |
| **Venue** | PACT 2023 (International Conference on Parallel Architectures and Compilation Techniques) |
| **ArXiv** | https://arxiv.org/abs/2310.04196 |
| **GitHub** | https://github.com/alexanderb14/mlirSynth |
| **Relevance** | Direct predecessor to LoopHole; first tool to lift C loops directly to MLIR dialects using enumerative synthesis |

### 6.1.2 Tenspiler

| Field | Value |
|---|---|
| **Title** | "Tenspiler: A Tensor-to-Tensor Transpiler Using Program Synthesis" |
| **Authors** | Jie Qiu, Colin Cai, Sahil Bhatia, Niranjan Hasabnis, Penporn Koanantakool, Alvin Cheung |
| **Venue** | ECOOP 2024 (European Conference on Object-Oriented Programming) |
| **URL** | https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECOOP.2024.37 |
| **GitHub** | https://github.com/tenspiler/tenspiler |
| **Relevance** | SMT-based synthesis targeting multiple ML tensor backends (NumPy, PyTorch, TensorFlow); demonstrated 105× CPU and 37× GPU speedups |

### 6.1.3 Tensorize (CGO 2025)

| Field | Value |
|---|---|
| **Title** | "From Loops to Tensors: Lifting C Loops to High-Level Tensor Computations" |
| **Authors** | Rongxiao Fu, John Wickerson, Nicolas Vasilache, Tobias Grosser |
| **Venue** | CGO 2025 (International Symposium on Code Generation and Optimization) |
| **ArXiv** | https://arxiv.org/abs/2501.09428 |
| **Relevance** | SymPy-based symbolic tracing; achieves 4102× GPU speedup for CNN kernels on Polybench; direct technical comparison baseline for LoopHole |

### 6.1.4 STAGG (PLDI 2025)

| Field | Value |
|---|---|
| **Title** | "STAGG: LLM-Based Synthesis of Tensor Algebra with Generalization Guarantees" |
| **Authors** | Sahil Bhatia, Jie Qiu, Sanjit A. Seshia, Alvin Cheung |
| **Venue** | PLDI 2025 (ACM SIGPLAN Conference on Programming Language Design and Implementation) |
| **ArXiv** | https://arxiv.org/abs/2504.19705 |
| **Relevance** | LLM-guided probabilistic CFG + A* search; 99% accuracy, 3.19s avg; represents current state-of-the-art |

---

## 6.2 Related Lifting and Transpilation Tools

### 6.2.1 QiMeng-Xpiler

| Field | Value |
|---|---|
| **Title** | "QiMeng-Xpiler: Automatic Generation of High-Performance Tensor Operator from C Code" |
| **Authors** | Jinhao Xu et al. |
| **Venue** | SC 2024 (International Conference for High Performance Computing, Networking, Storage, and Analysis) |
| **Relevance** | 3-stage pipeline (pattern matching → polyhedral analysis → code generation) for HPC operator lifting; tested on 37 HPC benchmarks |

### 6.2.2 LLMLift

| Field | Value |
|---|---|
| **Title** | "LLMLift: Using Large Language Models for Lifting Scalar Linear Algebra Code to Tensor Linear Algebra Programs" |
| **Authors** | Chaofan Du et al. |
| **Venue** | Preprint 2024 |
| **Relevance** | Direct GPT-4 lifting approach (no SMT); 74.6% accuracy, <1s; combined with verification to prune incorrect lifts |

### 6.2.3 STENSO

| Field | Value |
|---|---|
| **Title** | "STENSO: Synthesis of Tensor Computations via Program Enumeration" |
| **Authors** | Ryo Kimura et al. |
| **Venue** | ASPLOS 2024 |
| **Relevance** | TACO tensor DSL target via enumerative synthesis; exhaustive search with observational equivalence |

### 6.2.4 Konrul

| Field | Value |
|---|---|
| **Title** | "Konrul: Automatic Compiler Specification from Loop Programs" |
| **Authors** | Multiple (affiliation: Intel Labs) |
| **Venue** | PLDI 2025 (companion paper to STAGG listing period) |
| **Relevance** | Automatic compiler rule synthesis from loop examples; relevant for the "learn from examples" synthesis paradigm |

---

## 6.3 MLIR and LLVM Infrastructure

### 6.3.1 MLIR Framework

| Field | Value |
|---|---|
| **Title** | "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation" |
| **Authors** | Chris Lattner, Mehdi Amini, Uday Bondhugula et al. |
| **Venue** | CGO 2021 |
| **DOI** | https://doi.org/10.1109/CGO51591.2021.9370308 |
| **Docs** | https://mlir.llvm.org/ |

### 6.3.2 Linalg Dialect

| Field | Value |
|---|---|
| **Documentation** | https://mlir.llvm.org/docs/Dialects/Linalg/ |
| **Rationale** | https://mlir.llvm.org/docs/Dialects/Linalg/OpDSL/ |
| **Key Concepts** | linalg.generic, named ops, structured op interface, comprehensive lowering to loops |

### 6.3.3 StableHLO

| Field | Value |
|---|---|
| **Documentation** | https://openxla.org/stablehlo |
| **GitHub** | https://github.com/openxla/stablehlo |
| **Specification** | https://github.com/openxla/stablehlo/blob/main/docs/spec.md |
| **Relevance** | Target dialect for TPU/XLA compilation; subset of HLO that is stable across XLA versions |

### 6.3.4 Transform Dialect

| Field | Value |
|---|---|
| **Documentation** | https://mlir.llvm.org/docs/Dialects/Transform/ |
| **RFC** | https://discourse.llvm.org/t/rfc-transform-dialect/6793 |
| **Tutorial** | https://mlir.llvm.org/docs/Tutorials/transform/ |
| **Relevance** | Schedule specification for Linalg ops; enables hardware-specific tiling, packing, vectorization |

### 6.3.5 Affine Dialect

| Field | Value |
|---|---|
| **Documentation** | https://mlir.llvm.org/docs/Dialects/Affine/ |
| **Relevance** | Input dialect of LoopHole; contains affine.for, affine.load, affine.store that are the synthesis targets |

### 6.3.6 SparseTensor Dialect

| Field | Value |
|---|---|
| **Documentation** | https://mlir.llvm.org/docs/Dialects/SparseTensorOps/ |
| **Paper** | "Compiler Support for Sparse Tensor Computations in MLIR" (TACO journal 2022) |
| **Author** | Aart Bik, Penporn Koanantakool, Mohammad Sadeghi-Sorkhabi, Nicolas Vasilache |
| **GitHub (sparsifier)** | Part of LLVM monorepo: llvm-project/mlir/lib/Dialect/SparseTensor/ |
| **Relevance** | Target dialect for sparse lifting direction (Research Track 1) |

---

## 6.4 Frontends and Polyhedral Analysis

### 6.4.1 Polygeist

| Field | Value |
|---|---|
| **Title** | "Polygeist: Raising C to Polyhedral MLIR" |
| **Authors** | William Moses, Lorenzo Chelini, Ruizhe Zhao, Oleksandr Zinenko |
| **Venue** | PACT 2021 |
| **GitHub** | https://github.com/llvm/Polygeist |
| **Relevance** | C/C++/Fortran → MLIR Affine IR frontend; the "ingestion" layer for LoopHole |

### 6.4.2 PPCG (Polyhedral C Code Generator)

| Field | Value |
|---|---|
| **Title** | "Generating Code for Polyhedral Computations Using the ISL Code Generator" |
| **Authors** | Tobias Grosser, Armin Grösslinger, Christian Lengauer |
| **GitHub** | https://github.com/Meinersbur/ppcg |
| **Relevance** | Polyhedral-to-GPU code generation; older approach that LoopHole supersedes for modern targets |

### 6.4.3 Pluto

| Field | Value |
|---|---|
| **Title** | "A Practical Automatic Polyhedral Parallelizer and Locality Optimizer" |
| **Authors** | Uday Bondhugula, Albert Hartono, J. Ramanujam, P. Sadayappan |
| **Venue** | PLDI 2008 |
| **Relevance** | Classic polyhedral transformation; provides theoretical underpinning for affine loop analysis |

---

## 6.5 Program Synthesis Foundations

### 6.5.1 Z3 SMT Solver

| Field | Value |
|---|---|
| **Title** | "Z3: An Efficient SMT Solver" |
| **Authors** | Leonardo de Moura, Nikolaj Bjørner |
| **Venue** | TACAS 2008 |
| **GitHub** | https://github.com/Z3Prover/z3 |
| **Python API** | `pip install z3-solver` |
| **Key API** | z3.Int, z3.Real, z3.ForAll, z3.Solver, z3.sat, z3.unsat |

### 6.5.2 CVC5 SMT Solver

| Field | Value |
|---|---|
| **Title** | "cvc5: A Versatile and Industrial-Strength SMT Solver" |
| **Authors** | Haniel Barbosa et al. |
| **Venue** | TACAS 2022 |
| **GitHub** | https://github.com/cvc5/cvc5 |
| **Relevance** | Alternative to Z3; better support for nonlinear arithmetic and quantifiers; used by Tenspiler |

### 6.5.3 Rosette (Sketch-Based Synthesis)

| Field | Value |
|---|---|
| **Title** | "Growing Solver-Aided Languages with Rosette" |
| **Authors** | Emina Torlak, Rastislav Bodik |
| **Venue** | Onward! 2013 |
| **URL** | https://emina.github.io/rosette/ |
| **Relevance** | Sketch-based program synthesis; theoretical foundation for the POC's sketch approach |

### 6.5.4 SKETCH Program Synthesizer

| Field | Value |
|---|---|
| **Title** | "Program Synthesis by Sketching" |
| **Authors** | Armando Solar-Lezama |
| **Venue** | PhD Thesis UC Berkeley 2008 |
| **URL** | https://people.csail.mit.edu/asolar/papers/thesis.pdf |
| **Relevance** | Original work on sketch-based synthesis; the "sketch library" concept in LoopHole POC |

### 6.5.5 LILO (Interpretable Library Learning)

| Field | Value |
|---|---|
| **Title** | "LILO: Learning Interpretable Libraries by Compressing and Documenting Code" |
| **Authors** | Gabriel Grand, Lionel Wong, Maddy Bowers, Theo X. Olausson et al. |
| **Venue** | ICLR 2024 |
| **ArXiv** | https://arxiv.org/abs/2310.19791 |
| **Relevance** | Foundation for the Auto-Documentation research direction (Track 4) |

---

## 6.6 Machine Learning for Code

### 6.6.1 AlphaCode 2

| Field | Value |
|---|---|
| **Title** | "AlphaCode 2 Technical Report" |
| **Authors** | Google DeepMind |
| **URL** | https://alphacoderesearch.com/alphacode2_technical_report.pdf |
| **Relevance** | State-of-the-art LLM for competitive programming; demonstrates LLM capability for complex code generation |

### 6.6.2 CodeLlama

| Field | Value |
|---|---|
| **Title** | "Code Llama: Open Foundation Models for Code" |
| **Authors** | Baptiste Rozière et al. (Meta AI) |
| **ArXiv** | https://arxiv.org/abs/2308.12950 |
| **Relevance** | Open-weight LLM for code generation; usable locally for LoopHole LLM components |

---

## 6.7 Benchmarks and Evaluation

### 6.7.1 Polybench/C

| Field | Value |
|---|---|
| **Title** | PolyBench/C 4.2.1 — The Polyhedral Benchmark Suite |
| **Maintainer** | Louis-Noel Pouchet, et al. |
| **URL** | https://sourceforge.net/projects/polybench/ |
| **GitHub Mirror** | https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1 |
| **Content** | 30 numerical kernels from HPC domains: linear algebra, stencils, image processing, data mining |
| **Relevance** | Primary benchmark suite for LoopHole POC and evaluation; all SOTA papers use this |

### 6.7.2 MLPerf

| Field | Value |
|---|---|
| **URL** | https://mlcommons.org/en/inference-edge-21/ |
| **Relevance** | ML inference benchmark; relevant if LoopHole targets ML preprocessing kernels |

### 6.7.3 NPBench

| Field | Value |
|---|---|
| **Title** | "NPBench: Benchmarking the NumPy Scientific Python Interface" |
| **GitHub** | https://github.com/spcl/npbench |
| **Relevance** | Used by Tenspiler; Python NumPy reference implementations useful for cross-validation |

---

## 6.8 Hardware Backend Resources

### 6.8.1 XLA (Accelerated Linear Algebra)

| Field | Value |
|---|---|
| **Documentation** | https://www.tensorflow.org/xla |
| **OpenXLA** | https://github.com/openxla/xla |
| **Relevance** | Backend for StableHLO compilation; targets TPU, GPU, CPU |

### 6.8.2 IREE (Intermediate Representation Execution Environment)

| Field | Value |
|---|---|
| **GitHub** | https://github.com/iree-org/iree |
| **Documentation** | https://iree.dev/ |
| **Relevance** | Backend compiler for Linalg → CPU/GPU/Mobile; alternative to XLA; used by mlirSynth |

### 6.8.3 Triton

| Field | Value |
|---|---|
| **GitHub** | https://github.com/triton-lang/triton |
| **Documentation** | https://triton-lang.org/ |
| **Relevance** | GPU kernel language; StableHLO can sometimes lower to Triton for NVIDIA GPUs |

---

## 6.9 Formal Verification Tools

### 6.9.1 CBMC (C Bounded Model Checker)

| Field | Value |
|---|---|
| **GitHub** | https://github.com/diffblue/cbmc |
| **Documentation** | https://www.cprover.org/cbmc/ |
| **Relevance** | Alternative to Z3 for C-level equivalence checking; bit-precise reasoning on C programs |

### 6.9.2 Alive2

| Field | Value |
|---|---|
| **Title** | "Alive2: Bounded Translation Validation for LLVM" |
| **GitHub** | https://github.com/AliveToolkit/alive2 |
| **Relevance** | LLVM IR optimization validator; inspiration for MLIR-level equivalence checking approach |

---

## 6.10 Key Online Resources

| Resource | URL | Purpose |
|---|---|---|
| MLIR main docs | https://mlir.llvm.org/docs/ | Dialect and API reference |
| LLVM Discourse (MLIR) | https://discourse.llvm.org/c/mlir/31 | Community discussions, RFCs |
| MLIR Python Bindings | https://mlir.llvm.org/docs/Bindings/Python/ | API reference for Python integration |
| IREE Project Blog | https://iree.dev/blog/ | Applied MLIR lowering examples |
| OpenXLA Organization | https://github.com/openxla | StableHLO, XLA, IREE repos |
| Polygeist GitHub | https://github.com/llvm/Polygeist | C → MLIR Affine frontend |
| LLVM Project (monorepo) | https://github.com/llvm/llvm-project | Contains all MLIR core dialects |
| MLIR Tutorial | https://mlir.llvm.org/docs/Tutorials/Toy/ | Toy language tutorial (start here) |
| PolyBench | https://sourceforge.net/projects/polybench/ | Benchmark C kernels |
| Z3 Python Tutorial | https://ericpony.github.io/z3py-tutorial/ | Z3 Python API examples |

---

## 6.11 Recommended Reading Order

For a newcomer to the project, read in this order:

1. **MLIR Toy Tutorial** (https://mlir.llvm.org/docs/Tutorials/Toy/) — hands-on intro to defining dialects, lowering passes
2. **Linalg Dialect Rationale** — Understand the structured op model and generalization
3. **mlirSynth paper** (arXiv:2310.04196) — Understand the baseline synthesis approach
4. **Tensorize paper** (arXiv:2501.09428) — Understand SymPy-based lifting; the most technically complete paper on the approach
5. **STAGG paper** (arXiv:2504.19705) — Understand where the field is heading with LLMs
6. **Z3 Python Tutorial** — Become fluent in SMT-based equivalence checking
7. **SparseTensor Dialect docs** — Understand the frontier for Research Track 1

---

*Last updated: February 2026. For most recent papers, monitor ACM Digital Library for PLDI 2025, CGO 2025, ASPLOS 2025 proceedings.*
