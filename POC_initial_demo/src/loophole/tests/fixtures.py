"""
fixtures.py — Canonical MLIR Affine IR examples for testing and demo.

Each fixture is a complete MLIR function in textual format, representing
a standard linear algebra kernel in scalar loop-based style (as Polygeist
would produce from legacy C code).
"""

# ---------------------------------------------------------------------------
# Matrix Multiply: C[i,j] += A[i,k] * B[k,j]
# ---------------------------------------------------------------------------

MATMUL_MLIR = """\
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.for %k = 0 to 4 {
        %a = affine.load %A[%i, %k] : memref<4x4xf32>
        %b = affine.load %B[%k, %j] : memref<4x4xf32>
        %c = affine.load %C[%i, %j] : memref<4x4xf32>
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        affine.store %add, %C[%i, %j] : memref<4x4xf32>
      }
    }
  }
  return
}
"""

# Larger matmul for benchmarking
MATMUL_128_MLIR = """\
func.func @matmul_128(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) {
  affine.for %i = 0 to 128 {
    affine.for %j = 0 to 128 {
      affine.for %k = 0 to 128 {
        %a = affine.load %A[%i, %k] : memref<128x128xf32>
        %b = affine.load %B[%k, %j] : memref<128x128xf32>
        %c = affine.load %C[%i, %j] : memref<128x128xf32>
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        affine.store %add, %C[%i, %j] : memref<128x128xf32>
      }
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# 2-D Transpose: B[j,i] = A[i,j]
# ---------------------------------------------------------------------------

TRANSPOSE_2D_MLIR = """\
func.func @transpose(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %a = affine.load %A[%i, %j] : memref<4x4xf32>
      affine.store %a, %B[%j, %i] : memref<4x4xf32>
    }
  }
  return
}
"""

# Non-square transpose
TRANSPOSE_NONSQUARE_MLIR = """\
func.func @transpose_nonsquare(%A: memref<3x5xf32>, %B: memref<5x3xf32>) {
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 5 {
      %a = affine.load %A[%i, %j] : memref<3x5xf32>
      affine.store %a, %B[%j, %i] : memref<5x3xf32>
    }
  }
  return
}
"""

# 3-D transpose with non-reverse permutation [1, 2, 0]
TRANSPOSE_3D_PERM_120_MLIR = """\
func.func @transpose3d_perm120(%A: memref<2x3x4xf32>, %B: memref<3x4x2xf32>) {
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 3 {
      affine.for %k = 0 to 4 {
        %a = affine.load %A[%i, %j, %k] : memref<2x3x4xf32>
        affine.store %a, %B[%j, %k, %i] : memref<3x4x2xf32>
      }
    }
  }
  return
}
"""

# 3-D transpose with non-reverse permutation [2, 0, 1]
TRANSPOSE_3D_PERM_201_MLIR = """\
func.func @transpose3d_perm201(%A: memref<2x3x4xf32>, %B: memref<4x2x3xf32>) {
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 3 {
      affine.for %k = 0 to 4 {
        %a = affine.load %A[%i, %j, %k] : memref<2x3x4xf32>
        affine.store %a, %B[%k, %i, %j] : memref<4x2x3xf32>
      }
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# 1-D Convolution: O[n,w] += I[n,w+kw] * K[kw]
# ---------------------------------------------------------------------------

CONV_1D_MLIR = """\
func.func @conv1d(%I: memref<1x8xf32>, %K: memref<3xf32>, %O: memref<1x6xf32>) {
  affine.for %n = 0 to 1 {
    affine.for %w = 0 to 6 {
      affine.for %kw = 0 to 3 {
        %i_val = affine.load %I[%n, %w + %kw] : memref<1x8xf32>
        %k_val = affine.load %K[%kw] : memref<3xf32>
        %o_val = affine.load %O[%n, %w] : memref<1x6xf32>
        %mul = arith.mulf %i_val, %k_val : f32
        %add = arith.addf %o_val, %mul : f32
        affine.store %add, %O[%n, %w] : memref<1x6xf32>
      }
    }
  }
  return
}
"""

# 1-D convolution with non-unit stride and dilation: I[n, 2*w + 3*kw]
CONV_1D_STRIDED_DILATED_MLIR = """\
func.func @conv1d_strided_dilated(%I: memref<1x9xf32>, %K: memref<2xf32>, %O: memref<1x3xf32>) {
  affine.for %n = 0 to 1 {
    affine.for %w = 0 to 3 {
      affine.for %kw = 0 to 2 {
        %i_val = affine.load %I[%n, %w * 2 + %kw * 3] : memref<1x9xf32>
        %k_val = affine.load %K[%kw] : memref<2xf32>
        %o_val = affine.load %O[%n, %w] : memref<1x3xf32>
        %mul = arith.mulf %i_val, %k_val : f32
        %add = arith.addf %o_val, %mul : f32
        affine.store %add, %O[%n, %w] : memref<1x3xf32>
      }
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# 2-D Convolution (simple, no batch/channels): O[oh,ow] += I[oh+kh,ow+kw]*K[kh,kw]
# ---------------------------------------------------------------------------

CONV_2D_SIMPLE_MLIR = """\
func.func @conv2d(%I: memref<6x6xf32>, %K: memref<3x3xf32>, %O: memref<4x4xf32>) {
  affine.for %oh = 0 to 4 {
    affine.for %ow = 0 to 4 {
      affine.for %kh = 0 to 3 {
        affine.for %kw = 0 to 3 {
          %i_val = affine.load %I[%oh + %kh, %ow + %kw] : memref<6x6xf32>
          %k_val = affine.load %K[%kh, %kw] : memref<3x3xf32>
          %o_val = affine.load %O[%oh, %ow] : memref<4x4xf32>
          %mul = arith.mulf %i_val, %k_val : f32
          %add = arith.addf %o_val, %mul : f32
          affine.store %add, %O[%oh, %ow] : memref<4x4xf32>
        }
      }
    }
  }
  return
}
"""

# 2-D convolution with non-unit stride/dilation per axis:
# h: I[2*oh + 3*kh], w: I[ow + 2*kw]
CONV_2D_STRIDED_DILATED_MLIR = """\
func.func @conv2d_strided_dilated(%I: memref<8x8xf32>, %K: memref<2x2xf32>, %O: memref<3x6xf32>) {
  affine.for %oh = 0 to 3 {
    affine.for %ow = 0 to 6 {
      affine.for %kh = 0 to 2 {
        affine.for %kw = 0 to 2 {
          %i_val = affine.load %I[%oh * 2 + %kh * 3, %ow + %kw * 2] : memref<8x8xf32>
          %k_val = affine.load %K[%kh, %kw] : memref<2x2xf32>
          %o_val = affine.load %O[%oh, %ow] : memref<3x6xf32>
          %mul = arith.mulf %i_val, %k_val : f32
          %add = arith.addf %o_val, %mul : f32
          affine.store %add, %O[%oh, %ow] : memref<3x6xf32>
        }
      }
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# 2-D Convolution NHWC: O[n,oh,ow,oc] += I[n,oh+kh,ow+kw,ic] * K[kh,kw,ic,oc]
# ---------------------------------------------------------------------------

CONV_2D_NHWC_MLIR = """\
func.func @conv2d_nhwc(%I: memref<1x6x6x2xf32>, %K: memref<3x3x2x4xf32>, %O: memref<1x4x4x4xf32>) {
  affine.for %n = 0 to 1 {
    affine.for %oh = 0 to 4 {
      affine.for %ow = 0 to 4 {
        affine.for %kh = 0 to 3 {
          affine.for %kw = 0 to 3 {
            affine.for %ic = 0 to 2 {
              affine.for %oc = 0 to 4 {
                %i_val = affine.load %I[%n, %oh + %kh, %ow + %kw, %ic] : memref<1x6x6x2xf32>
                %k_val = affine.load %K[%kh, %kw, %ic, %oc] : memref<3x3x2x4xf32>
                %o_val = affine.load %O[%n, %oh, %ow, %oc] : memref<1x4x4x4xf32>
                %mul = arith.mulf %i_val, %k_val : f32
                %add = arith.addf %o_val, %mul : f32
                affine.store %add, %O[%n, %oh, %ow, %oc] : memref<1x4x4x4xf32>
              }
            }
          }
        }
      }
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# Dot Product: c += A[k] * B[k]
# ---------------------------------------------------------------------------

DOT_PRODUCT_MLIR = """\
func.func @dot(%A: memref<8xf32>, %B: memref<8xf32>, %c: memref<1xf32>) {
  affine.for %k = 0 to 8 {
    %a = affine.load %A[%k] : memref<8xf32>
    %b = affine.load %B[%k] : memref<8xf32>
    %c_val = affine.load %c[0] : memref<1xf32>
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c_val, %mul : f32
    affine.store %add, %c[0] : memref<1xf32>
  }
  return
}
"""

# ---------------------------------------------------------------------------
# Matrix-Vector Multiply: y[m] += A[m,k] * x[k]
# ---------------------------------------------------------------------------

MATVEC_MLIR = """\
func.func @matvec(%A: memref<4x4xf32>, %x: memref<4xf32>, %y: memref<4xf32>) {
  affine.for %m = 0 to 4 {
    affine.for %k = 0 to 4 {
      %a = affine.load %A[%m, %k] : memref<4x4xf32>
      %xk = affine.load %x[%k] : memref<4xf32>
      %ym = affine.load %y[%m] : memref<4xf32>
      %mul = arith.mulf %a, %xk : f32
      %add = arith.addf %ym, %mul : f32
      affine.store %add, %y[%m] : memref<4x4xf32>
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# Elementwise Add: C[i,j] = A[i,j] + B[i,j]
# ---------------------------------------------------------------------------

ELEMENTWISE_ADD_MLIR = """\
func.func @elementwise_add(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %a = affine.load %A[%i, %j] : memref<4x4xf32>
      %b = affine.load %B[%i, %j] : memref<4x4xf32>
      %add = arith.addf %a, %b : f32
      affine.store %add, %C[%i, %j] : memref<4x4xf32>
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# Row-wise Sum Reduction: B[i] += A[i,j]
# ---------------------------------------------------------------------------

REDUCE_SUM_MLIR = """\
func.func @reduce_sum(%A: memref<4x4xf32>, %B: memref<4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %a = affine.load %A[%i, %j] : memref<4x4xf32>
      %b = affine.load %B[%i] : memref<4xf32>
      %add = arith.addf %b, %a : f32
      affine.store %add, %B[%i] : memref<4xf32>
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# Mixed affine/scf nests for parser hardening (B-03)
# ---------------------------------------------------------------------------

MIXED_AFFINE_SCF_MATMUL_MLIR = """\
func.func @mixed_affine_scf_matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  affine.for %i = 0 to 4 {
    scf.for %j = %c0 to %c4 step %c1 {
      affine.for %k = 0 to 4 {
        %a = affine.load %A[%i, %k] : memref<4x4xf32>
        %b = affine.load %B[%k, %j] : memref<4x4xf32>
        %c = affine.load %C[%i, %j] : memref<4x4xf32>
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        affine.store %add, %C[%i, %j] : memref<4x4xf32>
      }
    }
  }
  return
}
"""

MIXED_SCF_AFFINE_REDUCTION_MLIR = """\
func.func @mixed_scf_affine_reduction(%A: memref<4x4xf32>, %B: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %m = %c0 to %c4 step %c1 {
    affine.for %k = 0 to 4 {
      %a = affine.load %A[%m, %k] : memref<4x4xf32>
      %b = affine.load %B[%m] : memref<4xf32>
      %sum = arith.addf %b, %a : f32
      affine.store %sum, %B[%m] : memref<4xf32>
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# ReLU: B[i,j] = max(A[i,j], 0)
# ---------------------------------------------------------------------------

RELU_MLIR = """\
func.func @relu(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %a = affine.load %A[%i, %j] : memref<4x4xf32>
      %zero = arith.constant 0.0 : f32
      %res = arith.maxf %a, %zero : f32
      affine.store %res, %B[%i, %j] : memref<4x4xf32>
    }
  }
  return
}
"""

# ---------------------------------------------------------------------------
# Index expression variation fixtures for normalization (B-04)
# ---------------------------------------------------------------------------

INDEX_VARIATION_EQ_A_MLIR = """\
func.func @index_variation_a(%A: memref<16xf32>, %B: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  affine.for %i = %c0 to %c8 {
    %a = affine.load %A[%i + (%c1 + %c2)] : memref<16xf32>
    affine.store %a, %B[(%i + 3)] : memref<16xf32>
  }
  return
}
"""

INDEX_VARIATION_EQ_B_MLIR = """\
func.func @index_variation_b(%A: memref<16xf32>, %B: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  affine.for %i = %c0 to %c8 {
    %a = affine.load %A[%c2 + (1 + %i)] : memref<16xf32>
    affine.store %a, %B[1 + (2 + %i)] : memref<16xf32>
  }
  return
}
"""

# ---------------------------------------------------------------------------
# Demo fixture dictionary — ordered for display
# ---------------------------------------------------------------------------

DEMO_FIXTURES = {
    "matmul_4x4":         MATMUL_MLIR,
    "transpose_2d":       TRANSPOSE_2D_MLIR,
    "conv1d":             CONV_1D_MLIR,
    "conv1d_strided_dilated": CONV_1D_STRIDED_DILATED_MLIR,
    "conv2d_simple":      CONV_2D_SIMPLE_MLIR,
    "conv2d_strided_dilated": CONV_2D_STRIDED_DILATED_MLIR,
    "conv2d_nhwc":        CONV_2D_NHWC_MLIR,
    "dot_product":        DOT_PRODUCT_MLIR,
    "matvec":             MATVEC_MLIR,
    "elementwise_add":    ELEMENTWISE_ADD_MLIR,
    "reduce_sum":         REDUCE_SUM_MLIR,
    "relu":               RELU_MLIR,
}

# All fixtures dict (includes larger variants)
ALL_FIXTURES = {
    **DEMO_FIXTURES,
    "matmul_128x128":         MATMUL_128_MLIR,
    "transpose_nonsquare":    TRANSPOSE_NONSQUARE_MLIR,
  "transpose_3d_perm_120":  TRANSPOSE_3D_PERM_120_MLIR,
  "transpose_3d_perm_201":  TRANSPOSE_3D_PERM_201_MLIR,
  "mixed_affine_scf_matmul": MIXED_AFFINE_SCF_MATMUL_MLIR,
  "mixed_scf_affine_reduction": MIXED_SCF_AFFINE_REDUCTION_MLIR,
  "index_variation_eq_a": INDEX_VARIATION_EQ_A_MLIR,
  "index_variation_eq_b": INDEX_VARIATION_EQ_B_MLIR,
}
