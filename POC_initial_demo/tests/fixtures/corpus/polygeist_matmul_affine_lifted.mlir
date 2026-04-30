// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: polygeist_matmul_affine -> stablehlo.dot_general
// Description: C[m,n] = sum_k A[m,k] * B[k,n]  — StableHLO dot_general (matmul)

module {

  func.func @lifted_polygeist_matmul_affine(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %result = stablehlo.dot_general %A, %B,
      contracting_dims = [1] x [0]
      : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    return %result : tensor<8x8xf32>
  }
}
