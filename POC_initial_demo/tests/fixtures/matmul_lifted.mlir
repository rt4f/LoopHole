// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: matmul → stablehlo.dot_general
// Description: C[m,n] = sum_k A[m,k] * B[k,n]  — StableHLO dot_general (matmul)

module {

  func.func @lifted_matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %result = stablehlo.dot_general %A, %B,
      contracting_dims = [1] x [0]
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
