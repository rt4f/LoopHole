// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: matvec -> stablehlo.dot_general_matvec
// Description: y[m] = sum_k A[m,k] * x[k]  — StableHLO dot_general (matvec)

module {

  func.func @lifted_matvec(%A: tensor<4x4xf32>, %B: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.dot_general %A, %B,
      contracting_dims = [1] x [0]
      : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
