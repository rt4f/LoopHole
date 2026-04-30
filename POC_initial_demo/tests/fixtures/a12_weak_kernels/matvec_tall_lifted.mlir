// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: matvec_tall -> stablehlo.dot_general_matvec
// Description: y[m] = sum_k A[m,k] * x[k]  — StableHLO dot_general (matvec)

module {

  func.func @lifted_matvec_tall(%A: tensor<8x3xf32>, %B: tensor<3xf32>) -> tensor<8xf32> {
    %result = stablehlo.dot_general %A, %B,
      contracting_dims = [1] x [0]
      : (tensor<8x3xf32>, tensor<3xf32>) -> tensor<8xf32>
    return %result : tensor<8xf32>
  }
}
