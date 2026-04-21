// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: dot → stablehlo.dot_general_vecdot
// Description: c = sum_k A[k] * B[k]  — StableHLO dot_general (vector dot)

module {

  func.func @lifted_dot(%A: tensor<8xf32>, %B: tensor<8xf32>) -> tensor<1xf32> {
    %result = stablehlo.dot_general %A, %B,
      contracting_dims = [0] x [0]
      : (tensor<8xf32>, tensor<8xf32>) -> tensor<1xf32>
    return %result : tensor<1xf32>
  }
}
