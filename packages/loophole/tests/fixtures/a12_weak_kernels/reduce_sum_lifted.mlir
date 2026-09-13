// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: reduce_sum -> stablehlo.reduce{add}
// Description: B[i] = sum_j A[i,j]  — StableHLO row-wise reduce sum

module {

  func.func @lifted_reduce_sum(%A: tensor<4x4xf32>) -> tensor<4xf32> {
    %init = stablehlo.constant dense<0.0> : tensor<f32>
    %result = stablehlo.reduce(%A init: %init) applies stablehlo.add
      across dimensions = [1]
      : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
