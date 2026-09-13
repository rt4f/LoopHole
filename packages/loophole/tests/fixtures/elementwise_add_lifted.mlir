// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: elementwise_add -> stablehlo.add
// Description: C[i,j] = A[i,j] + B[i,j]  — StableHLO elementwise add

module {

  func.func @lifted_elementwise_add(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %result = stablehlo.add %A, %B : tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
