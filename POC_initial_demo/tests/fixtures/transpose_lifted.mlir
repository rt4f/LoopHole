// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: transpose -> stablehlo.transpose
// Description: B[j,i] = A[i,j]  — StableHLO transpose

module {

  func.func @lifted_transpose(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %result = stablehlo.transpose %A, dims = [1, 0]
      : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
