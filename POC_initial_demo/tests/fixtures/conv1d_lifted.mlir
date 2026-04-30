// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: conv1d -> stablehlo.convolution_1d
// Description: O[n,w] = sum_kw I[n,w+kw] * K[kw]  — StableHLO 1-D convolution

module {

  func.func @lifted_conv1d(%I: tensor<1x1x8x1xf32>, %K: tensor<1x3x1x1xf32>) -> tensor<1x1x6x1xf32> {
    %result = stablehlo.convolution(%I, %K)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
                lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      : (tensor<1x1x8x1xf32>, tensor<1x3x1x1xf32>) -> tensor<1x1x6x1xf32>
    return %result : tensor<1x1x6x1xf32>
  }
}
