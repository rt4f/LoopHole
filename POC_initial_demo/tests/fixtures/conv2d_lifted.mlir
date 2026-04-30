// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)
// Source: conv2d -> stablehlo.convolution_2d
// Description: O[oh,ow] = sum_{kh,kw} I[oh+kh,ow+kw] * K[kh,kw]  — StableHLO 2-D convolution

module {

  func.func @lifted_conv2d(%I: tensor<1x6x6x1xf32>, %K: tensor<3x3x1x1xf32>) -> tensor<1x4x4x1xf32> {
    %result = stablehlo.convolution(%I, %K)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
                lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      : (tensor<1x6x6x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x4x4x1xf32>
    return %result : tensor<1x4x4x1xf32>
  }
}
