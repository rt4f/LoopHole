func.func @conv2d(%I: memref<6x6xf32>, %K: memref<3x3xf32>, %O: memref<4x4xf32>) {
  affine.for %oh = 0 to 4 {
    affine.for %ow = 0 to 4 {
      affine.for %kh = 0 to 3 {
        affine.for %kw = 0 to 3 {
          %i_val = affine.load %I[%oh + %kh, %ow + %kw] : memref<6x6xf32>
          %k_val = affine.load %K[%kh, %kw] : memref<3x3xf32>
          %o_val = affine.load %O[%oh, %ow] : memref<4x4xf32>
          %mul = arith.mulf %i_val, %k_val : f32
          %add = arith.addf %o_val, %mul : f32
          affine.store %add, %O[%oh, %ow] : memref<4x4xf32>
        }
      }
    }
  }
  return
}
