func.func @conv1d(%I: memref<1x8xf32>, %K: memref<3xf32>, %O: memref<1x6xf32>) {
  affine.for %n = 0 to 1 {
    affine.for %w = 0 to 6 {
      affine.for %kw = 0 to 3 {
        %i_val = affine.load %I[%n, %w + %kw] : memref<1x8xf32>
        %k_val = affine.load %K[%kw] : memref<3xf32>
        %o_val = affine.load %O[%n, %w] : memref<1x6xf32>
        %mul = arith.mulf %i_val, %k_val : f32
        %add = arith.addf %o_val, %mul : f32
        affine.store %add, %O[%n, %w] : memref<1x6xf32>
      }
    }
  }
  return
}