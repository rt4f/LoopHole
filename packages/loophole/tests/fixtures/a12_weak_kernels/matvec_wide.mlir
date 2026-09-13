func.func @matvec_wide(%A: memref<3x8xf32>, %x: memref<8xf32>, %y: memref<3xf32>) {
  affine.for %m = 0 to 3 {
    affine.for %k = 0 to 8 {
      %a = affine.load %A[%m, %k] : memref<3x8xf32>
      %xk = affine.load %x[%k] : memref<8xf32>
      %ym = affine.load %y[%m] : memref<3xf32>
      %mul = arith.mulf %a, %xk : f32
      %add = arith.addf %ym, %mul : f32
      affine.store %add, %y[%m] : memref<3xf32>
    }
  }
  return
}