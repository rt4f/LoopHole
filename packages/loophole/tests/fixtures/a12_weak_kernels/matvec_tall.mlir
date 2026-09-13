func.func @matvec_tall(%A: memref<8x3xf32>, %x: memref<3xf32>, %y: memref<8xf32>) {
  affine.for %m = 0 to 8 {
    affine.for %k = 0 to 3 {
      %a = affine.load %A[%m, %k] : memref<8x3xf32>
      %xk = affine.load %x[%k] : memref<3xf32>
      %ym = affine.load %y[%m] : memref<8xf32>
      %mul = arith.mulf %a, %xk : f32
      %add = arith.addf %ym, %mul : f32
      affine.store %add, %y[%m] : memref<8xf32>
    }
  }
  return
}