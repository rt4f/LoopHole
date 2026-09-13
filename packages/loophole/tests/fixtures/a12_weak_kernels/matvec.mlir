func.func @matvec(%A: memref<4x4xf32>, %x: memref<4xf32>, %y: memref<4xf32>) {
  affine.for %m = 0 to 4 {
    affine.for %k = 0 to 4 {
      %a = affine.load %A[%m, %k] : memref<4x4xf32>
      %xk = affine.load %x[%k] : memref<4xf32>
      %ym = affine.load %y[%m] : memref<4xf32>
      %mul = arith.mulf %a, %xk : f32
      %add = arith.addf %ym, %mul : f32
      affine.store %add, %y[%m] : memref<4xf32>
    }
  }
  return
}