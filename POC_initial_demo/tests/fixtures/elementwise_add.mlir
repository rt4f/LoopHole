func.func @elementwise_add(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %a = affine.load %A[%i, %j] : memref<4x4xf32>
      %b = affine.load %B[%i, %j] : memref<4x4xf32>
      %add = arith.addf %a, %b : f32
      affine.store %add, %C[%i, %j] : memref<4x4xf32>
    }
  }
  return
}
