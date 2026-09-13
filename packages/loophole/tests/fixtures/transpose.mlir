func.func @transpose(%A: memref<4x4xf32>, %B: memref<4x4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %a = affine.load %A[%i, %j] : memref<4x4xf32>
      affine.store %a, %B[%j, %i] : memref<4x4xf32>
    }
  }
  return
}
