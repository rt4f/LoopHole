// A-15 parametric pilot fixture: matrix-vector multiply with 4x4 shapes.
// _verify_parametric replaces the concrete bound 4 with symbolic M, K to prove
// equivalence for all valid shapes up to the unroll threshold.
func.func @matvec(%A: memref<4x4xf32>, %x: memref<4xf32>, %y: memref<4xf32>) {
  affine.for %m = 0 to 4 {
    affine.for %k = 0 to 4 {
      %a = affine.load %A[%m, %k] : memref<4x4xf32>
      %xi = affine.load %x[%k] : memref<4xf32>
      %yi = affine.load %y[%m] : memref<4xf32>
      %mul = arith.mulf %a, %xi : f32
      %add = arith.addf %yi, %mul : f32
      affine.store %add, %y[%m] : memref<4xf32>
    }
  }
  return
}
