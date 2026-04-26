// A-15 parametric pilot fixture: row-wise reduction sum with 4x4 shapes.
// _verify_parametric replaces the concrete bound 4 with symbolic M, N to prove
// equivalence for all valid shapes up to the unroll threshold.
func.func @reduce_sum(%A: memref<4x4xf32>, %B: memref<4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %a = affine.load %A[%i, %j] : memref<4x4xf32>
      %b = affine.load %B[%i] : memref<4xf32>
      %add = arith.addf %b, %a : f32
      affine.store %add, %B[%i] : memref<4xf32>
    }
  }
  return
}
