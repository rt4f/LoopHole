func.func @polygeist_if_guarded_store(%A: memref<16xf32>, %B: memref<16xf32>) {
  affine.for %i = 0 to 16 {
    affine.if #set(%i) {
      %v = affine.load %A[%i] : memref<16xf32>
      affine.store %v, %B[%i] : memref<16xf32>
    }
  }
  return
}
