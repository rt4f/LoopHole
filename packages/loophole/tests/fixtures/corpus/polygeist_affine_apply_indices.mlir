func.func @polygeist_affine_apply_indices(%A: memref<32xf32>, %B: memref<32xf32>) {
  affine.for %i = 0 to 16 {
    %j = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
    %v = affine.load %A[%j] : memref<32xf32>
    affine.store %v, %B[%j] : memref<32xf32>
  }
  return
}
