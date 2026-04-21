func.func @polygeist_conv1d_scf(%I: memref<1x16xf32>, %K: memref<3xf32>, %O: memref<1x14xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c14 = arith.constant 14 : index
  %c3 = arith.constant 3 : index
  scf.for %w = %c0 to %c14 step %c1 {
    scf.for %kw = %c0 to %c3 step %c1 {
      %i = memref.load %I[%c0, %w + %kw] : memref<1x16xf32>
      %k = memref.load %K[%kw] : memref<3xf32>
      %o = memref.load %O[%c0, %w] : memref<1x14xf32>
      %m = arith.mulf %i, %k : f32
      %s = arith.addf %o, %m : f32
      memref.store %s, %O[%c0, %w] : memref<1x14xf32>
    }
  }
  return
}
