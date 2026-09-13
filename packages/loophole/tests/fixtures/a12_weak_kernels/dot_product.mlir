func.func @dot(%A: memref<8xf32>, %B: memref<8xf32>, %c: memref<1xf32>) {
  affine.for %k = 0 to 8 {
    %a = affine.load %A[%k] : memref<8xf32>
    %b = affine.load %B[%k] : memref<8xf32>
    %c_val = affine.load %c[0] : memref<1xf32>
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c_val, %mul : f32
    affine.store %add, %c[0] : memref<1xf32>
  }
  return
}