func.func @polygeist_dot_iter_args(%A: memref<16xf32>, %B: memref<16xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0.0 : f32
  %sum = scf.for %k = %c0 to %c16 step %c1 iter_args(%acc = %z) -> (f32) {
    %a = memref.load %A[%k] : memref<16xf32>
    %b = memref.load %B[%k] : memref<16xf32>
    %m = arith.mulf %a, %b : f32
    %n = arith.addf %acc, %m : f32
    scf.yield %n : f32
  }
  return %sum : f32
}
