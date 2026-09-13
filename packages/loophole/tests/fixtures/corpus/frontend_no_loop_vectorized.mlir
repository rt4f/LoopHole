func.func @frontend_no_loop_vectorized(%A: memref<16xf32>, %B: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %v = vector.transfer_read %A[%c0], %cst : memref<16xf32>, vector<16xf32>
  vector.transfer_write %v, %B[%c0] : vector<16xf32>, memref<16xf32>
  return
}
