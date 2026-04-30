void matmul_inplace_128(float A[128][128], float B[128][128], float C[128][128]) {
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 128; ++j) {
            for (int k = 0; k < 128; ++k) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
}
