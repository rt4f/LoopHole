void matmul_2x2(float A[2][2], float B[2][2], float C[2][2]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < 2; ++k) {
                acc += A[i][k] * B[k][j];
            }
            C[i][j] = acc;
        }
    }
}
