#include <stddef.h>
void kernel(float A[8][8], float B[8][8], float C[8][8]) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++){
        C[i][j] += A[i][j] * B[i][j];
      }
    }
  }