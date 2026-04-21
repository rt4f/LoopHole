#include <stdio.h>

/* Function that squares an integer */
int square(int x) {
    return x * x;
}

/* Main program: loops from 0 to 4, prints the square of each i */
int main(void) {
    for (int i = 0; i < 5; ++i) {
        int result = square(i);          // call the function
        printf("i = %d, i^2 = %d\n", i, result);
    }
    return 0;    // program finished successfully
}
