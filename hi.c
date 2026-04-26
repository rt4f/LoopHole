/*
 * Keep the lifted target as a pure compute kernel so cgeist/Polygeist
 * can lower it without vararg libc call issues.
 */
void dot_product(const float a[8], const float b[8], float c[1]) {
    for (int i = 0; i < 8; ++i) {
        c[0] += a[i] * b[i];
    }
}

int main(void) {
    float a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    float c[1] = {0.0f};

    dot_product(a, b, c);
    return c[0] > 0.0f ? 0 : 1;
}
