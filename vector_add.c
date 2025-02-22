#include <immintrin.h>
#include <stddef.h>

__declspec(dllexport) void add_vectors(const double* a, const double* b, double* result, size_t n) {
    size_t i;
    for (i = 0; i < n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
}

__declspec(dllexport) void square_vector(const double* input, double* result, size_t n) {
    size_t i;
    for (i = 0; i < n; i += 4) {
        __m256d vinput = _mm256_loadu_pd(&input[i]);
        __m256d vresult = _mm256_mul_pd(vinput, vinput);
        _mm256_storeu_pd(&result[i], vresult);
    }
}

__declspec(dllexport) double* allocate_aligned_memory(size_t n) {
    size_t padded_n = (n + 3) & ~3; // Ensure n is a multiple of 4 for AVX
    return (double*)_mm_malloc(padded_n * sizeof(double), 32);
}

__declspec(dllexport) void free_aligned_memory(double* ptr) {
    _mm_free(ptr);
}