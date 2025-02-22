#include <immintrin.h>
#include <stddef.h>
#include <math.h>

const int ALIGN = 64;

__declspec(dllexport) void add_vectors(const double* a, const double* b, double* result, size_t n) {
    __builtin_assume_aligned(a, ALIGN);
    __builtin_assume_aligned(b, ALIGN);
    __builtin_assume_aligned(result, ALIGN);
    size_t i;
    for (i = 0; i < n; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
}

__declspec(dllexport) void square_vector(const double* input, double* result, size_t n) {
    __builtin_assume_aligned(input, ALIGN);
    __builtin_assume_aligned(result, ALIGN);
    size_t i;
    for (i = 0; i < n; i += 4) {
        __m256d vinput  = _mm256_load_pd(&input[i]);
        __m256d vresult = _mm256_mul_pd(vinput, vinput);
        _mm256_storeu_pd(&result[i], vresult);
    }
}


__declspec(dllexport) double compute_rms_full(const double* input, size_t n) {
    __builtin_assume_aligned(input, ALIGN);
    size_t i;
    __m256d vsum = _mm256_setzero_pd();
    for (i = 0; i < n; i += 4) {
        __m256d vinput = _mm256_load_pd(&input[i]);
        __m256d vsquare = _mm256_mul_pd(vinput, vinput);
        vsum = _mm256_add_pd(vsum, vsquare);
    }
    double sum[4];
    _mm256_storeu_pd(sum, vsum);
    double total_sum = sum[0] + sum[1] + sum[2] + sum[3];
    return sqrt(total_sum / n);
}


__declspec(dllexport) double* compute_rms_windowed(const double* input, size_t n, size_t window) {
    __builtin_assume_aligned(input, ALIGN);
    size_t num_windows = (n + window - 1) / window;
    double* rms_values = (double*)_mm_malloc(num_windows * sizeof(double), ALIGN);
    size_t i, j;
    for (i = 0; i < n; i += window) {
        __m256d vsum = _mm256_setzero_pd();
        size_t limit = (i + window > n) ? n - i : window;
        for (j = 0; j < limit; j += 4) {
            __m256d vinput;
            if (j + 4 <= limit) {
                vinput = _mm256_load_pd(&input[i + j]);
            } else {
                double temp[4] = {0, 0, 0, 0};
                for (size_t k = 0; k < limit - j; ++k) {
                    temp[k] = input[i + j + k];
                }
                vinput = _mm256_loadu_pd(temp);
            }
            __m256d vsquare = _mm256_mul_pd(vinput, vinput);
            vsum = _mm256_add_pd(vsum, vsquare);
        }
        double sum[4];
        _mm256_storeu_pd(sum, vsum);
        double total_sum = sum[0] + sum[1] + sum[2] + sum[3];
        rms_values[i / window] = sqrt(total_sum / limit);
    }
    return rms_values;
}


__declspec(dllexport) double* allocate_aligned_memory(size_t n) {
    size_t padded_n = (n + 3) & ~3; // Ensure n is a multiple of 4 for AVX
    return __builtin_assume_aligned( (double*)_mm_malloc(padded_n * sizeof(double), ALIGN), ALIGN );
}

__declspec(dllexport) void free_aligned_memory(double* ptr) {
    _mm_free(ptr);
}