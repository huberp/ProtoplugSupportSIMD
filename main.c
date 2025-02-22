#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

extern void add_vectors(const double* a, const double* b, double* result, size_t n);
extern void square_vector(const double* a, double* result, size_t n);
extern double* allocate_aligned_memory(size_t n);
extern void free_aligned_memory(double* ptr);
extern double* compute_rms_windowed(const double* input, size_t n, size_t window);

int main() {
    size_t n = 64; // Example size
    size_t window = 12; // Example window size
    size_t padded_n = (n + 3) & ~3; // Ensure n is a multiple of 4 for AVX

    double* a = allocate_aligned_memory(n);
    double* b = allocate_aligned_memory(n);
    double* result = allocate_aligned_memory(n);

    if (!a || !b || !result) {
        // Handle allocation failure
        return -1;
    }

    // Initialize a and b with some values
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)i;
        b[i] = (double)(n - i);
    }
    printf("ADD\n");
    // Call the add_vectors function
    add_vectors(a, b, result, padded_n);

    // Output the result to the console
    for (size_t i = 0; i < n; ++i) {
        printf("result[%zu] = %f\n", i, result[i]);
    }
    printf("\nSQUARE\n");
    square_vector(a, result, padded_n);

    // Output the result to the console
    for (size_t i = 0; i < n; ++i) {
        printf("result[%zu] = %f\n", i, result[i]);
    }

    printf("\nCOMPUTE RMS WINDOWED\n");
    // Call the compute_rms_windowed function
    double* rms_values = compute_rms_windowed(a, n, window);

    // Output the RMS values to the console
    size_t num_windows = (n + window - 1) / window;
    for (size_t i = 0; i < num_windows; ++i) {
        printf("RMS value for window %zu: %f\n", i, rms_values[i]);
    }

    // Free the allocated memory
    _mm_free(a);
    _mm_free(b);
    _mm_free(result);

    return 0;
}