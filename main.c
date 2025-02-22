#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

extern void add_vectors(const double* a, const double* b, double* result, size_t n);
extern void square_vector(const double* a, double* result, size_t n);
extern double* allocate_aligned_memory(size_t n);
extern void free_aligned_memory(double* ptr);

int main() {
    size_t n = 64; // Example size
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


    // Free the allocated memory
    _mm_free(a);
    _mm_free(b);
    _mm_free(result);

    return 0;
}