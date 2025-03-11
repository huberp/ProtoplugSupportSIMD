#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

extern void add_vectors(const double* a, const double* b, double* result, size_t n);
extern void square_vector(const double* a, double* result, size_t n);
extern double* allocate_aligned_memory(size_t n);
extern void free_aligned_memory(double* ptr);
extern double* compute_rms_windowed(const double* input, size_t n, size_t window);
extern void compute_abs_ratio(const double* a, const double* b, double* result, size_t n);
extern void squared_difference(const double* a, const double* b, double* result, size_t n);
extern void compute_a_plus_bx(double a, double b, const double* x, double* result, size_t n);

void demo_add_vectors(size_t n) {
    double* a = allocate_aligned_memory(n);
    double* b = allocate_aligned_memory(n);
    double* result = allocate_aligned_memory(n);

    if (!a || !b || !result) {
        // Handle allocation failure
        return;
    }

    // Initialize a and b with some values
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)i;
        b[i] = (double)(n - i);
    }

    printf("ADD\n");
    // Call the add_vectors function
    add_vectors(a, b, result, n);

    // Output the result to the console
    for (size_t i = 0; i < n; ++i) {
        printf("result[%zu] = %f\n", i, result[i]);
    }

    free_aligned_memory(a);
    free_aligned_memory(b);
    free_aligned_memory(result);
}

void demo_square_vector(size_t n) {
    double* a = allocate_aligned_memory(n);
    double* result = allocate_aligned_memory(n);

    if (!a || !result) {
        // Handle allocation failure
        return;
    }

    // Initialize a with some values
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)i;
    }

    printf("\nSQUARE\n");
    // Call the square_vector function
    square_vector(a, result, n);

    // Output the result to the console
    for (size_t i = 0; i < n; ++i) {
        printf("result[%zu] = %f\n", i, result[i]);
    }

    free_aligned_memory(a);
    free_aligned_memory(result);
}

void demo_compute_rms_windowed(size_t n, size_t window) {
    double* a = allocate_aligned_memory(n);

    if (!a) {
        // Handle allocation failure
        return;
    }

    // Initialize a with some values
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)i;
    }

    printf("\nCOMPUTE RMS WINDOWED\n");
    // Call the compute_rms_windowed function
    double* rms_values = compute_rms_windowed(a, n, window);

    // Output the RMS values to the console
    size_t num_windows = (n + window - 1) / window;
    for (size_t i = 0; i < num_windows; ++i) {
        printf("RMS value for window %zu: %f\n", i, rms_values[i]);
    }

    free_aligned_memory(a);
    free_aligned_memory(rms_values);
}

void demo_compute_abs_ratio(size_t n) {
    double* a = allocate_aligned_memory(n);
    double* b = allocate_aligned_memory(n);
    double* result = allocate_aligned_memory(n);

    if (!a || !b || !result) {
        // Handle allocation failure
        return;
    }

    // Initialize a and b with some values
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)i + 1;
        b[i] = (double)(n - i) * ((i % 2 == 0) ? 1 : -1);
    }

    printf("\nCOMPUTE ABS RATIO\n");
    // Call the compute_abs_ratio function
    compute_abs_ratio(a, b, result, n);

    // Output the result to the console
    for (size_t i = 0; i < n; ++i) {
        printf("result[%zu] = %f\n", i, result[i]);
    }

    free_aligned_memory(a);
    free_aligned_memory(b);
    free_aligned_memory(result);
}

void demo_squared_difference(size_t n) {
    double* a = allocate_aligned_memory(n);
    double* b = allocate_aligned_memory(n);
    double* result = allocate_aligned_memory(n);

    if (!a || !b || !result) {
        // Handle allocation failure
        return;
    }

    // Initialize a and b with some values
    for (size_t i = 0; i < n; ++i) {
        a[i] = (double)i + 1;
        b[i] = (double)(n - i);
    }

    printf("\nSQUARED DIFFERENCE\n");
    // Call the squared_difference function
    squared_difference(a, b, result, n);

    // Output the result to the console
    for (size_t i = 0; i < n; ++i) {
        printf("result[%zu] = %f\n", i, result[i]);
    }

    free_aligned_memory(a);
    free_aligned_memory(b);
    free_aligned_memory(result);
}

void demo_compute_a_plus_bx(double a, double b, size_t n) {
    double* x = allocate_aligned_memory(n);
    double* result = allocate_aligned_memory(n);

    if (!x || !result) {
        // Handle allocation failure
        return;
    }

    // Initialize x with some values
    for (size_t i = 0; i < n; ++i) {
        x[i] = (double)i;
    }

    printf("\nCOMPUTE A + B * X\n");
    // Call the compute_a_plus_bx function
    compute_a_plus_bx(a, b, x, result, n);

    // Output the result to the console
    for (size_t i = 0; i < n; ++i) {
        printf("result[%zu] = %f\n", i, result[i]);
    }

    free_aligned_memory(x);
    free_aligned_memory(result);
}

int main() {
    size_t n = 64; // Example size
    size_t window = 12; // Example window size

    demo_add_vectors(n);
    demo_square_vector(n);
    demo_compute_rms_windowed(n, window);
    demo_compute_abs_ratio(n);
    demo_squared_difference(n);
    demo_compute_a_plus_bx(10000.0, 2.0, n);

    return 0;
}