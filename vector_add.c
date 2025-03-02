#include <stddef.h>
#include <math.h>

//https://learn.arm.com/learning-paths/cross-platform/intrinsics/simde/
#define SIMDE_ENABLE_NATIVE_ALIASES

//https://stackoverflow.com/questions/171435/portability-of-warning-preprocessor-directive
#ifdef __GNUC__
//from https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html
//Instead of put such pragma in code:
//#pragma GCC diagnostic ignored "-Wformat"
//use:
//PRAGMA_GCC(diagnostic ignored "-Wformat")
#define DO_PRAGMA(x) _Pragma (#x)
#define PRAGMA_GCC(x) DO_PRAGMA(GCC #x)

#define PRAGMA_MESSAGE(x) DO_PRAGMA(message #x)
#define PRAGMA_WARNING(x) DO_PRAGMA(warning #x)
#endif //__GNUC__
#ifdef _MSC_VER
/*
#define PRAGMA_OPTIMIZE_OFF __pragma(optimize("", off))
// These two lines are equivalent
#pragma optimize("", off)
PRAGMA_OPTIMIZE_OFF
*/
#define PRAGMA_GCC(x)
// https://support2.microsoft.com/kb/155196?wa=wsignin1.0
#define __STR2__(x) #x
#define __STR1__(x) __STR2__(x)
#define __PRAGMA_LOC__ __FILE__ "("__STR1__(__LINE__)") "
#define PRAGMA_WARNING(x) __pragma(message(__PRAGMA_LOC__ ": warning: " #x))
#define PRAGMA_MESSAGE(x) __pragma(message(__PRAGMA_LOC__ ": message : " #x))

#endif

#include "simde/check.h"
#include "simde/x86/avx.h"
#include "simde/simde-features.h"

#if defined(SIMDE_X86_AVX2_NATIVE) 
  PRAGMA_MESSAGE("AVX2 supported.")
#else
  PRAGMA_WARNING("AVX2 support is not available. Code will not compile.")
#endif

const int ALIGN = 64;

__declspec(dllexport) void add_vectors(const double* a, const double* b, double* result, size_t n) {
    const double* _a      = __builtin_assume_aligned(a, ALIGN);
    const double* _b      = __builtin_assume_aligned(b, ALIGN);
          double* _result = __builtin_assume_aligned(result, ALIGN);

    //printf("a: %p, b: %p, result: %p\n", _a, _b, result);
    //for (size_t i = 0; i < n; ++i) {  // Process in chunks of 4 doubles
    //    printf("DLL i[%d]: %f, %f\n", i, a[i], b[i]);
    //}
    for (size_t i = 0; i < n; i += 4, _a+=4, _b+=4, _result+=4) {
        const simde__m256d va = simde_mm256_load_pd(__builtin_assume_aligned(_a, ALIGN));
        const simde__m256d vb = simde_mm256_load_pd(__builtin_assume_aligned(_b, ALIGN));
        const simde__m256d vresult = simde_mm256_add_pd(va, vb);
        simde_mm256_storeu_pd(_result, vresult);
    }
}

__declspec(dllexport) void square_vector(const double* input, double* result, size_t n) {
    __builtin_assume_aligned(input, ALIGN);
    __builtin_assume_aligned(result, ALIGN);
    size_t i;
    for (i = 0; i < n; i += 4, input+=4) {
        const simde__m256d vinput  = simde_mm256_load_pd( __builtin_assume_aligned(input, ALIGN));
        const simde__m256d vresult = simde_mm256_mul_pd(vinput, vinput);
        _mm256_storeu_pd(&result[i], vresult);
    }
}


__declspec(dllexport) double compute_rms_full(const double* input, size_t n) {
    __builtin_assume_aligned(input, ALIGN);
    size_t i;
    simde__m256d vsum = simde_mm256_setzero_pd();
    for (i = 0; i < n; i += 4, input+=4) {
        const simde__m256d vinput  = simde_mm256_load_pd( __builtin_assume_aligned(input, ALIGN));
        const simde__m256d vsquare = simde_mm256_mul_pd(vinput, vinput);
        vsum = _mm256_add_pd(vsum, vsquare);
    }
    double sum[4];
    simde_mm256_storeu_pd(sum, vsum);
    double total_sum = sum[0] + sum[1] + sum[2] + sum[3];
    return sqrt(total_sum / n);
}


__declspec(dllexport) double* compute_rms_windowed(const double* input, size_t n, size_t window) {
    __builtin_assume_aligned(input, ALIGN);
    size_t num_windows = (n + window - 1) / window;
    double* rms_values = (double*)_mm_malloc(num_windows * sizeof(double), ALIGN);
    size_t i, j;
    for (i = 0; i < n; i += window) {
        simde__m256d vsum = simde_mm256_setzero_pd();
        size_t limit = (i + window > n) ? n - i : window;
        for (j = 0; j < limit; j += 4) {
            __m256d vinput;
            if (j + 4 <= limit) {
                vinput = simde_mm256_load_pd(&input[i + j]);
            } else {
                double temp[4] = {0, 0, 0, 0};
                for (size_t k = 0; k < limit - j; ++k) {
                    temp[k] = input[i + j + k];
                }
                vinput = simde_mm256_loadu_pd(temp);
            }
            simde__m256d vsquare = simde_mm256_mul_pd(vinput, vinput);
            vsum = simde_mm256_add_pd(vsum, vsquare);
        }
        double sum[4];
        simde_mm256_storeu_pd(sum, vsum);
        double total_sum = sum[0] + sum[1] + sum[2] + sum[3];
        rms_values[i / window] = sqrt(total_sum / limit);
    }
    return rms_values;
}

/**
 * Computes the ratio of the absolute value of the sum of two vectors to the sum of their absolute values.
 * The result is stored in the output array.
 *
 * @param a The first input vector, aligned to ALIGN.
 * @param b The second input vector, aligned to ALIGN.
 * @param result The output vector, aligned to ALIGN.
 * @param n The number of elements in the input and output vectors.
 */
__declspec(dllexport) void compute_abs_ratio(const double* a, const double* b, double* result, size_t n) {
    const double* _a      = __builtin_assume_aligned(a, ALIGN);
    const double* _b      = __builtin_assume_aligned(b, ALIGN);
          double* _result = __builtin_assume_aligned(result, ALIGN);

    for (size_t i = 0; i < n; i += 4, _a += 4, _b += 4, _result += 4) {
        const simde__m256d va = simde_mm256_load_pd(_a);
        const simde__m256d vb = simde_mm256_load_pd(_b);

        const simde__m256d vabs_a = simde_mm256_andnot_pd(simde_mm256_set1_pd(-0.0), va); // abs(a)
        const simde__m256d vabs_b = simde_mm256_andnot_pd(simde_mm256_set1_pd(-0.0), vb); // abs(b)
        const simde__m256d vabs_sum = simde_mm256_add_pd(vabs_a, vabs_b); // abs(a) + abs(b)

        const simde__m256d vsum = simde_mm256_add_pd(va, vb); // a + b
        const simde__m256d vabs_sum_ab = simde_mm256_andnot_pd(simde_mm256_set1_pd(-0.0), vsum); // abs(a + b)

        const simde__m256d vresult = simde_mm256_div_pd(vabs_sum_ab, vabs_sum); // abs(a + b) / (abs(a) + abs(b))
        simde_mm256_storeu_pd(_result, vresult);
    }
}

/**
 * Computes the squared difference of two vectors.
 * The result is stored in the output array.
 *
 * @param a The first input vector, aligned to ALIGN.
 * @param b The second input vector, aligned to ALIGN.
 * @param result The output vector, aligned to ALIGN.
 * @param n The number of elements in the input and output vectors.
 */
__declspec(dllexport) void squared_difference(const double* a, const double* b, double* result, size_t n) {
    const double* _a      = __builtin_assume_aligned(a, ALIGN);
    const double* _b      = __builtin_assume_aligned(b, ALIGN);
          double* _result = __builtin_assume_aligned(result, ALIGN);

    for (size_t i = 0; i < n; i += 4, _a += 4, _b += 4, _result += 4) {
        const simde__m256d va = simde_mm256_load_pd(_a);
        const simde__m256d vb = simde_mm256_load_pd(_b);
        const simde__m256d vdiff = simde_mm256_sub_pd(va, vb);
        const simde__m256d vsquared_diff = simde_mm256_mul_pd(vdiff, vdiff);
        simde_mm256_storeu_pd(_result, vsquared_diff);
    }
}


__declspec(dllexport) double* allocate_aligned_memory(size_t n) {
    size_t padded_n = (n + 3) & ~3; // Ensure n is a multiple of 4 for AVX
    return __builtin_assume_aligned( (double*)_mm_malloc(padded_n * sizeof(double), ALIGN), ALIGN );
}

__declspec(dllexport) void free_aligned_memory(double* ptr) {
    _mm_free(ptr);
}