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
    __builtin_assume_aligned(a, ALIGN);
    __builtin_assume_aligned(b, ALIGN);
    __builtin_assume_aligned(result, ALIGN);
    size_t i;
    for (i = 0; i < n; i += 4, a+=4, b+=4) {
        const simde__m256d va = simde_mm256_load_pd(__builtin_assume_aligned(a, ALIGN));
        const simde__m256d vb = simde_mm256_load_pd(__builtin_assume_aligned(b, ALIGN));
        const simde__m256d vresult = simde_mm256_add_pd(va, vb);
        simde_mm256_storeu_pd(&result[i], vresult);
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


__declspec(dllexport) double* allocate_aligned_memory(size_t n) {
    size_t padded_n = (n + 3) & ~3; // Ensure n is a multiple of 4 for AVX
    return __builtin_assume_aligned( (double*)_mm_malloc(padded_n * sizeof(double), ALIGN), ALIGN );
}

__declspec(dllexport) void free_aligned_memory(double* ptr) {
    _mm_free(ptr);
}