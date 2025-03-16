
#include <xsimd/xsimd.hpp>
#include <cstddef>
#include <cmath>
#include <vector>

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

#if defined(SIMDE_X86_AVX2_NATIVE) 
  PRAGMA_MESSAGE("AVX2 supported.")
#else
  PRAGMA_WARNING("AVX2 support is not available. Code will not compile.")
#endif

extern "C" {

#define ALIGN 64

/**
 * Computes a + b for each element in the arrays a and b.
 * The result is stored in the output array.
 *
 * @param a The first input vector, aligned to ALIGN.
 * @param b The second input vector, aligned to ALIGN.
 * @param result The output vector, aligned to ALIGN.
 * @param n The number of elements in the input and output vectors.
 */
__declspec(dllexport) void add_vectors(const double* a, const double* b, double* result, size_t n) {
    const double* _a      = reinterpret_cast<const double*>(__builtin_assume_aligned(a, ALIGN));
    const double* _b      = reinterpret_cast<const double*>(__builtin_assume_aligned(b, ALIGN));
          double* _result = reinterpret_cast<double*>(__builtin_assume_aligned(result, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;
    printf("c add.1 simd_size=%d, %d\n", simd_size, _a);

    for (size_t i = 0; i < n; i += simd_size, _a += simd_size, _b += simd_size, _result += simd_size) {
        simd_type va = xsimd::load_aligned(_a);
        simd_type vb = xsimd::load_aligned(_b);
        simd_type vresult = va + vb;
        vresult.store_aligned(_result);
    }
    printf("c add.2 simd_size=%d, %d\n", simd_size, _a);
}

/**
 * Computes a - b for each element in the arrays a and b.
 * The result is stored in the output array.
 *
 * @param a The first input vector, aligned to ALIGN.
 * @param b The second input vector, aligned to ALIGN.
 * @param result The output vector, aligned to ALIGN.
 * @param n The number of elements in the input and output vectors.
 */
__declspec(dllexport) void sub_vectors(const double* a, const double* b, double* result, size_t n) {
    const double* _a      = reinterpret_cast<const double*>(__builtin_assume_aligned(a, ALIGN));
    const double* _b      = reinterpret_cast<const double*>(__builtin_assume_aligned(b, ALIGN));
          double* _result = reinterpret_cast<double*>(__builtin_assume_aligned(result, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    for (size_t i = 0; i < n; i += simd_size, _a += simd_size, _b += simd_size, _result += simd_size) {
        simd_type va = xsimd::load_aligned(_a);
        simd_type vb = xsimd::load_aligned(_b);
        simd_type vresult = va - vb;
        vresult.store_aligned(_result);
    }
}

/**
 * Computes a * b for each element in the arrays a and b.
 * The result is stored in the output array.
 *
 * @param a The first input vector, aligned to ALIGN.
 * @param b The second input vector, aligned to ALIGN.
 * @param result The output vector, aligned to ALIGN.
 * @param n The number of elements in the input and output vectors.
 */
__declspec(dllexport) void mul_vectors(const double* a, const double* b, double* result, size_t n) {
    const double* _a      = reinterpret_cast<const double*>(__builtin_assume_aligned(a, ALIGN));
    const double* _b      = reinterpret_cast<const double*>(__builtin_assume_aligned(b, ALIGN));
          double* _result = reinterpret_cast<double*>(__builtin_assume_aligned(result, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    for (size_t i = 0; i < n; i += simd_size, _a += simd_size, _b += simd_size, _result += simd_size) {
        simd_type va = xsimd::load_aligned(_a);
        simd_type vb = xsimd::load_aligned(_b);
        simd_type vresult = va * vb;
        vresult.store_aligned(_result);
    }
}

/**
 * Computes abs(abs(a + b) - abs(a) - abs(b)) for each element in the arrays a and b.
 * The result is stored in the output array.
 *
 * @param a The first input vector, aligned to ALIGN.
 * @param b The second input vector, aligned to ALIGN.
 * @param result The output vector, aligned to ALIGN.
 * @param n The number of elements in the input and output vectors.
 */
__declspec(dllexport) void compute_abs_diff_sum(const double* a, const double* b, double* result, size_t n) {
    const double* _a      = reinterpret_cast<const double*>(__builtin_assume_aligned(a, ALIGN));
    const double* _b      = reinterpret_cast<const double*>(__builtin_assume_aligned(b, ALIGN));
          double* _result = reinterpret_cast<double*>(__builtin_assume_aligned(result, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    for (size_t i = 0; i < n; i += simd_size, _a += simd_size, _b += simd_size, _result += simd_size) {
        simd_type va = xsimd::load_aligned(_a);
        simd_type vb = xsimd::load_aligned(_b);

        simd_type vsum = va + vb; // a + b
        simd_type vabs_sum = xsimd::abs(vsum); // abs(a + b)
        simd_type vabs_a = xsimd::abs(va); // abs(a)
        simd_type vabs_b = xsimd::abs(vb); // abs(b)

        simd_type vdiff = vabs_sum - vabs_a - vabs_b; // abs(a + b) - abs(a) - abs(b)
        simd_type vabs_diff = xsimd::abs(vdiff); // abs(abs(a + b) - abs(a) - abs(b))

        vabs_diff.store_aligned(_result);
    }
}

/**
 * Computes the square of each element in the input array.
 * The result is stored in the output array.
 *
 * @param input The input vector, aligned to ALIGN.
 * @param result The output vector, aligned to ALIGN.
 * @param n The number of elements in the input and output vectors.
 */
__declspec(dllexport) void square_vector(const double* input, double* result, size_t n) {
    const double* _input  = reinterpret_cast<const double*>(__builtin_assume_aligned(input, ALIGN));
          double* _result = reinterpret_cast<double*>(__builtin_assume_aligned(result, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    for (size_t i = 0; i < n; i += simd_size, _input += simd_size, _result += simd_size) {
        simd_type vinput = xsimd::load_aligned(_input);
        simd_type vresult = vinput * vinput;
        vresult.store_aligned(_result);
    }
}

/**
 * Computes the root mean square (RMS) of the input array.
 *
 * @param input The input vector, aligned to ALIGN.
 * @param n The number of elements in the input vector.
 * @return The RMS value.
 */
__declspec(dllexport) double compute_rms_full(const double* input, size_t n) {
    const double* _input = reinterpret_cast<const double*>(__builtin_assume_aligned(input, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    simd_type vsum = xsimd::batch<double>::broadcast(0.0);
    for (size_t i = 0; i < n; i += simd_size, _input += simd_size) {
        simd_type vinput = xsimd::load_aligned(_input);
        simd_type vsquare = vinput * vinput;
        vsum += vsquare;
    }

    std::vector<double> sum(simd_size);
    vsum.store_aligned(sum.data());
    double total_sum = 0.0;
    for (size_t i = 0; i < simd_size; ++i) {
        total_sum += sum[i];
    }

    return std::sqrt(total_sum / n);
}

/**
 * Computes the RMS value for each window in the input array.
 *
 * @param input The input vector, aligned to ALIGN.
 * @param n The number of elements in the input vector.
 * @param window The size of each window.
 * @return A pointer to the array of RMS values for each window.
 */
__declspec(dllexport) double* compute_rms_windowed(const double* input, size_t n, size_t window) {
    const double* _input = reinterpret_cast<const double*>(__builtin_assume_aligned(input, ALIGN));
    size_t num_windows = (n + window - 1) / window;
    double* rms_values = reinterpret_cast<double*>(__builtin_assume_aligned(_mm_malloc(num_windows * sizeof(double), ALIGN), ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    for (size_t i = 0; i < n; i += window) {
        simd_type vsum = xsimd::batch<double>::broadcast(0.0);
        size_t limit = (i + window > n) ? n - i : window;
        for (size_t j = 0; j < limit; j += simd_size) {
            simd_type vinput;
            if (j + simd_size <= limit) {
                vinput = xsimd::load_aligned(&_input[i + j]);
            } else {
                std::vector<double> temp(simd_size, 0.0);
                for (size_t k = 0; k < limit - j; ++k) {
                    temp[k] = _input[i + j + k];
                }
                vinput = xsimd::load_unaligned(temp.data());
            }
            simd_type vsquare = vinput * vinput;
            vsum += vsquare;
        }
        std::vector<double> sum(simd_size);
        vsum.store_aligned(sum.data());
        double total_sum = 0.0;
        for (size_t k = 0; k < simd_size; ++k) {
            total_sum += sum[k];
        }
        rms_values[i / window] = std::sqrt(total_sum / limit);
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
    const double* _a      = reinterpret_cast<const double*>(__builtin_assume_aligned(a, ALIGN));
    const double* _b      = reinterpret_cast<const double*>(__builtin_assume_aligned(b, ALIGN));
          double* _result = reinterpret_cast<double*>(__builtin_assume_aligned(result, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    for (size_t i = 0; i < n; i += simd_size, _a += simd_size, _b += simd_size, _result += simd_size) {
        simd_type va = xsimd::load_aligned(_a);
        simd_type vb = xsimd::load_aligned(_b);

        simd_type vabs_a = xsimd::abs(va); // abs(a)
        simd_type vabs_b = xsimd::abs(vb); // abs(b)
        simd_type vabs_sum = vabs_a + vabs_b; // abs(a) + abs(b)

        simd_type vsum = va + vb; // a + b
        simd_type vabs_sum_ab = xsimd::abs(vsum); // abs(a + b)

        simd_type vresult = vabs_sum_ab / vabs_sum; // abs(a + b) / (abs(a) + abs(b))
        vresult.store_aligned(_result);
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
    const double* _a      = reinterpret_cast<const double*>(__builtin_assume_aligned(a, ALIGN));
    const double* _b      = reinterpret_cast<const double*>(__builtin_assume_aligned(b, ALIGN));
          double* _result = reinterpret_cast<double*>(__builtin_assume_aligned(result, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    for (size_t i = 0; i < n; i += simd_size, _a += simd_size, _b += simd_size, _result += simd_size) {
        simd_type va = xsimd::load_aligned(_a);
        simd_type vb = xsimd::load_aligned(_b);
        simd_type vdiff = va - vb;
        simd_type vsquared_diff = vdiff * vdiff;
        vsquared_diff.store_aligned(_result);
    }
}

/**
 * Computes a + b * x for each element in the array x.
 * The result is stored in the output array.
 *
 * @param a The scalar value to be added.
 * @param b The scalar value to be multiplied with each element of x.
 * @param x The input array, aligned to ALIGN.
 * @param result The output array, aligned to ALIGN.
 * @param n The number of elements in the input and output arrays.
 */
__declspec(dllexport) void compute_a_plus_bx(double a, double b, const double* x, double* result, size_t n) {
    const double* _x      = reinterpret_cast<const double*>(__builtin_assume_aligned(x, ALIGN));
          double* _result = reinterpret_cast<double*>(__builtin_assume_aligned(result, ALIGN));

    using simd_type = xsimd::simd_type<double>;
    size_t simd_size = simd_type::size;

    simd_type va = xsimd::batch<double>::broadcast(a);
    simd_type vb = xsimd::batch<double>::broadcast(b);

    for (size_t i = 0; i < n; i += simd_size, _x += simd_size, _result += simd_size) {
        simd_type vx = xsimd::load_aligned(_x);
        simd_type vbx = vb * vx;
        simd_type vresult = va + vbx;
        vresult.store_aligned(_result);
    }
}

/**
 * Allocates aligned memory for a vector.
 *
 * @param n The number of elements in the vector.
 * @return A pointer to the allocated memory.
 */
__declspec(dllexport) double* allocate_aligned_memory(size_t n) {
    size_t padded_n = (n + 3) & ~3; // Ensure n is a multiple of 4 for AVX
    return reinterpret_cast<double*>(__builtin_assume_aligned(_mm_malloc(padded_n * sizeof(double), ALIGN), ALIGN));
}

/**
 * Frees the aligned memory allocated for a vector.
 *
 * @param ptr The pointer to the allocated memory.
 */
__declspec(dllexport) void free_aligned_memory(double* ptr) {
    _mm_free(reinterpret_cast<void*>(ptr));
}
} // extern "C"