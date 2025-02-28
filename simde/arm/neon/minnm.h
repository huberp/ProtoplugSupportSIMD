/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *   2020      Evan Nemerson <evan@nemerson.com>
 *   2023      Yi-Yen Chung <eric681@andestech.com> (Copyright owned by Andes Technology)
 */

#if !defined(SIMDE_ARM_NEON_MINNM_H)
#define SIMDE_ARM_NEON_MINNM_H

#include "types.h"
#include "cle.h"
#include "bsl.h"

HEDLEY_DIAGNOSTIC_PUSH
SIMDE_DISABLE_UNWANTED_DIAGNOSTICS
SIMDE_BEGIN_DECLS_

SIMDE_FUNCTION_ATTRIBUTES
simde_float16_t
simde_vminnmh_f16(simde_float16_t a, simde_float16_t b) {
  #if defined(SIMDE_ARM_NEON_A32V8_NATIVE) && defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6) && defined(SIMDE_ARM_NEON_FP16)
    return vminnmh_f16(a, b);
  #else
    #if defined(simde_math_fminf)
      return simde_float16_from_float32(simde_math_fminf(simde_float16_to_float32(a), simde_float16_to_float32(b)));
    #else
      simde_float32_t a_ = simde_float16_to_float32(a);
      simde_float32_t b_ = simde_float16_to_float32(b);
      simde_float32_t r_;
      if (a_ < b_) {
        r_ = a_;
      } else if (a_ > b_) {
        r_ = b_;
      } else if (a_ == a_) {
        r_ = a_;
      } else {
        r_ = b_;
      }
      return simde_float16_from_float32(r_);
    #endif
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(SIMDE_ENABLE_NATIVE_ALIASES) && \
  !(defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6) && defined(SIMDE_ARM_NEON_FP16)))
  #undef vminnmh_f16
  #define vminnmh_f16(a, b) simde_vminnmh_f16((a), (b))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float16x4_t
simde_vminnm_f16(simde_float16x4_t a, simde_float16x4_t b) {
  #if defined(SIMDE_ARM_NEON_A32V8_NATIVE) && defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6) && defined(SIMDE_ARM_NEON_FP16)
    return vminnm_f16(a, b);
  #else
    simde_float16x4_private
      r_,
      a_ = simde_float16x4_to_private(a),
      b_ = simde_float16x4_to_private(b);

    SIMDE_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = simde_vminnmh_f16(a_.values[i], b_.values[i]);
    }

    return simde_float16x4_from_private(r_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(SIMDE_ENABLE_NATIVE_ALIASES) && \
  !(defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6) && defined(SIMDE_ARM_NEON_FP16)))
  #undef vminnm_f16
  #define vminnm_f16(a, b) simde_vminnm_f16((a), (b))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float32x2_t
simde_vminnm_f32(simde_float32x2_t a, simde_float32x2_t b) {
  #if defined(SIMDE_ARM_NEON_A32V8_NATIVE) && defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6)
    return vminnm_f32(a, b);
  #else
    simde_float32x2_private
      r_,
      a_ = simde_float32x2_to_private(a),
      b_ = simde_float32x2_to_private(b);

    SIMDE_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      #if defined(simde_math_fminf)
        r_.values[i] = simde_math_fminf(a_.values[i], b_.values[i]);
      #else
        if (a_.values[i] < b_.values[i]) {
          r_.values[i] = a_.values[i];
        } else if (a_.values[i] > b_.values[i]) {
          r_.values[i] = b_.values[i];
        } else if (a_.values[i] == a_.values[i]) {
          r_.values[i] = a_.values[i];
        } else {
          r_.values[i] = b_.values[i];
        }
      #endif
    }

    return simde_float32x2_from_private(r_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(SIMDE_ENABLE_NATIVE_ALIASES) && \
  !(defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6)))
  #undef vminnm_f32
  #define vminnm_f32(a, b) simde_vminnm_f32((a), (b))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float64x1_t
simde_vminnm_f64(simde_float64x1_t a, simde_float64x1_t b) {
  #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
    return vminnm_f64(a, b);
  #else
    simde_float64x1_private
      r_,
      a_ = simde_float64x1_to_private(a),
      b_ = simde_float64x1_to_private(b);

    SIMDE_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      #if defined(simde_math_fmin)
        r_.values[i] = simde_math_fmin(a_.values[i], b_.values[i]);
      #else
        if (a_.values[i] < b_.values[i]) {
          r_.values[i] = a_.values[i];
        } else if (a_.values[i] > b_.values[i]) {
          r_.values[i] = b_.values[i];
        } else if (a_.values[i] == a_.values[i]) {
          r_.values[i] = a_.values[i];
        } else {
          r_.values[i] = b_.values[i];
        }
      #endif
    }

    return simde_float64x1_from_private(r_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminnm_f64
  #define vminnm_f64(a, b) simde_vminnm_f64((a), (b))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float16x8_t
simde_vminnmq_f16(simde_float16x8_t a, simde_float16x8_t b) {
  #if defined(SIMDE_ARM_NEON_A32V8_NATIVE) && defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6) && defined(SIMDE_ARM_NEON_FP16)
    return vminnmq_f16(a, b);
  #else
    simde_float16x8_private
      r_,
      a_ = simde_float16x8_to_private(a),
      b_ = simde_float16x8_to_private(b);

    SIMDE_VECTORIZE
    for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
      r_.values[i] = simde_vminnmh_f16(a_.values[i], b_.values[i]);
    }

    return simde_float16x8_from_private(r_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(SIMDE_ENABLE_NATIVE_ALIASES) && \
  !(defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6) && defined(SIMDE_ARM_NEON_FP16)))
  #undef vminnmq_f16
  #define vminnmq_f16(a, b) simde_vminnmq_f16((a), (b))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float32x4_t
simde_vminnmq_f32(simde_float32x4_t a, simde_float32x4_t b) {
  #if defined(SIMDE_ARM_NEON_A32V8_NATIVE) && defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6)
    return vminnmq_f32(a, b);
  #elif SIMDE_NATURAL_VECTOR_SIZE_GE(128) && defined(SIMDE_FAST_NANS)
    return simde_vbslq_f32(simde_vcleq_f32(a, b), a, b);
  #elif defined(SIMDE_POWER_ALTIVEC_P6_NATIVE)
    return vec_min(a, b);
  #else
    simde_float32x4_private
      r_,
      a_ = simde_float32x4_to_private(a),
      b_ = simde_float32x4_to_private(b);

    #if defined(SIMDE_X86_SSE_NATIVE)
      #if !defined(SIMDE_FAST_NANS)
        __m128 r = _mm_min_ps(a_.m128, b_.m128);
        __m128 bnan = _mm_cmpunord_ps(b_.m128, b_.m128);
        r = _mm_andnot_ps(bnan, r);
        r_.m128 = _mm_or_ps(r, _mm_and_ps(a_.m128, bnan));
      #else
        r_.m128 = _mm_min_ps(a_.m128, b_.m128);
      #endif
    #elif defined(SIMDE_WASM_SIMD128_NATIVE) && defined(SIMDE_FAST_NANS)
      r_.v128 = wasm_f32x4_min(a_.v128, b_.v128);
    #else
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        #if defined(simde_math_fminf)
          r_.values[i] = simde_math_fminf(a_.values[i], b_.values[i]);
        #else
          if (a_.values[i] < b_.values[i]) {
            r_.values[i] = a_.values[i];
          } else if (a_.values[i] > b_.values[i]) {
            r_.values[i] = b_.values[i];
          } else if (a_.values[i] == a_.values[i]) {
            r_.values[i] = a_.values[i];
          } else {
            r_.values[i] = b_.values[i];
          }
        #endif
      }
    #endif

    return simde_float32x4_from_private(r_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A32V8_ENABLE_NATIVE_ALIASES) || (defined(SIMDE_ENABLE_NATIVE_ALIASES) && \
  !(defined(__ARM_NEON_FP) && (__ARM_NEON_FP >= 6)))
  #undef vminnmq_f32
  #define vminnmq_f32(a, b) simde_vminnmq_f32((a), (b))
#endif

SIMDE_FUNCTION_ATTRIBUTES
simde_float64x2_t
simde_vminnmq_f64(simde_float64x2_t a, simde_float64x2_t b) {
  #if defined(SIMDE_ARM_NEON_A64V8_NATIVE)
    return vminnmq_f64(a, b);
  #elif SIMDE_NATURAL_VECTOR_SIZE_GE(128) && defined(SIMDE_FAST_NANS)
    return simde_vbslq_f64(simde_vcleq_f64(a, b), a, b);
  #elif defined(SIMDE_POWER_ALTIVEC_P7_NATIVE)
    return vec_min(a, b);
  #else
    simde_float64x2_private
      r_,
      a_ = simde_float64x2_to_private(a),
      b_ = simde_float64x2_to_private(b);

    #if defined(SIMDE_X86_SSE2_NATIVE)
      #if !defined(SIMDE_FAST_NANS)
        __m128d r = _mm_min_pd(a_.m128d, b_.m128d);
        __m128d bnan = _mm_cmpunord_pd(b_.m128d, b_.m128d);
        r = _mm_andnot_pd(bnan, r);
        r_.m128d = _mm_or_pd(r, _mm_and_pd(a_.m128d, bnan));
      #else
        r_.m128d = _mm_min_pd(a_.m128d, b_.m128d);
      #endif
    #elif defined(SIMDE_WASM_SIMD128_NATIVE) && defined(SIMDE_FAST_NANS)
      r_.v128 = wasm_f64x2_min(a_.v128, b_.v128);
    #else
      SIMDE_VECTORIZE
      for (size_t i = 0 ; i < (sizeof(r_.values) / sizeof(r_.values[0])) ; i++) {
        #if defined(simde_math_fmin)
          r_.values[i] = simde_math_fmin(a_.values[i], b_.values[i]);
        #else
          if (a_.values[i] < b_.values[i]) {
            r_.values[i] = a_.values[i];
          } else if (a_.values[i] > b_.values[i]) {
            r_.values[i] = b_.values[i];
          } else if (a_.values[i] == a_.values[i]) {
            r_.values[i] = a_.values[i];
          } else {
            r_.values[i] = b_.values[i];
          }
        #endif
      }
    #endif

    return simde_float64x2_from_private(r_);
  #endif
}
#if defined(SIMDE_ARM_NEON_A64V8_ENABLE_NATIVE_ALIASES)
  #undef vminnmq_f64
  #define vminnmq_f64(a, b) simde_vminnmq_f64((a), (b))
#endif

SIMDE_END_DECLS_
HEDLEY_DIAGNOSTIC_POP

#endif /* !defined(SIMDE_ARM_NEON_MINNM_H) */
