#pragma once

#include "addadd_cpu.h"
#include "mulmul_cpu.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace ortops {

inline bool cpu_supports_avx2() {
#if defined(__x86_64__) || defined(__i386) || defined(_M_X64) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
  int regs[4] = {0, 0, 0, 0};
  __cpuidex(regs, 0, 0);
  if (regs[0] < 7)
    return false;
  __cpuidex(regs, 7, 0);
  return (regs[1] & (1 << 5)) != 0;
#else
  return false;
#endif
#else
  return false;
#endif
}

inline bool cpu_supports_f16c() {
#if defined(__x86_64__) || defined(__i386) || defined(_M_X64) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("f16c");
#elif defined(_MSC_VER)
  int regs[4] = {0, 0, 0, 0};
  __cpuidex(regs, 1, 0);
  return (regs[2] & (1 << 29)) != 0;
#else
  return false;
#endif
#else
  return false;
#endif
}

/**
 * @brief Applies @p fn(begin, end) over @c [0, N) in parallel using
 * std::thread.  Each thread receives a contiguous sub-range.  All threads are
 * joined before the function returns.
 */
static inline void parallel_for_addadd(int64_t N,
                                        const std::function<void(int64_t, int64_t)> &fn) {
  if (N <= 0)
    return;
  const int64_t n_threads =
      std::max(int64_t{1}, static_cast<int64_t>(std::thread::hardware_concurrency()));
  const int64_t chunk = (N + n_threads - 1) / n_threads;

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(n_threads));
  for (int64_t t = 0; t < n_threads; ++t) {
    const int64_t begin = t * chunk;
    if (begin >= N)
      break;
    const int64_t end = std::min(begin + chunk, N);
    threads.emplace_back(fn, begin, end);
  }
  for (auto &th : threads)
    th.join();
}

template <typename T>
inline T add3_scalar(const T &a, const T &b, const T &c) {
  if constexpr (std::is_same_v<T, float>) {
    return a + b + c;
  } else {
    return T(static_cast<float>(a) + static_cast<float>(b) + static_cast<float>(c));
  }
}

template <typename T>
inline Ort::Status ComputeAddAddCpuImpl(const Ort::Custom::Tensor<T> &A,
                                        const Ort::Custom::Tensor<T> &B,
                                        const Ort::Custom::Tensor<T> &C,
                                        Ort::Custom::Tensor<T> &output) {
  const std::vector<int64_t> shapeA = A.Shape();
  const std::vector<int64_t> shapeB = B.Shape();
  const std::vector<int64_t> shapeC = C.Shape();

  int64_t nA = 1;
  for (int64_t d : shapeA)
    nA *= d;
  int64_t nB = 1;
  for (int64_t d : shapeB)
    nB *= d;
  int64_t nC = 1;
  for (int64_t d : shapeC)
    nC *= d;

  if (nA == 0 || nB == 0 || nC == 0) {
    return Ort::Status(
        (std::string("AddAdd: all inputs must be non-empty (got nA=") + std::to_string(nA) +
         ", nB=" + std::to_string(nB) + ", nC=" + std::to_string(nC) + ")")
            .c_str(),
        OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  const int64_t N = std::max(nA, std::max(nB, nC));

  const std::size_t max_rank = std::max(shapeA.size(), std::max(shapeB.size(), shapeC.size()));
  std::vector<int64_t> out_shape(max_rank);
  {
    auto pad = [&](const std::vector<int64_t> &s) -> std::vector<int64_t> {
      std::vector<int64_t> p(max_rank - s.size(), 1);
      p.insert(p.end(), s.begin(), s.end());
      return p;
    };
    const auto pA = pad(shapeA);
    const auto pB = pad(shapeB);
    const auto pC = pad(shapeC);
    for (std::size_t i = 0; i < max_rank; ++i)
      out_shape[i] = std::max(std::max(pA[i], pB[i]), pC[i]);
  }

  T *out = output.Allocate(out_shape);
  const T *pA = A.Data();
  const T *pB = B.Data();
  const T *pC = C.Data();

  if (nA == N && nB == N && nC == N) {
    static const bool has_avx2 = cpu_supports_avx2();
    static const bool has_f16c = cpu_supports_f16c();
    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX2__
      if (has_avx2) {
        parallel_for_addadd(N, [&](int64_t begin, int64_t end) {
          int64_t i = begin;
          const int64_t vec_end = end - ((end - begin) % 8);
          for (; i < vec_end; i += 8) {
            const __m256 va = _mm256_loadu_ps(pA + i);
            const __m256 vb = _mm256_loadu_ps(pB + i);
            const __m256 vc = _mm256_loadu_ps(pC + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_add_ps(va, vb), vc));
          }
          for (; i < end; ++i)
            out[i] = pA[i] + pB[i] + pC[i];
        });
        return Ort::Status{nullptr};
      }
#endif
      parallel_for_addadd(N, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i)
          out[i] = pA[i] + pB[i] + pC[i];
      });
    } else if constexpr (std::is_same_v<T, Float16>) {
#if defined(__AVX2__) && defined(__F16C__)
      if (has_avx2 && has_f16c) {
        parallel_for_addadd(N, [&](int64_t begin, int64_t end) {
          int64_t i = begin;
          const int64_t vec_end = end - ((end - begin) % 8);
          for (; i < vec_end; i += 8) {
            const __m128i ha = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pA + i));
            const __m128i hb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pB + i));
            const __m128i hc = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pC + i));
            const __m256 va = _mm256_cvtph_ps(ha);
            const __m256 vb = _mm256_cvtph_ps(hb);
            const __m256 vc = _mm256_cvtph_ps(hc);
            const __m256 sum = _mm256_add_ps(_mm256_add_ps(va, vb), vc);
            const __m128i hout =
                _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), hout);
          }
          for (; i < end; ++i)
            out[i] = add3_scalar(pA[i], pB[i], pC[i]);
        });
        return Ort::Status{nullptr};
      }
#endif
      parallel_for_addadd(N, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i)
          out[i] = add3_scalar(pA[i], pB[i], pC[i]);
      });
    } else {
      parallel_for_addadd(N, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i)
          out[i] = add3_scalar(pA[i], pB[i], pC[i]);
      });
    }
    return Ort::Status{nullptr};
  }

  parallel_for_addadd(N, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i)
      out[i] = add3_scalar(pA[i % nA], pB[i % nB], pC[i % nC]);
  });

  return Ort::Status{nullptr};
}

inline AddAddKernelCpuFloat::AddAddKernelCpuFloat(const OrtApi * /* api */,
                                                  const OrtKernelInfo * /* info */) {}

inline Ort::Status AddAddKernelCpuFloat::Compute(const Ort::Custom::Tensor<float> &A,
                                                 const Ort::Custom::Tensor<float> &B,
                                                 const Ort::Custom::Tensor<float> &C,
                                                 Ort::Custom::Tensor<float> &output) {
  return ComputeAddAddCpuImpl(A, B, C, output);
}

inline AddAddKernelCpuFloat16::AddAddKernelCpuFloat16(const OrtApi * /* api */,
                                                      const OrtKernelInfo * /* info */) {}

inline Ort::Status AddAddKernelCpuFloat16::Compute(const Ort::Custom::Tensor<Float16> &A,
                                                   const Ort::Custom::Tensor<Float16> &B,
                                                   const Ort::Custom::Tensor<Float16> &C,
                                                   Ort::Custom::Tensor<Float16> &output) {
  return ComputeAddAddCpuImpl(A, B, C, output);
}

inline AddAddKernelCpuBFloat16::AddAddKernelCpuBFloat16(const OrtApi * /* api */,
                                                        const OrtKernelInfo * /* info */) {}

inline Ort::Status AddAddKernelCpuBFloat16::Compute(const Ort::Custom::Tensor<BFloat16> &A,
                                                    const Ort::Custom::Tensor<BFloat16> &B,
                                                    const Ort::Custom::Tensor<BFloat16> &C,
                                                    Ort::Custom::Tensor<BFloat16> &output) {
  return ComputeAddAddCpuImpl(A, B, C, output);
}

} // namespace ortops
