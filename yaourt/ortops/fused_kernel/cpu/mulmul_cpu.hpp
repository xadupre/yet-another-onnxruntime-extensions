#pragma once

#include "mulmul_cpu.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace ortops {

/**
 * @brief Applies @p fn(begin, end) over @c [0, N) in parallel using
 * std::thread.  Each thread receives a contiguous sub-range.  All threads are
 * joined before the function returns, making the call safe inside dynamically
 * loaded shared libraries.
 */
static inline void parallel_for(int64_t N,
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

inline MulMulKernelCpu::MulMulKernelCpu(const OrtApi * /* api */,
                                         const OrtKernelInfo * /* info */) {}

inline Ort::Status MulMulKernelCpu::Compute(const Ort::Custom::Tensor<float> &A,
                                             const Ort::Custom::Tensor<float> &B,
                                             const Ort::Custom::Tensor<float> &C,
                                             Ort::Custom::Tensor<float> &output) {
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
        (std::string("MulMul: all inputs must be non-empty (got nA=") + std::to_string(nA) +
         ", nB=" + std::to_string(nB) + ", nC=" + std::to_string(nC) + ")")
            .c_str(),
        OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  const int64_t N = std::max(nA, std::max(nB, nC));

  // Build output shape: broadcast across the maximum rank.
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

  float *out = output.Allocate(out_shape);
  const float *pA = A.Data();
  const float *pB = B.Data();
  const float *pC = C.Data();

  // Fast path: no broadcasting – use AVX2 + multi-threading.
  if (nA == N && nB == N && nC == N) {
#ifdef __AVX2__
    // Each thread processes a contiguous range, using 8-wide AVX2 vectors
    // plus a scalar tail.
    parallel_for(N, [&](int64_t begin, int64_t end) {
      int64_t i = begin;
      const int64_t vec_end = end - ((end - begin) % 8);
      for (; i < vec_end; i += 8) {
        const __m256 va = _mm256_loadu_ps(pA + i);
        const __m256 vb = _mm256_loadu_ps(pB + i);
        const __m256 vc = _mm256_loadu_ps(pC + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(va, vb), vc));
      }
      for (; i < end; ++i)
        out[i] = pA[i] * pB[i] * pC[i];
    });
#else
    parallel_for(N, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i)
        out[i] = pA[i] * pB[i] * pC[i];
    });
#endif
    return Ort::Status{nullptr};
  }

  // General broadcasting path – parallel scalar loop.
  parallel_for(N, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i)
      out[i] = pA[i % nA] * pB[i % nB] * pC[i % nC];
  });

  return Ort::Status{nullptr};
}

} // namespace ortops
