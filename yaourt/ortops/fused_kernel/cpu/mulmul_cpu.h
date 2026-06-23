#pragma once

/**
 * @file mulmul_cpu.h
 * @brief Fused MulMul CPU custom operator.
 *
 * Declares the kernel class for the element-wise ternary multiplication
 * applied to three broadcastable input tensors A, B, C:
 *
 * @f[ \text{output} = A \times B \times C @f]
 *
 * The implementation uses AVX2 SIMD instructions for vectorised throughput
 * and std::thread for multi-threaded parallelism.
 *
 * Supported element type: @c float.
 */

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>
#undef ORT_API_MANUAL_INIT

namespace ortops {

/**
 * @brief CPU kernel for the MulMul operator.
 *
 * Computes @f$ \text{output}[i] = A[i \bmod n_A] \times B[i \bmod n_B]
 * \times C[i \bmod n_C] @f$ for every output index @c i, supporting
 * 1-D (scalar) broadcasting via element-count modulo arithmetic — the same
 * scheme used by the corresponding CUDA operator.
 *
 * The hot loop is vectorised with AVX2 (8 × float32 per iteration) and
 * parallelised with std::thread across chunks of the output array.
 * Using std::thread (rather than OpenMP) ensures all spawned threads are
 * joined before Compute returns, avoiding conflicts with ORT's own thread
 * pool and with dlopen / dlclose lifecycle management.
 */
struct MulMulKernelCpu {
  MulMulKernelCpu(const OrtApi *api, const OrtKernelInfo *info);
  Ort::Status Compute(const Ort::Custom::Tensor<float> &A,
                      const Ort::Custom::Tensor<float> &B,
                      const Ort::Custom::Tensor<float> &C,
                      Ort::Custom::Tensor<float> &output);
};

} // namespace ortops
