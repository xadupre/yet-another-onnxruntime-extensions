#pragma once

/**
 * @file addadd_cpu.h
 * @brief Fused AddAdd CPU custom operator.
 *
 * Declares CPU kernels for the element-wise ternary addition applied to three
 * broadcastable input tensors A, B, C:
 *
 * @f[ \text{output} = A + B + C @f]
 *
 * Supported element types: @c float, @c Ort::Float16_t, @c Ort::BFloat16_t.
 */

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>
#undef ORT_API_MANUAL_INIT

namespace ortops {

using Float16 = Ort::Float16_t;
using BFloat16 = Ort::BFloat16_t;

/**
 * @brief CPU kernel for the AddAdd operator using float32 tensors.
 */
struct AddAddKernelCpuFloat {
  bool has_avx2_;
  AddAddKernelCpuFloat(const OrtApi *api, const OrtKernelInfo *info);
  Ort::Status Compute(const Ort::Custom::Tensor<float> &A,
                      const Ort::Custom::Tensor<float> &B,
                      const Ort::Custom::Tensor<float> &C,
                      Ort::Custom::Tensor<float> &output);
};

/**
 * @brief CPU kernel for the AddAdd operator using float16 tensors.
 */
struct AddAddKernelCpuFloat16 {
  bool has_avx2_;
  bool has_f16c_;
  AddAddKernelCpuFloat16(const OrtApi *api, const OrtKernelInfo *info);
  Ort::Status Compute(const Ort::Custom::Tensor<Float16> &A,
                      const Ort::Custom::Tensor<Float16> &B,
                      const Ort::Custom::Tensor<Float16> &C,
                      Ort::Custom::Tensor<Float16> &output);
};

/**
 * @brief CPU kernel for the AddAdd operator using bfloat16 tensors.
 */
struct AddAddKernelCpuBFloat16 {
  AddAddKernelCpuBFloat16(const OrtApi *api, const OrtKernelInfo *info);
  Ort::Status Compute(const Ort::Custom::Tensor<BFloat16> &A,
                      const Ort::Custom::Tensor<BFloat16> &B,
                      const Ort::Custom::Tensor<BFloat16> &C,
                      Ort::Custom::Tensor<BFloat16> &output);
};

} // namespace ortops
