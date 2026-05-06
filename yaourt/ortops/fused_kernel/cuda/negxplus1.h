#pragma once

/**
 * @file negxplus1.h
 * @brief NegXplus1 CUDA custom operator — element-wise complement (1 − x).
 *
 * Declares the kernel and operator classes for the unary element-wise
 * operation:
 *
 * @f[ \text{output}_i = 1 - x_i @f]
 *
 * This is the arithmetic complement of the input, useful for computing
 * probability complements (e.g. @f$ 1 - p @f$) without an extra Constant and
 * Sub node in the graph.
 *
 * Supported element types: @c float, @c half, @c int32_t.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the NegXplus1 operator.
 *
 * @tparam T  Element type (@c float, @c half, or @c int32_t).
 */
template <typename T> struct NegXplus1Kernel {
  NegXplus1Kernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

/**
 * @brief ORT custom-op descriptor for the NegXplus1 operator.
 *
 * Registers the operator under the name @c "NegXplus1" in the CUDA execution
 * provider.  Expects one required input and produces one required output, both
 * of type @c T.
 *
 * @tparam T  Element type (@c float, @c half, or @c int32_t).
 */
template <typename T>
struct NegXplus1Op : Ort::CustomOpBase<NegXplus1Op<T>, NegXplus1Kernel<T>> {
  typedef Ort::CustomOpBase<NegXplus1Op<T>, NegXplus1Kernel<T>> parent_type;
  NegXplus1Op() : parent_type() {}
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;

  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(std::size_t index) const;

  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const;
};

} // namespace ortops
