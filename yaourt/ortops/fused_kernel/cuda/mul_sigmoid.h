#pragma once

/**
 * @file mul_sigmoid.h
 * @brief Fused MulSigmoid CUDA custom operator (SiLU / Swish activation).
 *
 * Declares the kernel and operator classes for the element-wise unary
 * operation:
 *
 * @f[ \text{output} = x \times \sigma(x), \quad
 *     \sigma(x) = \frac{1}{1 + e^{-x}} @f]
 *
 * This is equivalent to the SiLU (Sigmoid Linear Unit) / Swish activation
 * function, which is commonly used in transformer feed-forward blocks.
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the MulSigmoid (SiLU/Swish) operator.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T> struct MulSigmoidKernel {
  MulSigmoidKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

/**
 * @brief ORT custom-op descriptor for the MulSigmoid operator.
 *
 * Registers the operator under the name @c "MulSigmoid" in the CUDA execution
 * provider.  Expects one required input and produces one required output, both
 * of type @c T.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T>
struct MulSigmoidOp : Ort::CustomOpBase<MulSigmoidOp<T>, MulSigmoidKernel<T>> {
  typedef Ort::CustomOpBase<MulSigmoidOp<T>, MulSigmoidKernel<T>> parent_type;
  MulSigmoidOp() : parent_type() {}
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
