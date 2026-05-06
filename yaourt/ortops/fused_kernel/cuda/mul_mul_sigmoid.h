#pragma once

/**
 * @file mul_mul_sigmoid.h
 * @brief Fused MulMulSigmoid CUDA custom operator.
 *
 * Declares the kernel and operator classes for the element-wise binary
 * operation applied to two broadcastable input tensors x and y:
 *
 * @f[ \text{output} = x \times y \times \sigma(y), \quad
 *     \sigma(y) = \frac{1}{1 + e^{-y}} @f]
 *
 * The sigmoid is applied only to y, making this a gated variant of the
 * SiLU / Swish activation commonly found in gated linear units (GLUs) used in
 * transformer feed-forward networks.
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the MulMulSigmoid operator.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T> struct MulMulSigmoidKernel {
  MulMulSigmoidKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

/**
 * @brief ORT custom-op descriptor for the MulMulSigmoid operator.
 *
 * Registers the operator under the name @c "MulMulSigmoid" in the CUDA
 * execution provider.  Expects two required inputs (x, y) and produces one
 * required output, all of type @c T.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T>
struct MulMulSigmoidOp : Ort::CustomOpBase<MulMulSigmoidOp<T>, MulMulSigmoidKernel<T>> {
  typedef Ort::CustomOpBase<MulMulSigmoidOp<T>, MulMulSigmoidKernel<T>> parent_type;
  MulMulSigmoidOp() : parent_type() {}
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
