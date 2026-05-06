#pragma once

/**
 * @file replace_zero.h
 * @brief ReplaceZero CUDA custom operator — substitute zero elements.
 *
 * Declares the kernel and operator classes for the unary element-wise
 * operation:
 *
 * @f[ \text{output}_i = \begin{cases} \texttt{by} & \text{if } x_i = 0 \\
 *                                     x_i          & \text{otherwise} \end{cases} @f]
 *
 * The replacement scalar @c by is read from a kernel attribute of the same
 * name.  The operator is useful for masking out padding tokens or avoiding
 * division by zero in subsequent operations.
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the ReplaceZero operator.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T> struct ReplaceZeroKernel {
  ReplaceZeroKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  float by_; ///< Scalar value used to replace zero elements (attribute "by").
};

/**
 * @brief ORT custom-op descriptor for the ReplaceZero operator.
 *
 * Registers the operator under the name @c "ReplaceZero" in the CUDA execution
 * provider.  Expects one required input and produces one required output, both
 * of type @c T.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T>
struct ReplaceZeroOp : Ort::CustomOpBase<ReplaceZeroOp<T>, ReplaceZeroKernel<T>> {
  typedef Ort::CustomOpBase<ReplaceZeroOp<T>, ReplaceZeroKernel<T>> parent_type;
  ReplaceZeroOp() : parent_type() {}
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
