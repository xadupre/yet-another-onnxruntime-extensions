#pragma once

/**
 * @file addaddmulmul.h
 * @brief Fused AddAdd / MulMul CUDA custom operator for three inputs.
 *
 * Declares the kernel and operator classes for two element-wise ternary
 * operations on three broadcastable input tensors A, B, C:
 *
 *  - **AddAdd** (@c addition = @c true ): @f$ \text{output} = A + B + C @f$
 *  - **MulMul** (@c addition = @c false): @f$ \text{output} = A \times B \times C @f$
 *
 * When all inputs share the same shape the kernel uses a no-broadcast path for
 * better performance.
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the AddAdd / MulMul operator.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → element-wise addition; @c false → element-wise
 *                   multiplication.
 */
template <typename T, bool addition> struct AddAddMulMulKernel {
  AddAddMulMulKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

/**
 * @brief ORT custom-op descriptor for the AddAdd / MulMul operator.
 *
 * Registers the operator under the name @c "AddAdd" (addition) or @c "MulMul"
 * (multiplication) in the CUDA execution provider.  Expects exactly three
 * required inputs and produces one required output, all of type @c T.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → registers as "AddAdd"; @c false → "MulMul".
 */
template <typename T, bool addition>
struct AddAddMulMulOp
    : Ort::CustomOpBase<AddAddMulMulOp<T, addition>, AddAddMulMulKernel<T, addition>> {
  typedef Ort::CustomOpBase<AddAddMulMulOp<T, addition>, AddAddMulMulKernel<T, addition>>
      parent_type;
  AddAddMulMulOp() : parent_type() {}
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
