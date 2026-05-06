#pragma once

/**
 * @file addaddaddmulmulmul.h
 * @brief Fused AddAddAdd / MulMulMul CUDA custom operator for four inputs.
 *
 * Declares the kernel and operator classes for two element-wise quaternary
 * operations on four broadcastable input tensors A, B, C, D:
 *
 *  - **AddAddAdd** (@c addition = @c true ):
 *    @f$ \text{output} = A + B + C + D @f$
 *  - **MulMulMul** (@c addition = @c false):
 *    @f$ \text{output} = A \times B \times C \times D @f$
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the AddAddAdd / MulMulMul operator.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → element-wise addition of four inputs;
 *                   @c false → element-wise multiplication of four inputs.
 */
template <typename T, bool addition> struct AddAddAddMulMulMulKernel {
  AddAddAddMulMulMulKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

/**
 * @brief ORT custom-op descriptor for the AddAddAdd / MulMulMul operator.
 *
 * Registers the operator under the name @c "AddAddAdd" (addition) or
 * @c "MulMulMul" (multiplication) in the CUDA execution provider.  Expects
 * exactly four required inputs and produces one required output, all of
 * type @c T.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → registers as "AddAddAdd"; @c false → "MulMulMul".
 */
template <typename T, bool addition>
struct AddAddAddMulMulMulOp : Ort::CustomOpBase<AddAddAddMulMulMulOp<T, addition>,
                                                AddAddAddMulMulMulKernel<T, addition>> {
  typedef Ort::CustomOpBase<AddAddAddMulMulMulOp<T, addition>,
                            AddAddAddMulMulMulKernel<T, addition>>
      parent_type;
  AddAddAddMulMulMulOp() : parent_type() {}
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
