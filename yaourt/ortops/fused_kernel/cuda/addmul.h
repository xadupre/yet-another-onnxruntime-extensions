#pragma once

/**
 * @file addmul.h
 * @brief Fused AddMul / MulAdd CUDA custom operator.
 *
 * Declares the kernel and operator classes for two element-wise ternary
 * operations on three broadcastable input tensors A, B, C:
 *
 *  - **AddMul** (@c addition = @c true ): @f$ \text{output} = (A + B) \times C @f$
 *  - **MulAdd** (@c addition = @c false): @f$ \text{output} = A \times B + C @f$
 *
 * Both variants support an optional kernel attribute @c transposeMiddle.  When
 * set to @c true on a 4-D input the two middle axes of the output are
 * transposed, which avoids a separate Transpose node in common attention-kernel
 * patterns.
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the AddMul / MulAdd operator.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → AddMul; @c false → MulAdd.
 */
template <typename T, bool addition> struct AddMulKernel {
  AddMulKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  bool switch_middle_axis_; ///< When @c true, swap axes 1 and 2 of the 4-D output.
};

/**
 * @brief ORT custom-op descriptor for the AddMul / MulAdd operator.
 *
 * Registers the operator under the name @c "AddMul" (addition) or @c "MulAdd"
 * (multiplication-first) in the CUDA execution provider.  Expects exactly three
 * required inputs and produces one required output, all of type @c T.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → registers as "AddMul"; @c false → "MulAdd".
 */
template <typename T, bool addition>
struct AddMulOp : Ort::CustomOpBase<AddMulOp<T, addition>, AddMulKernel<T, addition>> {
  typedef Ort::CustomOpBase<AddMulOp<T, addition>, AddMulKernel<T, addition>> parent_type;
  AddMulOp() : parent_type() {}
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
