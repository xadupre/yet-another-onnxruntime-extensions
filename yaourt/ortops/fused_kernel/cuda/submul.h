#pragma once

/**
 * @file submul.h
 * @brief Fused SubMul / MulSub CUDA custom operator.
 *
 * Declares the kernel and operator classes for element-wise ternary operations
 * on three broadcastable input tensors A, B, C.  The template parameter
 * @c addition selects whether subtraction precedes or follows multiplication,
 * and the optional kernel attribute @c negative inverts the sign of the
 * subtraction operand:
 *
 *  - **SubMul** (@c addition = @c true, @c negative = @c false):
 *    @f$ \text{output} = (A - B) \times C @f$
 *  - **SubMul** (@c addition = @c true, @c negative = @c true):
 *    @f$ \text{output} = (B - A) \times C @f$
 *  - **MulSub** (@c addition = @c false, @c negative = @c false):
 *    @f$ \text{output} = A \times B - C @f$
 *  - **MulSub** (@c addition = @c false, @c negative = @c true):
 *    @f$ \text{output} = C - A \times B @f$
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the SubMul / MulSub operator.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → SubMul (subtraction first);
 *                   @c false → MulSub (multiplication first).
 */
template <typename T, bool addition> struct SubMulKernel {
  SubMulKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  bool negative_; ///< When @c true, the subtraction order is reversed.
};

/**
 * @brief ORT custom-op descriptor for the SubMul / MulSub operator.
 *
 * Registers the operator under the name @c "SubMul" (addition) or @c "MulSub"
 * (multiplication-first) in the CUDA execution provider.  Expects exactly three
 * required inputs and produces one required output, all of type @c T.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → registers as "SubMul"; @c false → "MulSub".
 */
template <typename T, bool addition>
struct SubMulOp : Ort::CustomOpBase<SubMulOp<T, addition>, SubMulKernel<T, addition>> {
  typedef Ort::CustomOpBase<SubMulOp<T, addition>, SubMulKernel<T, addition>> parent_type;
  SubMulOp() : parent_type() {}
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
