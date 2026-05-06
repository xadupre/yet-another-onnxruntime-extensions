#pragma once

/**
 * @file add_or_mul_shared_input.h
 * @brief Fused AddSharedInput / MulSharedInput CUDA custom operator.
 *
 * Declares the kernel and operator classes for an operation that applies the
 * same first input A to two different second inputs B and C simultaneously,
 * producing two outputs:
 *
 *  - **AddSharedInput** (@c addition = @c true ):
 *    @f$ \text{out}_0 = A + B,\quad \text{out}_1 = A + C @f$
 *  - **MulSharedInput** (@c addition = @c false):
 *    @f$ \text{out}_0 = A \times B,\quad \text{out}_1 = A \times C @f$
 *
 * This fused form avoids reading A twice when computing two independent
 * operations that share the same operand.
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the AddSharedInput / MulSharedInput
 *        operator.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → addition; @c false → multiplication.
 */
template <typename T, bool addition> struct AddOrMulSharedInputKernel {
  AddOrMulSharedInputKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

/**
 * @brief ORT custom-op descriptor for the AddSharedInput / MulSharedInput
 *        operator.
 *
 * Registers the operator under the name @c "AddSharedInput" (addition) or
 * @c "MulSharedInput" (multiplication) in the CUDA execution provider.
 * Expects exactly three required inputs (A, B, C) and produces two required
 * outputs (A op B, A op C), all of type @c T.
 *
 * @tparam T         Element type (@c float or @c half).
 * @tparam addition  @c true → registers as "AddSharedInput";
 *                   @c false → "MulSharedInput".
 */
template <typename T, bool addition>
struct AddOrMulSharedInputOp
    : Ort::CustomOpBase<AddOrMulSharedInputOp<T, addition>,
                        AddOrMulSharedInputKernel<T, addition>> {
  typedef Ort::CustomOpBase<AddOrMulSharedInputOp<T, addition>,
                            AddOrMulSharedInputKernel<T, addition>>
      parent_type;
  AddOrMulSharedInputOp() : parent_type() {}
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
