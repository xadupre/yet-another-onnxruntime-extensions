#pragma once

/**
 * @file tri_matrix.h
 * @brief TriMatrix CUDA custom operator — triangular matrix generator.
 *
 * Declares the kernel and operator classes that create a 2-D triangular matrix
 * whose lower-triangle, diagonal, and upper-triangle elements are filled with
 * user-supplied scalar values.
 *
 * For a matrix of size @f$ (n\_rows \times n\_cols) @f$, element
 * @f$ (r, c) @f$ is set to:
 *
 * @f[ \text{output}[r, c] = \begin{cases}
 *   \texttt{lower} & \text{if } r > c \\
 *   \texttt{diag}  & \text{if } r = c \\
 *   \texttt{upper} & \text{if } r < c
 * \end{cases} @f]
 *
 * Inputs:
 *  - 0: @c shape  — 1-D @c int64 tensor @f$ [n\_rows, n\_cols] @f$ (CPU).
 *  - 1: @c values — 1-D tensor of type @c T with three elements
 *                   @f$ [\text{lower}, \text{diag}, \text{upper}] @f$ (CPU).
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the TriMatrix operator.
 *
 * @tparam T  Element type of the output matrix (@c float or @c half).
 */
template <typename T> struct TriMatrixKernel {
  TriMatrixKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

/**
 * @brief ORT custom-op descriptor for the TriMatrix operator.
 *
 * Registers the operator under the name @c "TriMatrix" in the CUDA execution
 * provider.  Both inputs are read from CPU memory.  Produces one required
 * output tensor of type @c T containing the triangular matrix.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T>
struct TriMatrixOp : Ort::CustomOpBase<TriMatrixOp<T>, TriMatrixKernel<T>> {
  typedef Ort::CustomOpBase<TriMatrixOp<T>, TriMatrixKernel<T>> parent_type;
  TriMatrixOp() : parent_type() {}
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;

  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(std::size_t index) const;
  OrtMemType GetInputMemoryType(std::size_t index) const;

  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const;
};

} // namespace ortops
