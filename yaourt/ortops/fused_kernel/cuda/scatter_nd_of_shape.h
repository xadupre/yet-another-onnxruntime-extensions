#pragma once

/**
 * @file scatter_nd_of_shape.h
 * @brief ScatterNDOfShape CUDA custom operator.
 *
 * Declares the kernel and operator classes for scattering @c updates into a
 * zero-initialised output tensor whose shape is defined by the first input.
 * The operation is equivalent to:
 *
 * @code
 *   output = zeros(shape)
 *   output[indices[i]] op= updates[i]  for each i
 * @endcode
 *
 * where @c op is determined by the @c reduction kernel attribute
 * (@c Reduction::None overwrites, @c Add accumulates, etc.).  A second
 * strategy attribute (@c Strategy::None or @c Optimize) lets the kernel choose
 * a shape-specific fast path at runtime.
 *
 * Inputs:
 *  - 0: @c shape   — 1-D @c int64 tensor defining the output shape (CPU).
 *  - 1: @c indices — integer indices tensor (CPU).
 *  - 2: @c updates — data tensor of type @c T (GPU).
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include "scatter_nd_of_shape_common.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the ScatterNDOfShape operator.
 *
 * @tparam T  Element type of @c updates and output (@c float or @c half).
 */
template <typename T> struct ScatterNDOfShapeKernel {
  ScatterNDOfShapeKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  void ComputeNone(cudaStream_t &stream, const std::vector<int64_t> &input_shape,
                   const std::vector<int64_t> &indices_shape, T *output_data,
                   const int64_t *indices_data, const T *updates_data) const;
  void ComputeOptimize(cudaStream_t &stream, const std::vector<int64_t> &input_shape,
                       const std::vector<int64_t> &indices_shape, T *output_data,
                       const int64_t *indices_data, const T *updates_data) const;

  Reduction reduction_;      ///< How conflicts are resolved when multiple updates target the same element.
  Strategy strategy_;        ///< Execution strategy selected at kernel construction time.
  int maxThreadPerBlock_;    ///< Device limit used to size CUDA thread blocks.
};

/**
 * @brief ORT custom-op descriptor for the ScatterNDOfShape operator.
 *
 * Registers the operator under the name @c "ScatterNDOfShape" in the CUDA
 * execution provider.  Inputs 0 (shape) and 1 (indices) are read from CPU
 * memory; input 2 (updates) is read from GPU memory.  Produces one required
 * output of type @c T.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T>
struct ScatterNDOfShapeOp
    : Ort::CustomOpBase<ScatterNDOfShapeOp<T>, ScatterNDOfShapeKernel<T>> {
  typedef Ort::CustomOpBase<ScatterNDOfShapeOp<T>, ScatterNDOfShapeKernel<T>> parent_type;
  ScatterNDOfShapeOp() : parent_type() {}
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
