#pragma once

/**
 * @file scatter_nd_of_shape_masked.h
 * @brief MaskedScatterNDOfShape CUDA custom operator.
 *
 * Declares the kernel and operator classes for a masked variant of the
 * ScatterNDOfShape operation.  Each scatter step is skipped when the
 * corresponding index value equals a configurable @c masked_value, allowing
 * padding indices to be ignored without pre-filtering:
 *
 * @code
 *   output = zeros(shape)
 *   for each i:
 *     if indices[i] != masked_value:
 *       output[indices[i]] op= updates[i]
 * @endcode
 *
 * Inputs:
 *  - 0: @c shape        — 1-D @c int64 tensor defining the output shape (CPU).
 *  - 1: @c indices      — integer indices tensor (CPU).
 *  - 2: @c updates      — data tensor of type @c T (GPU).
 *
 * The @c reduction and @c masked_value behaviour is configured via kernel
 * attributes of the same names.
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include "scatter_nd_of_shape_common.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the MaskedScatterNDOfShape operator.
 *
 * @tparam T  Element type of @c updates and output (@c float or @c half).
 */
template <typename T> struct MaskedScatterNDOfShapeKernel {
  MaskedScatterNDOfShapeKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  void ComputeOptimize(cudaStream_t &stream, const std::vector<int64_t> &input_shape,
                       const std::vector<int64_t> &indices_shape, T *output_data,
                       const int64_t *indices_data, const T *updates_data) const;

  Reduction reduction_;    ///< How conflicts are resolved when multiple updates target the same element.
  int maxThreadPerBlock_;  ///< Device limit used to size CUDA thread blocks.
  int64_t masked_value_;   ///< Index value that triggers a skip (e.g. a padding token id).
};

/**
 * @brief ORT custom-op descriptor for the MaskedScatterNDOfShape operator.
 *
 * Registers the operator under the name @c "MaskedScatterNDOfShape" in the
 * CUDA execution provider.  Inputs 0 (shape) and 1 (indices) are read from CPU
 * memory; input 2 (updates) is read from GPU memory.  Produces one required
 * output of type @c T.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T>
struct MaskedScatterNDOfShapeOp
    : Ort::CustomOpBase<MaskedScatterNDOfShapeOp<T>, MaskedScatterNDOfShapeKernel<T>> {
  typedef Ort::CustomOpBase<MaskedScatterNDOfShapeOp<T>, MaskedScatterNDOfShapeKernel<T>>
      parent_type;
  MaskedScatterNDOfShapeOp() : parent_type() {}
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
