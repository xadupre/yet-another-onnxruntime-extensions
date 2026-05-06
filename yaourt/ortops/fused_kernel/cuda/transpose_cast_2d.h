#pragma once

/**
 * @file transpose_cast_2d.h
 * @brief Transpose2DCast CUDA custom operator — 2-D matrix transpose with type
 *        conversion.
 *
 * Declares the kernel and operator classes for transposing a 2-D matrix and
 * simultaneously casting its elements to a different numeric type.  Two
 * registered operator names are available depending on the output type:
 *
 *  - **Transpose2DCastFP16** — input @c float → output @c half.
 *  - **Transpose2DCastFP32** — input @c half → output @c float.
 *
 * The operator fuses the Transpose and Cast nodes into a single tiled CUDA
 * kernel, avoiding a round-trip through global memory compared to executing
 * them separately.
 *
 * The input and output types are chosen at construction time and stored in the
 * operator descriptor.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief CUDA kernel implementation for the Transpose2DCast operator.
 *
 * Uses a shared-memory tiled transpose to maximize memory coalescing while
 * applying the element-wise type cast.
 */
struct Transpose2DCastKernel {
  Transpose2DCastKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);
};

/**
 * @brief ORT custom-op descriptor for the Transpose2DCast operator.
 *
 * Registers the operator under the name @c "Transpose2DCastFP16" or
 * @c "Transpose2DCastFP32" (depending on @c output_type_) in the CUDA
 * execution provider.  Expects one required input and produces one required
 * output; the element types are fixed by the constructor arguments.
 */
struct Transpose2DCastOp : Ort::CustomOpBase<Transpose2DCastOp, Transpose2DCastKernel> {
  typedef Ort::CustomOpBase<Transpose2DCastOp, Transpose2DCastKernel> parent_type;
  Transpose2DCastOp(ONNXTensorElementDataType input_type,
                    ONNXTensorElementDataType output_type)
      : parent_type() {
    input_type_ = input_type;
    output_type_ = output_type;
  }
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;
  const char *GetName() const;
  const char *GetExecutionProviderType() const;

  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(std::size_t index) const;

  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const;

private:
  ONNXTensorElementDataType input_type_;  ///< Expected element type of the input tensor.
  ONNXTensorElementDataType output_type_; ///< Element type produced in the output tensor.
};

} // namespace ortops
