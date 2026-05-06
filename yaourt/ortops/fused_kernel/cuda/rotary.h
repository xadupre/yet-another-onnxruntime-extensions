#pragma once

/**
 * @file rotary.h
 * @brief Rotary positional embedding CUDA custom operator.
 *
 * Declares the kernel and operator classes that apply a rotary transformation
 * to the last dimension of an input tensor.  The operation implements rotary
 * position encodings (RoPE) as used in models such as LLaMA and GPT-NeoX.
 *
 * Each pair of elements @f$ (x_i,\; x_{i + \text{stride}/2}) @f$ in the last
 * dimension is rotated depending on @c RotarySide:
 *
 *  - **LEFT** side (@c rotary_side_ = LEFT):
 *    @f$ \text{out}[i] = x[i + \text{stride}/2],\quad
 *        \text{out}[i + \text{stride}/2] = -x[i] @f$
 *  - **RIGHT** side (@c rotary_side_ = RIGHT):
 *    @f$ \text{out}[i] = -x[i + \text{stride}/2],\quad
 *        \text{out}[i + \text{stride}/2] = x[i] @f$
 *
 * The side is selected via the kernel attribute @c "side" (integer, 1 = LEFT,
 * 2 = RIGHT).  The first input provides the data tensor and is expected in
 * device memory; the second input (optional shape hint) is read from CPU
 * memory.
 *
 * Supported element types: @c float, @c half.
 */

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>

namespace ortops {

/**
 * @brief Selects which half of the rotary operation to apply.
 *
 * The two complementary halves must be applied to the same input tensor with
 * LEFT and RIGHT, respectively, and their outputs concatenated (or summed with
 * cosine / sine embeddings) to obtain the full RoPE result.
 */
enum class RotarySide : int {
  LEFT = 1,  ///< Rotates the pair so the second half becomes the first output.
  RIGHT = 2, ///< Rotates the pair so the negated first half is the first output.
};

/**
 * @brief CUDA kernel implementation for the Rotary operator.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T> struct RotaryKernel {
  RotaryKernel(const OrtApi &api, const OrtKernelInfo *info);
  void Compute(OrtKernelContext *context);

private:
  RotarySide rotary_side_; ///< Which rotary half to compute (LEFT or RIGHT).
};

/**
 * @brief ORT custom-op descriptor for the Rotary operator.
 *
 * Registers the operator under the name @c "Rotary" in the CUDA execution
 * provider.  Expects one required data input (device memory) plus one required
 * shape input (CPU memory), and produces one required output of type @c T.
 *
 * @tparam T  Element type (@c float or @c half).
 */
template <typename T> struct RotaryOp : Ort::CustomOpBase<RotaryOp<T>, RotaryKernel<T>> {
  typedef Ort::CustomOpBase<RotaryOp<T>, RotaryKernel<T>> parent_type;
  RotaryOp() : parent_type() {}
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
