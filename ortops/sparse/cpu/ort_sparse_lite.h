#pragma once

// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_library
// Adapted from https://github.com/sdpython/onnx-extended

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>
#undef ORT_API_MANUAL_INIT

namespace ortops {

/// Converts a 2-D dense float32 tensor into a compact flat sparse encoding.
/// Only non-zero elements are stored. The 1-D output tensor encodes the
/// original shape, the number of non-zero elements, their flat indices
/// (stored as uint32), and their values (float32). The encoding is suitable
/// as input to SparseToDense for a lossless round-trip.
///
/// Constraints: input must be exactly 2-D; only float32 is supported.
///
/// @param[in] X 2-D dense float32 input tensor of shape [n_rows, n_cols].
///   Zero elements are not stored in the sparse encoding.
/// @param[out] Y 1-D float32 tensor containing the sparse encoding of X.
///   Layout: header | flat indices (uint32) | non-zero values (float32).
struct DenseToSparseKernelLite {
  DenseToSparseKernelLite(const OrtApi *api, const OrtKernelInfo *info);
  Ort::Status Compute(const Ort::Custom::Tensor<float> &X,
                      Ort::Custom::Tensor<float> &Y);
};

/// Converts the compact sparse encoding produced by DenseToSparse back into
/// a 2-D dense float32 tensor. Positions that were zero in the original
/// tensor are filled with 0.0. The output shape is recovered from the
/// sparse header embedded in the input.
///
/// Constraints: input must be 1-D with a valid sparse header; the encoded
/// shape must be 2-D; only float32 is supported.
///
/// @param[in] X 1-D float32 sparse encoding produced by DenseToSparse.
/// @param[out] Y Reconstructed 2-D dense float32 tensor. The shape is
///   recovered from the sparse header embedded in X.
struct SparseToDenseKernelLite {
  SparseToDenseKernelLite(const OrtApi *api, const OrtKernelInfo *info);
  Ort::Status Compute(const Ort::Custom::Tensor<float> &X,
                      Ort::Custom::Tensor<float> &Y);
};

} // namespace ortops
