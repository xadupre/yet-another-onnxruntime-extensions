#pragma once

// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_library
// Adapted from https://github.com/sdpython/onnx-extended

#include "common/sparse_tensor.h"
#include "ort_sparse_lite.h"

namespace ortops {

//////////
// Kernels
//////////

// DenseToSparse

inline DenseToSparseKernelLite::DenseToSparseKernelLite(const OrtApi * /* api */,
                                                        const OrtKernelInfo * /* info */) {}

inline Ort::Status DenseToSparseKernelLite::Compute(const Ort::Custom::Tensor<float> &X,
                                                     Ort::Custom::Tensor<float> &Y) {
  const float *X_data = X.Data();
  std::vector<int64_t> dimensions_in = X.Shape();
  if (dimensions_in.size() != 2) {
    return Ort::Status("DenseToSparse only allows 2D inputs.", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  int64_t n_rows = dimensions_in[0];
  int64_t n_cols = dimensions_in[1];

  int64_t n_elements = n_rows * n_cols;
  uint32_t n_els = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_elements); ++i) {
    if (X_data[i] != 0)
      ++n_els;
  }
  std::size_t size_float = onnx_sparse::sparse_struct::size_float(n_els, 1);

  std::vector<int64_t> dimensions_out{static_cast<int64_t>(size_float)};
  float *out = Y.Allocate(dimensions_out);

  onnx_sparse::sparse_struct *sp = reinterpret_cast<onnx_sparse::sparse_struct *>(out);
  sp->set(dimensions_in, n_els, 1);
  uint32_t *indices = sp->indices();
  float *values = sp->values();

  n_els = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_elements); ++i) {
    if (X_data[i] != 0) {
      indices[n_els] = static_cast<uint32_t>(i);
      values[n_els] = X_data[i];
      ++n_els;
    }
  }
  return Ort::Status{nullptr};
}

// SparseToDense

inline SparseToDenseKernelLite::SparseToDenseKernelLite(const OrtApi * /* api */,
                                                        const OrtKernelInfo * /* info */) {}

inline Ort::Status SparseToDenseKernelLite::Compute(const Ort::Custom::Tensor<float> &X,
                                                     Ort::Custom::Tensor<float> &Y) {
  const float *X_data = X.Data();
  std::vector<int64_t> dimensions_in = X.Shape();
  if (dimensions_in.size() != 1) {
    return Ort::Status("SparseToDense only allows 1D inputs.", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  const onnx_sparse::sparse_struct *sp =
      reinterpret_cast<const onnx_sparse::sparse_struct *>(X_data);
  if (sp->n_dims != 2) {
    return Ort::Status("SparseToDense expects a 2D encoded tensor.",
                       OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  std::vector<int64_t> dimensions_out{sp->shape[0], sp->shape[1]};
  float *out = Y.Allocate(dimensions_out);

  std::fill(out, out + sp->shape[0] * sp->shape[1], 0.f);
  const uint32_t *indices = sp->indices();
  const float *values = sp->values();
  for (std::size_t i = 0; i < sp->n_elements; ++i) {
    out[indices[i]] = values[i];
  }
  return Ort::Status{nullptr};
}

} // namespace ortops
