#pragma once

// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_library
// Adapted from https://github.com/sdpython/onnx-extended

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>
#undef ORT_API_MANUAL_INIT

namespace ortops {

struct DenseToSparseKernelLite {
  DenseToSparseKernelLite(const OrtApi *api, const OrtKernelInfo *info);
  Ort::Status Compute(const Ort::Custom::Tensor<float> &X,
                      Ort::Custom::Tensor<float> &Y);
};

struct SparseToDenseKernelLite {
  SparseToDenseKernelLite(const OrtApi *api, const OrtKernelInfo *info);
  Ort::Status Compute(const Ort::Custom::Tensor<float> &X,
                      Ort::Custom::Tensor<float> &Y);
};

} // namespace ortops
