#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "onnx_extended_helpers.h"

namespace ortops {

////////////////////////
// errors and exceptions
////////////////////////

template <typename T> struct CTypeToOnnxType {
  ONNXTensorElementDataType onnx_type() const;
};

template <> struct CTypeToOnnxType<float> {
  inline ONNXTensorElementDataType onnx_type() const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

template <> struct CTypeToOnnxType<int64_t> {
  inline ONNXTensorElementDataType onnx_type() const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
};

template <> struct CTypeToOnnxType<int32_t> {
  inline ONNXTensorElementDataType onnx_type() const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  }
};

template <> struct CTypeToOnnxType<double> {
  inline ONNXTensorElementDataType onnx_type() const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  }
};

inline void _ThrowOnError_(OrtStatus *ort_status, const char *filename, int line,
                           const OrtApi &api) {
  if (ort_status) {
    OrtErrorCode code = api.GetErrorCode(ort_status);
    if (code == ORT_OK) {
      api.ReleaseStatus(ort_status);
    } else {
      std::string message(api.GetErrorMessage(ort_status));
      api.ReleaseStatus(ort_status);
      if (code != ORT_OK) {
        throw std::runtime_error(onnx_extended_helpers::MakeString(
            "error: onnxruntime(", code, "), ", message, "\n    ", filename, ":", line));
      }
    }
  }
}

#define ThrowOnError(api, ort_status) _ThrowOnError_(ort_status, __FILE__, __LINE__, api)

inline std::string KernelInfoGetInputName(const OrtApi &api, const OrtKernelInfo *info,
                                          int index) {
  std::size_t size;
  OrtStatus *status = api.KernelInfo_GetInputName(info, index, nullptr, &size);
  if (status != nullptr) {
    OrtErrorCode code = api.GetErrorCode(status);
    if (code == ORT_FAIL) {
      api.ReleaseStatus(status);
      return std::string();
    } else {
      ThrowOnError(api, status);
    }
    api.ReleaseStatus(status);
  }
  std::string str_out;
  str_out.resize(size);
  ThrowOnError(api, api.KernelInfo_GetInputName(info, index, &str_out[0], &size));
  str_out.resize(size - 1); // remove the terminating character '\0'
  return str_out;
}

} // namespace ortops
