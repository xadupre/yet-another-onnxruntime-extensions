#pragma once

// Adapted from https://github.com/sdpython/onnx-extended

#include "common/common_kernels.h"
#include "cublas_v2.h"
#include "onnx_extended_helpers.h"
#include <cuda_runtime.h>

namespace ortops {

inline const char *cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "<unknown>";
  }
}

template <typename T>
void _check_cuda(T err, const char *const func, const char *const file, const int line) {
  if (err != cudaSuccess) {
    EXT_THROW("CUDA error at ", file, ":", line, " code=", static_cast<unsigned int>(err),
              " \"", cudaGetErrorString(err), "\" in ", func);
  }
}

#define CUDA_THROW_IF_ERROR(expr) _check_cuda((expr), #expr, __FILE__, __LINE__)

} // namespace ortops
