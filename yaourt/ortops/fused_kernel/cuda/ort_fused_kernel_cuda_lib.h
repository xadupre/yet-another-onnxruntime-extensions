#pragma once

/**
 * @file ort_fused_kernel_cuda_lib.h
 * @brief Public C entry point for the fused-kernel CUDA custom-op library.
 *
 * This header declares the single exported symbol that ONNX Runtime calls when
 * it loads the shared library at session-creation time.  The implementation
 * (ort_fused_kernel_cuda_lib.cu) registers all CUDA custom operators defined
 * in the @c yaourt.ortops.fused_kernel.cuda domain.
 */

#include "ortapi_c_api_header.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Registers all fused-kernel CUDA custom operators with an ORT session.
 *
 * ONNX Runtime discovers and calls this function automatically when loading the
 * shared library.  It adds every operator in the
 * @c yaourt.ortops.fused_kernel.cuda domain to the provided session options so
 * that they become available during model inference.
 *
 * @param options   The session options object to which the custom-op domain is
 *                  added.
 * @param api_base  The ORT API base pointer supplied by the runtime.
 * @return          @c nullptr on success, or a newly allocated @c OrtStatus
 *                  describing the error on failure.
 */
ORT_EXPORT OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                     const OrtApiBase *api_base);

#ifdef __cplusplus
}
#endif
