#pragma once

#include "ortapi_c_api_header.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                     const OrtApiBase *api_base);

ORT_EXPORT bool ORT_API_CALL CpuSupportsAvx2();
ORT_EXPORT bool ORT_API_CALL CpuSupportsF16c();

#ifdef __cplusplus
}
#endif
