#pragma once

#include <onnxruntime_c_api.h>

#if defined(_WIN32)

// nothing needed - ORT_EXPORT already defined by onnxruntime headers

#elif defined(__MACOSX__) || defined(__APPLE__)

// nothing needed

#else

#undef ORT_EXPORT
#define ORT_EXPORT __attribute__((visibility("default")))

#endif
