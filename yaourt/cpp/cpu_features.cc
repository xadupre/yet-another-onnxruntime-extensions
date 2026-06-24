#include "cpu_features.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace ortops {

bool cpu_supports_avx2() {
#if defined(__x86_64__) || defined(__i386) || defined(_M_X64) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
  int regs[4] = {0, 0, 0, 0};
  __cpuidex(regs, 0, 0);
  if (regs[0] < 7)
    return false;
  __cpuidex(regs, 7, 0);
  return (regs[1] & (1 << 5)) != 0;
#else
  return false;
#endif
#else
  return false;
#endif
}

bool cpu_supports_avx512f() {
#if defined(__x86_64__) || defined(__i386) || defined(_M_X64) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f");
#elif defined(_MSC_VER)
  int regs[4] = {0, 0, 0, 0};
  __cpuidex(regs, 0, 0);
  if (regs[0] < 7)
    return false;
  __cpuidex(regs, 7, 0);
  return (regs[1] & (1 << 16)) != 0;
#else
  return false;
#endif
#else
  return false;
#endif
}

bool cpu_supports_f16c() {
#if defined(__x86_64__) || defined(__i386) || defined(_M_X64) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("f16c");
#elif defined(_MSC_VER)
  int regs[4] = {0, 0, 0, 0};
  __cpuidex(regs, 1, 0);
  return (regs[2] & (1 << 29)) != 0;
#else
  return false;
#endif
#else
  return false;
#endif
}

} // namespace ortops
