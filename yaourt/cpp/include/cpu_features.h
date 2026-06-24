#pragma once

namespace ortops {

bool cpu_supports_avx2();
bool cpu_supports_avx512f();
bool cpu_supports_f16c();

} // namespace ortops
