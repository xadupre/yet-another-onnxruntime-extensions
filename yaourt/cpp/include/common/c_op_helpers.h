#pragma once

#include <cstdint>
#include <vector>

namespace onnx_c_ops {

template <class NTYPE> NTYPE flattened_dimension(const std::vector<NTYPE> &values) {
  NTYPE r = 1;
  for (auto it = values.begin(); it != values.end(); ++it)
    r *= *it;
  return r;
}

template <class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE> &values, int64_t first) {
  NTYPE r = 1;
  auto end = values.begin() + first;
  for (auto it = values.begin(); it != end; ++it)
    r *= *it;
  return r;
}

template <class DIMTYPE>
DIMTYPE SizeFromDimension(const std::vector<DIMTYPE> &shape, std::size_t start,
                          std::size_t end) {
  DIMTYPE size = 1;
  for (std::size_t i = start; i < end; i++) {
    if (shape[i] < 0)
      return -1;
    size *= shape[i];
  }
  return size;
}

inline int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  return axis < 0 ? axis + tensor_rank : axis;
}

} // namespace onnx_c_ops
