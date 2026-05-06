#pragma once

/**
 * @file scatter_nd_of_shape_common.h
 * @brief Shared types used by the ScatterNDOfShape and MaskedScatterNDOfShape
 *        CUDA custom operators.
 *
 * This header defines the enumerations and helper structures that are common to
 * both the ScatterNDOfShape and MaskedScatterNDOfShape kernels.  It is included
 * by scatter_nd_of_shape.h and scatter_nd_of_shape_masked.h.
 */

namespace ortops {

/**
 * @brief Reduction mode applied when multiple updates target the same output
 *        element during a scatter operation.
 */
enum class Reduction : int {
  None = 0, ///< No reduction; later writes overwrite earlier ones.
  Add = 1,  ///< Accumulate updates by addition.
  Mul = 2,  ///< Accumulate updates by multiplication.
  Min = 3,  ///< Keep the minimum of all updates.
  Max = 4,  ///< Keep the maximum of all updates.
};

/**
 * @brief Execution strategy selected by the scatter kernel at runtime.
 *
 * The kernel inspects the tensor shapes at inference time and picks the most
 * efficient code path automatically.
 */
enum class Strategy : int {
  None = 0,     ///< Generic path — no shape-specific optimisation.
  Optimize = 1, ///< Optimised path exploiting a specific shape pattern.
};

/**
 * @brief Fixed-capacity array used to pass tensor dimensions to CUDA kernels.
 *
 * Storing up to 12 dimension values avoids dynamic allocation inside device
 * code.  Callers must ensure that the actual rank does not exceed this limit.
 */
struct Shape2 {
  int64_t dims[12]; ///< Dimension values; unused slots are left uninitialised.
};

} // namespace ortops
