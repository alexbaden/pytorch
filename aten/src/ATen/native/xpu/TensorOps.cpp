#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/xpu/EmptyTensor.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_strided_native.h>
#endif

#include <ATen/xpu/EmptyTensor.h>

namespace at::native {

Tensor empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  Tensor result = at::detail::empty_strided_xpu(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  return result;
}

} // namespace at::native