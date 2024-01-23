#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/EmptyTensor.h>
#include <ATen/xpu/EmptyTensor.h>
#include <c10/core/DeviceGuard.h>

namespace at::detail {

TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  // TODO: lazy init
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_xpu());
  // const DeviceGuard device_guard(device);
  /*
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_generic(
      size, allocator, cuda_dks, dtype, memory_format_opt);
*/
  // return {};

  constexpr c10::DispatchKeySet xpu_dks(c10::DispatchKey::XPU);
  auto* allocator = at::getCPUAllocator();
  return at::detail::empty_generic(
      size, allocator, xpu_dks, dtype, memory_format_opt);
}

TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  // TODO: lazy init
  const auto device = device_or_default(options.device_opt());
  TORCH_INTERNAL_ASSERT(false);
  const DeviceGuard device_guard(device);
  /*
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_generic(
      size, allocator, cuda_dks, dtype, memory_format_opt);
*/
  return {};
}

TensorBase empty_xpu(
    IntArrayRef size,
    ScalarType dtype,
    c10::optional<Device> device_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  // TODO: lazy init
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_xpu());
  const DeviceGuard device_guard(device);
  // auto* allocator = at::cuda::getCUDADeviceAllocator();
  // constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  constexpr c10::DispatchKeySet xpu_dks(c10 ::DispatchKey::XPU);
  // TODO: need xpu allocator
  return at::detail::empty_generic(
      size, nullptr, xpu_dks, dtype, memory_format_opt);
}

TensorBase empty_xpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  // TODO: can probably remove? not sure this is applicable
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      layout_or_default(layout_opt) == Layout::Strided);

  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_xpu(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_xpu(IntArrayRef size, const TensorOptions& options) {
  return at::detail::empty_xpu(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

} // namespace at::detail