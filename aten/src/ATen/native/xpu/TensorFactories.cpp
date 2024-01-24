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

#include <ATen/native/Resize.h>
#include <ATen/xpu/EmptyTensor.h>
#include <ATen/xpu/XPUDevice.h>

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

Tensor _copy_from_xpu(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  at::xpu::copy(self, dst, non_blocking);
  return dst;
}

Tensor empty_xpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  Tensor result = at::detail::empty_xpu(
      size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      memory_format_opt);
  return result;
}

Tensor& set_storage_xpu_(
    Tensor& result,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  checkSetStorage(result, storage, storage_offset, size, stride);
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt =
      stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : c10::nullopt;

  if (result.sizes() == size &&
      (!stride_opt || result.strides() == stride_opt)) {
    return result;
  }

  at::xpu::resize_impl_xpu_(result, size, stride_opt);
  return result;
}

} // namespace at::native