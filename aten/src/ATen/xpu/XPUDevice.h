#pragma once

#include <ATen/Context.h>
#include <ATen/core/TensorBase.h>
#include <c10/xpu/XPUFunctions.h>

#include <c10/util/OptionalArrayRef.h>

namespace at {

// TODO: remove duplication w/ ATen_fwd.h
using OptionalIntArrayRef = c10::OptionalArrayRef<int64_t>;

} // namespace at

namespace at::xpu {

inline Device getDeviceFromPtr(void* ptr) {
  auto device = c10::xpu::get_device_idx_from_pointer(ptr);
  return {c10::DeviceType::XPU, static_cast<DeviceIndex>(device)};
}

void copy(
    const at::TensorBase& self,
    const at::TensorBase& dst,
    bool non_blocking);

at::TensorBase resize_impl_xpu_(
    at::TensorBase& src,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool device_guard = true);

} // namespace at::xpu
