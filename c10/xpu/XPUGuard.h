
#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>

#include <c10/xpu/XPUFunctions.h>

namespace c10 {
namespace xpu {

struct XPUGuard final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::XPU;

  XPUGuard() = default;
  explicit XPUGuard(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::XPU);
  }
  DeviceType type() const override {
    return DeviceType::XPU;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_xpu());
    int old_device_index = c10::xpu::exchange_device(d.index());
    return Device(DeviceType::XPU, static_cast<DeviceIndex>(old_device_index));
  }
  Device getDevice() const override {
    auto device = c10::xpu::current_device();
    return Device(DeviceType::XPU, static_cast<DeviceIndex>(device));
  }

  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_xpu());
    c10::xpu::set_device(d.index());
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    setDevice(d);
  }
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::DEFAULT, getDevice());
  }
  Stream getDefaultStream(Device d) const override {
    return Stream(Stream::DEFAULT, getDevice());
  }
  Stream exchangeStream(Stream s) const noexcept override {
    return Stream(Stream::DEFAULT, getDevice());
  }
  DeviceIndex deviceCount() const noexcept override {
    return c10::xpu::device_count();
  }
};

} // namespace xpu
} // namespace c10