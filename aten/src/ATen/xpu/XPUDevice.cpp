#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>

#include <ATen/xpu/XPUDevice.h>

#include <c10/xpu/XPUStream.h>

namespace at::xpu {

void copy(
    const at::TensorBase& src,
    const at::TensorBase& dst,
    bool non_blocking) {
  // TODO: handle non_blocking
  TORCH_INTERNAL_ASSERT(dst.nbytes() == src.nbytes());

  if (src.device() != dst.device()) {
    if (src.device().is_cpu()) {
      TORCH_INTERNAL_ASSERT(dst.device().is_xpu());

      const auto device_index = dst.device().index();
      auto copy_stream = c10::xpu::getStreamFromPool(false, device_index);

      copy_stream.queue().memcpy(dst.data_ptr(), src.data_ptr(), src.nbytes());
    } else {
      TORCH_INTERNAL_ASSERT(src.device().is_xpu());
      TORCH_INTERNAL_ASSERT(dst.device().is_cpu());

      TORCH_INTERNAL_ASSERT(false, "TODO device to host");
    }

  } else {
    TORCH_INTERNAL_ASSERT("device device copy not yet supported");
  }
}

} // namespace at::xpu