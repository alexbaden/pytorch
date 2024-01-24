#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

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

static inline void maybe_resize_storage_xpu(
    TensorImpl* self,
    size_t new_size_bytes) {
  if (self->numel() == 0) {
    // do not resize to 0
    return;
  }

  const Storage& storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (new_size_bytes > storage.nbytes()) {
    TORCH_CHECK(
        storage.resizable(), "Trying to resize storage that is not resizable");
    auto allocator = storage.allocator();
    TORCH_CHECK(
        allocator != nullptr, "Trying to resize storage without an allocator");

    c10::Device device = storage.device();
    if (new_size_bytes == 0) {
      storage.set_data_ptr_noswap(at::DataPtr(nullptr, device));
      storage.set_nbytes(0);
      return;
    }

    // device guard?
    auto copy_stream = c10::xpu::getStreamFromPool(false, device.index());

    at::DataPtr data = allocator->allocate(new_size_bytes);
    if (storage.data_ptr()) {
      at::globalContext().lazyInitXPU();

      copy_stream.queue().memcpy(
          data.get(),
          storage.data(),
          std::min(storage.nbytes(), new_size_bytes));
    }

    // Destructively overwrite data_ptr
    storage.set_data_ptr_noswap(std::move(data));
    storage.set_nbytes(new_size_bytes);
  }
}

at::TensorBase resize_impl_xpu_(
    at::TensorBase& src,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool device_guard) {
  // TODO: device guards

  const auto itemsize = src.dtype().itemsize();
  const auto storage_offset = src.storage_offset();
  size_t storage_size = 1;
  auto impl = src.unsafeGetTensorImpl();
  if (stride) {
    impl->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    impl->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }
  maybe_resize_storage_xpu(impl, storage_size);

  // why does this return?
  return src;
}

} // namespace at::xpu