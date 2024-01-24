#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/TensorIteratorDynamicCasting.h>

#include <sycl/sycl.hpp>


namespace at::native {

#if 0

namespace {

enum class OpType { GE, GT, LE, LT };

template <typename scalar_t>
struct CompareFunctor {
  constexpr CompareFunctor(OpType op) : op_(op){};
  OpType op_;
  bool operator()(scalar_t a, scalar_t b) const {
    if (op_ == OpType::GE) {
      return a >= b;
    } else if (op_ == OpType::GT) {
      return a > b;
    } else if (op_ == OpType::LE) {
      return a <= b;
    } else { // LT
      return a < b;
    }
  }
};

// Reflects the comparison operator, so reflect(op)(a, b) == op(b, a)
OpType reflect(OpType x) {
  switch (x) {
    case OpType::GE:
      return OpType::LE;
    case OpType::GT:
      return OpType::LT;
    case OpType::LE:
      return OpType::GE;
    case OpType::LT:
      return OpType::GT;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid OpType");
}

template <typename func_t, typename array_t, typename inp_calc_t, int vec_size>
struct VectorizedElementwiseKernelFunctor {
  void operator()(sycl::nd_item<1> itemId) const {
    {
      vectorized_elementwise_kernel<
          vec_size,
          vec_size,
          func_t,
          array_t,
          inp_calc_t>(itemId, N, fn, data, input_calc);
    }
  }
  VectorizedElementwiseKernelFunctor(
      int64_t N_,
      const func_t fn_,
      array_t data_,
      inp_calc_t input_calc_)
      : N(N_), fn(fn_), data(data_), input_calc(input_calc_) {}

 private:
  int64_t N;
  const func_t fn;
  array_t data;
  inp_calc_t input_calc;
};

// Assumption:
// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t, typename inp_calc_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& fn,
    array_t data,
    inp_calc_t input_calc,
    int vec_size,
    sycl::queue queue, size_t group_size) {
  constexpr auto max_scalar_bytes = max_scalar_size<func_t>();
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  auto vec_loops_kernel =
      [&](vec_size) {
        TORCH_CHECK(max_scalar_bytes * vec_size <= 16);
        if constexpr (max_scalar_bytes * vec_size <= 16) {
          auto cgf = [&](sycl::handler & h) {
            int group_work_size = group_size * vec_size;
            int num_groups = (N + group_work_size - 1) / group_work_size;
            VectorizedElementwiseKernelFunctor<
                func_t,
                array_t,
                inp_calc_t,
                vec_size>
                kfn(N, fn, data, input_calc);
            queue.parallel_for<decltype(kfn)>(
                sycl::nd_range<1>(
                    sycl::range<1>(num_groups * group_size),
                    sycl::range<1>(group_size)),
                kfn);
          };
          queue->submit(cgf);
        }
      }

  switch (vec_size) {
    case 16: {
      vec_loops_kernel(16);
      break;
    }
    case 8: {
      vec_loops_kernel(8);
      break;
    }
    case 4: {
      vec_loops_kernel(4);
      break;
    }
    case 2: {
      vec_loops_kernel(2);
      break;
    }
    case 1: {
      vec_loops_kernel(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size", vec_size);
  }

#undef VEC_LOOPS_KERNEL
}

template <typename T>
uint32_t dpcppPrefVectorWidth(DeviceProp* dev_prop) {
  if (std::is_same<T, char>::value) {
    return dev_prop->pref_vec_width_char;
  }
  if (std::is_same<T, short>::value) {
    return dev_prop->pref_vec_width_short;
  }
  if (std::is_same<T, int>::value) {
    return dev_prop->pref_vec_width_int;
  }
  if (std::is_same<T, int64_t>::value) {
    return dev_prop->pref_vec_width_long;
  }
  if (std::is_same<T, float>::value) {
    return dev_prop->pref_vec_width_float;
  }
  if (std::is_same<T, double>::value) {
    return dev_prop->pref_vec_width_double;
  }
  if (std::is_same<T, sycl::half>::value) {
    return dev_prop->pref_vec_width_half;
  }
  throw std::invalid_argument(
      "Invalid data type to fetch preferred vector width!");
}

static inline int preferred_vector_width(DeviceProp* dev_prop, int elem_sz) {
  size_t ret;
  switch (elem_sz) {
    case 1:
      static_assert(sizeof(char) == 1, "the char size is not 1 bytes");
      ret = dpcppPrefVectorWidth<char>(dev_prop);
      break;
    case 2:
      static_assert(sizeof(short) == 2, "the short size is not 2 bytes");
      ret = dpcppPrefVectorWidth<short>(dev_prop);
      break;
    case 4:
      ret = dpcppPrefVectorWidth<int>(dev_prop);
      static_assert(sizeof(int) == 4, "the long size is not 4 bytes");
      break;
    case 8:
      static_assert(sizeof(int64_t) == 8, "the long size is not 8");
      ret = dpcppPrefVectorWidth<int64_t>(dev_prop);
      break;
    default:
      // no vectorize
      ret = 1;
  }
  return ret;
}

template <typename scalar_t>
inline int can_vectorize_up_to_loop(DeviceId dev_id, char* pointer) {
  int elem_size = sizeof(scalar_t);
  int preferred_width = preferred_vector_width(dev_id, elem_size);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment =
      std::alignment_of<aligned_vector_loop<scalar_t, 2>>::value;
  constexpr int vec4_alignment =
      std::alignment_of<aligned_vector_loop<scalar_t, 4>>::value;
  constexpr int vec8_alignment =
      std::alignment_of<aligned_vector_loop<scalar_t, 8>>::value;
  constexpr int vec16_alignment =
      std::alignment_of<aligned_vector_loop<scalar_t, 16>>::value;
  if (address % vec16_alignment == 0) {
    return std::min<int>(preferred_width, 16);
  } else if (address % vec8_alignment == 0) {
    return std::min<int>(preferred_width, 8);
  } else if (address % vec4_alignment == 0) {
    return std::min<int>(preferred_width, 4);
  } else if (address % vec2_alignment == 0) {
    return std::min<int>(preferred_width, 2);
  }
  return 1;
}

template <typename func_t>
void xpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  TORCH_INTERNAL_ASSERT(iter.device().is_xpu());
  auto device = iter.device();
  auto queue = c10::xpu::getStreamFromPool(false, device.index());
  // TODO: centralize, maybe in XPUFunctions.h 
  DeviceProp device_prop;
  c10::xpu::get_device_properties(&device_prop, device.index());
  auto subgroup_sizes = dev_prop.subgroup_sizes;
  int64_t simd_width = std::max_element(subgroup_sizes);
  int64_t hw_threads = dev_prop.gpu_hw_threads_per_eu;
  int64_t group_size = simd_width * hw_threads;

  if (contiguous) {
    int vec_size = can_vectorize_up_to_loop<func_t>(&device_prop, data);
    auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
    return launch_vectorized_kernel(numel, f, data, input_offset_calculator, vec_size, queue, group_size);
  }
  TORCH_INTERNAL_ASSERT(false);
#if 0
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  launch_legacy_kernel<128, unroll_factor>(numel, [=] (int idx) {
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke(f, &data.data[1], &offsets.data[1], 1);
  });
#endif
}

template <typename func_t>
void xpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  // TODO: looks like an optimization that can be implemented later
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return xpu_kernel_impl_nocast(iter, f);
  }

#if 0
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
#if 0
    auto loader = memory::LoadWithCast<traits::arity>(iter);
    auto storer = memory::StoreWithCast<1>(iter);
    auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_unrolled_kernel(
        numel,
        f,
        data,
        input_offset_calculator,
        output_offset_calculator,
        loader,
        storer);
#endif
  } else {
#if 0
    at::detail::Array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
    launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
      auto offsets = offset_calc.get(idx);
      void* out = data[0] + offsets[0];
      arg0_t result =
          invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
#endif
  }
#endif
}

template <typename func_t>
void xpu_kernel(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
        iter.device(arg).is_xpu(),
        "argument ",
        arg,
        ": expected a XPU device but found ",
        iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      xpu_kernel(sub_iter, f);
    }
    return;
  }

  xpu_kernel_impl(iter, f);
}

} // namespace

template <typename scalar_t>
void compare_scalar_kernel(TensorIteratorBase& iter, OpType op, scalar_t rhs) {
  CompareFunctor<scalar_t> f(op);
  xpu_kernel(iter, [=](scalar_t lhs) -> bool { return f(lhs, rhs); });
}

template <typename scalar_t>
void compare_kernel_impl(TensorIteratorBase& iter, OpType op) {
  // If either input is a cpu scalar, perform the equivalent comparison
  // where the scalar is on the right hand side. This saves us from
  // generating two otherwise identical kernels with mirrored
  // arguments.
  if (iter.is_cpu_scalar(1)) {
    const scalar_t lhs = iter.scalar_value<scalar_t>(1);
    iter.remove_operand(1);
    // TODO: device guard?
    // const DeviceGuard device_guard(iter.device(1));
    compare_scalar_kernel(iter, reflect(op), lhs);
  } else if (iter.is_cpu_scalar(2)) {
    const scalar_t rhs = iter.scalar_value<scalar_t>(2);
    iter.remove_operand(2);
    compare_scalar_kernel(iter, op, rhs);
  } else {
    CompareFunctor<scalar_t> f(op);
    xpu_kernel(iter, f);
  }
}

C10_NOINLINE void compare_kernel_with_scalars(
    TensorIteratorBase& iter,
    OpType op) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "compare_xpu", [&]() {
        compare_kernel_impl<scalar_t>(iter, op);
      });
}

#endif 


void ge_kernel_xpu(TensorIteratorBase& iter) {
    // TODO: fails silently 
//   compare_kernel_with_scalars(iter, OpType::GE);
}

void gt_kernel_xpu(TensorIteratorBase& iter) {
    // TODO: fails silently 
//   compare_kernel_with_scalars(iter, OpType::GT);
}

void le_kernel_xpu(TensorIteratorBase& iter) {
    // TODO: fails silently 
//   compare_kernel_with_scalars(iter, OpType::LE);
}

void lt_kernel_xpu(TensorIteratorBase& iter) {
    // TODO: fails silently 
//   compare_kernel_with_scalars(iter, OpType::LT);
}

REGISTER_XPU_DISPATCH(ge_stub, &ge_kernel_xpu);
REGISTER_XPU_DISPATCH(gt_stub, &gt_kernel_xpu);
REGISTER_XPU_DISPATCH(le_stub, &le_kernel_xpu);
REGISTER_XPU_DISPATCH(lt_stub, &lt_kernel_xpu);

} // namespace at::native