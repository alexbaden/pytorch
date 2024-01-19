#include <c10/xpu/XPUGuard.h>

namespace c10 {
namespace xpu {

C10_REGISTER_GUARD_IMPL(XPU, XPUGuard);

} // namespace xpu
} // namespace c10