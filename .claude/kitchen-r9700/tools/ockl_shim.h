// Inline stand-ins for the device-lib work-item queries (code object v5 layout),
// so a -nogpulib compile emits what a ROCm build inlines from ockl.bc.
#pragma once
#include <stddef.h>
#include <stdint.h>
extern "C" __attribute__((device, always_inline)) inline size_t __ockl_get_local_id(unsigned d) {
    return d == 0 ? __builtin_amdgcn_workitem_id_x() : d == 1 ? __builtin_amdgcn_workitem_id_y() : __builtin_amdgcn_workitem_id_z();
}
extern "C" __attribute__((device, always_inline)) inline size_t __ockl_get_group_id(unsigned d) {
    return d == 0 ? __builtin_amdgcn_workgroup_id_x() : d == 1 ? __builtin_amdgcn_workgroup_id_y() : __builtin_amdgcn_workgroup_id_z();
}
extern "C" __attribute__((device, always_inline)) inline size_t __ockl_get_local_size(unsigned d) {
    const uint16_t* p = (const uint16_t*)((const char*)__builtin_amdgcn_implicitarg_ptr() + 12);
    return p[d];
}
extern "C" __attribute__((device, always_inline)) inline size_t __ockl_get_num_groups(unsigned d) {
    const uint32_t* p = (const uint32_t*)__builtin_amdgcn_implicitarg_ptr();
    return p[d];
}
