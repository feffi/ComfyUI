// Host-side SIMT emulation of the subset of HIP the kitchen GEMM kernels use.
// Blocks run one after another; the threads of a block run as std::threads with
// a block barrier (__syncthreads) and per-wave barriers (shuffles, WMMA).
#pragma once
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))

typedef void* hipStream_t;
typedef int hipError_t;
constexpr int hipSuccess = 0;
enum { hipDeviceAttributeMultiprocessorCount = 1 };

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
struct emu_uint3 { unsigned x, y, z; };
struct alignas(16) uint4 { unsigned x, y, z, w; };
struct alignas(8) uint2 { unsigned x, y; };
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return {a, b, c, d}; }
inline uint2 make_uint2(unsigned a, unsigned b) { return {a, b}; }

typedef _Float16 __half;
inline float __half2float(__half h) { return (float)h; }
inline __half __float2half(float f) { return (__half)f; }
inline unsigned __float_as_uint(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
inline float __uint_as_float(unsigned u) { float f; memcpy(&f, &u, 4); return f; }

namespace emu {
extern thread_local emu_uint3 tid, bid;
extern thread_local int lane, wave;
extern emu_uint3 bdim, gdim;
extern int wgps;
extern std::barrier<>* block_bar;
extern std::vector<std::unique_ptr<std::barrier<>>> wave_bars;
struct WaveSlots { alignas(32) unsigned char raw[32][64]; };
extern std::vector<WaveSlots> slots;
extern long launches;

inline void wave_sync() { wave_bars[wave]->arrive_and_wait(); }

template <typename T>
T shfl_xor(T v, int off) {
    auto& s = slots[wave];
    memcpy(s.raw[lane], &v, sizeof(T));
    wave_sync();
    T r;
    memcpy(&r, s.raw[lane ^ off], sizeof(T));
    wave_sync();
    return r;
}

inline dim3 as_dim3(dim3 d) { return d; }
inline dim3 as_dim3(int x) { return dim3(x); }

template <typename K, typename G, typename B, typename... Args>
void launch(K kernel, G grid_, B block_, int /*shmem*/, hipStream_t, Args... args) {
    const dim3 grid = as_dim3(grid_), block = as_dim3(block_);
    const unsigned nthreads = block.x * block.y * block.z;
    ++launches;
    bdim = {block.x, block.y, block.z};
    gdim = {grid.x, grid.y, grid.z};
    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
        std::barrier<> bb(nthreads);
        block_bar = &bb;
        const unsigned nwaves = (nthreads + 31) / 32;
        wave_bars.clear();
        for (unsigned w = 0; w < nwaves; ++w)
            wave_bars.emplace_back(new std::barrier<>(std::min(32u, nthreads - 32 * w)));
        slots.assign(nwaves, WaveSlots{});
        std::vector<std::thread> ts;
        ts.reserve(nthreads);
        for (unsigned t = 0; t < nthreads; ++t) {
            ts.emplace_back([&, t] {
                tid = {t % block.x, (t / block.x) % block.y, t / (block.x * block.y)};
                bid = {bx, by, bz};
                lane = t % 32;
                wave = t / 32;
                kernel(args...);
            });
        }
        for (auto& th : ts) th.join();
    }
}
}  // namespace emu

#define threadIdx emu::tid
#define blockIdx emu::bid
#define blockDim emu::bdim
#define gridDim emu::gdim

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
template <typename C> C emu_unsupported(C c) { abort(); return c; }
#define __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, c) emu_unsupported(c)
#define __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(sa, a, sb, b, c, cl) emu_unsupported(c)
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c) emu_unsupported(c)
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c) emu_unsupported(c)
inline void __syncthreads() { emu::block_bar->arrive_and_wait(); }
template <typename T> T __shfl_xor(T v, int off, int = 32) { return emu::shfl_xor(v, off); }

typedef int* hipEvent_t;
enum hipStreamCaptureStatus { hipStreamCaptureStatusNone = 0, hipStreamCaptureStatusActive = 1 };
namespace emu { extern long event_clock; }
inline hipError_t hipEventCreate(hipEvent_t* e) { *e = new int(0); return hipSuccess; }
inline hipError_t hipEventDestroy(hipEvent_t e) { delete e; return hipSuccess; }
inline hipError_t hipEventRecord(hipEvent_t e, hipStream_t) { *e = (int)emu::launches; return hipSuccess; }
inline hipError_t hipEventSynchronize(hipEvent_t) { return hipSuccess; }
// Elapsed "time" is the launch count between the events, so every candidate ties
// and the first eligible one wins; the emulator checks results, not speed.
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t a, hipEvent_t b) { *ms = (float)(*b - *a); return hipSuccess; }
inline hipError_t hipStreamIsCapturing(hipStream_t, hipStreamCaptureStatus* s) { *s = hipStreamCaptureStatusNone; return hipSuccess; }
inline hipError_t hipGetDevice(int* d) { *d = 0; return hipSuccess; }
inline hipError_t hipDeviceGetAttribute(int* v, int, int) { *v = emu::wgps; return hipSuccess; }

// ---- gfx12 WMMA, wave32 -------------------------------------------------------
// A: lane l holds row l%16, K bytes [8*(l/16), +8). B: lane l holds column l%16,
// the same K bytes. D: lane l holds column l%16, rows e + 8*(l/16).
typedef int emu_v2i __attribute__((ext_vector_type(2)));
typedef float emu_v8f __attribute__((ext_vector_type(8)));
typedef int emu_v8i __attribute__((ext_vector_type(8)));
namespace emu {
float fp8_e4m3(uint8_t b);
template <typename Acc, typename Dec>
Acc wmma16(emu_v2i a, emu_v2i b, Acc c, Dec dec) {
    auto& s = slots[wave];
    memcpy(s.raw[lane], &a, 8);
    memcpy(s.raw[lane] + 8, &b, 8);
    wave_sync();
    Acc d;
    const int n = lane % 16;
    for (int e = 0; e < 8; ++e) {
        const int m = e + 8 * (lane / 16);
        double acc = 0;
        for (int k = 0; k < 16; ++k) {
            const uint8_t av = s.raw[(k / 8) * 16 + m][k % 8];
            const uint8_t bv = s.raw[(k / 8) * 16 + n][8 + k % 8];
            acc += (double)dec(av) * (double)dec(bv);
        }
        using E = std::remove_cvref_t<decltype(c[0])>;
        d[e] = c[e] + static_cast<E>(acc);
    }
    wave_sync();
    return d;
}
}  // namespace emu
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c) \
    emu::wmma16<emu_v8f>(a, b, c, [](uint8_t v) { return emu::fp8_e4m3(v); })
#define __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(sa, a, sb, b, c, clamp) \
    emu::wmma16<emu_v8i>(a, b, c, [](uint8_t v) { return (int)(int8_t)v; })
typedef float emu_v2f __attribute__((ext_vector_type(2)));
inline emu_v2f __builtin_amdgcn_cvt_pk_f32_fp8(int w, bool hi) {
    const unsigned u = (unsigned)w >> (hi ? 16 : 0);
    return emu_v2f{emu::fp8_e4m3(u & 0xff), emu::fp8_e4m3((u >> 8) & 0xff)};
}
inline int __builtin_amdgcn_sudot4(bool, int a, bool, int b, int c, bool) {
    for (int i = 0; i < 4; ++i) c += (int8_t)(a >> (8 * i)) * (int8_t)(b >> (8 * i));
    return c;
}
