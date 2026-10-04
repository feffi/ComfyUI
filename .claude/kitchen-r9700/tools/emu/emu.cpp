#include <hip/hip_runtime.h>
namespace emu {
thread_local emu_uint3 tid, bid;
thread_local int lane, wave;
emu_uint3 bdim, gdim;
int wgps = 32;
std::barrier<>* block_bar;
std::vector<std::unique_ptr<std::barrier<>>> wave_bars;
std::vector<WaveSlots> slots;
long launches;
float fp8_e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
    return s ? -v : v;
}
}  // namespace emu
