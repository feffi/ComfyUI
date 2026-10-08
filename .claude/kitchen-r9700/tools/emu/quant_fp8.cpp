// Kitchen's HIP per-tensor fp8 encode (pack_fp8 in fp8_utils.h, called as
// quantize_per_tensor_fp8_kernel calls it, at scale 1) over a range of input bit
// patterns. log2f is nudged by <ulps> ulps to stand in for the device's approximate
// v_log_f32; exp2f only sees integers and the divisions are by powers of two, so
// both are exact on the device too.
// usage: quant_fp8 <bf16|fp16|fp32> <ulps> <first pattern> <count> <out file>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_ulps = 0;
static float nudged_log2f(float x) {
    float r = std::log2(x);
    for (int i = 0; i < std::abs(g_ulps); ++i) r = std::nextafter(r, g_ulps > 0 ? INFINITY : -INFINITY);
    return r;
}
#define log2f nudged_log2f
#include "fp8_utils.h"

using namespace comfy::hip_backend;

template <typename T>
static float load(uint32_t bits) {
    T v;
    memcpy(&v, &bits, sizeof(T));
    return to_float(v);
}

int main(int argc, char** argv) {
    const char* type = argv[1];
    g_ulps = atoi(argv[2]);
    const uint32_t first = strtoul(argv[3], nullptr, 0), count = strtoul(argv[4], nullptr, 0);
    const float scale = 1.0f;
    std::vector<uint8_t> out(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t bits = first + i;
        const float x = !strcmp(type, "bf16") ? load<__bf16>(bits) : !strcmp(type, "fp16") ? load<__half>(bits) : load<float>(bits);
        out[i] = pack_fp8(x / scale, kFp8E4M3Code);
    }
    FILE* f = fopen(argv[5], "wb");
    fwrite(out.data(), 1, count, f);
    fclose(f);
}
