#include <cstdio>
#include <cstdlib>
#include <random>
#include "ops/gemm_fp8.hip"

int main(int argc, char** argv) {
    const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
    const int out_code = atoi(argv[4]);   // 0 f32, 1 f16, 2 bf16
    const int bias_code = atoi(argv[5]);  // -1 none, else dtype code
    const unsigned seed = atoi(argv[6]);
    emu::wgps = atoi(argv[7]);
    const char* dump = argc > 8 ? argv[8] : nullptr;
    std::mt19937 rng(seed);
    auto fp8 = [&] { uint8_t v; do { v = rng() & 0xff; } while ((v & 0x7f) == 0x7f || ((v >> 3) & 15) > 10); return v; };
    std::vector<uint8_t> A((size_t)M * K), B((size_t)N * K);
    for (auto& v : A) v = fp8();
    for (auto& v : B) v = fp8();
    float sa = 0.75f, sb = 1.25f;
    std::vector<float> biasf(N);
    std::vector<__bf16> bias_bf(N);
    for (int n = 0; n < N; ++n) { biasf[n] = (float)(int)(rng() % 64) - 32.0f; bias_bf[n] = (__bf16)biasf[n]; }
    const void* bias = bias_code < 0 ? nullptr : (bias_code == 0 ? (const void*)biasf.data() : (const void*)bias_bf.data());
    const size_t osz = out_code == 0 ? 4 : 2;
    std::vector<uint8_t> C((size_t)M * N * osz, 0xAB);
    long before = emu::launches;
    launch_scaled_mm_fp8_kernel(A.data(), B.data(), C.data(), &sa, &sb, bias, bias_code < 0 ? 0 : bias_code,
                                M, N, K, out_code, nullptr);
    double maxerr = 0, maxref = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double r = 0;
            for (int k = 0; k < K; ++k) r += (double)emu::fp8_e4m3(A[(size_t)m * K + k]) * emu::fp8_e4m3(B[(size_t)n * K + k]);
            r = r * ((double)sa * sb) + (bias ? biasf[n] : 0.0);
            double o;
            const uint8_t* p = &C[((size_t)m * N + n) * osz];
            if (out_code == 0) { float f; memcpy(&f, p, 4); o = f; }
            else if (out_code == 1) { _Float16 h; memcpy(&h, p, 2); o = (double)h; }
            else { __bf16 h; memcpy(&h, p, 2); o = (double)(float)h; }
            maxerr = std::max(maxerr, std::abs(o - r));
            maxref = std::max(maxref, std::abs(r));
        }
    const double tol = (out_code == 0 ? 1e-5 : out_code == 1 ? 2e-3 : 1e-2) * maxref + 1e-3;
    printf("M=%d N=%d K=%d out=%d bias=%d wgps=%d launches=%ld maxerr=%.4g maxref=%.4g %s\n", M, N, K, out_code,
           bias_code, emu::wgps, emu::launches - before, maxerr, maxref, maxerr <= tol ? "OK" : "FAIL");
    if (dump) { FILE* f = fopen(dump, "wb"); fwrite(C.data(), 1, C.size(), f); fclose(f); }
    return maxerr <= tol ? 0 : 1;
}
