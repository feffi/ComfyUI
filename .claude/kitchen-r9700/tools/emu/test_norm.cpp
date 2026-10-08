// Runs kitchen's adaln and rms_rope launchers over fixed cases and dumps every output
// (case*.bin) and the fp32 value of each element before its narrowing store (case*.f32),
// so two kitchen trees can be compared byte for byte.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "ops/adaln.hip"
#include "ops/rms_rope.hip"

static std::mt19937 rng(1234);
static float rnd(float scale) { return std::uniform_real_distribution<float>(-scale, scale)(rng); }

template <typename T>
static std::vector<uint8_t> make(size_t n, float scale) {
    std::vector<uint8_t> b(n * sizeof(T) + 64);
    T* p = reinterpret_cast<T*>(b.data());
    for (size_t i = 0; i < n; ++i) {
        float v = rnd(scale);
        if (i % 97 == 0) v = 0.0f;
        if (i % 131 == 0) v *= 40.0f;
        p[i] = static_cast<T>(v);
    }
    return b;
}
static std::vector<uint8_t> make_code(int code, size_t n, float scale) {
    if (code == 0) return make<float>(n, scale);
    if (code == 1) return make<_Float16>(n, scale);
    return make<__bf16>(n, scale);
}
static int esize(int code) { return code == 0 ? 4 : 2; }

// Shadow of each output element: the fp32 value before its narrowing store.
static void shadow_dump(const std::string& dir, int idx, const uint8_t* base, size_t n, int es) {
    std::vector<uint32_t> sh(n, 0xffffffffu);
    for (size_t j = 0; j < n; ++j) {
        auto it = emu::shadow.find(reinterpret_cast<uintptr_t>(base + j * es));
        if (it != emu::shadow.end()) memcpy(&sh[j], &it->second, 4);
    }
    char name[512];
    snprintf(name, sizeof(name), "%s/case%03d.f32", dir.c_str(), idx);
    FILE* fp = fopen(name, "wb");
    fwrite(sh.data(), 4, n, fp);
    fclose(fp);
}

static void dump(const std::string& dir, int idx, const void* p, size_t bytes) {
    char name[512];
    snprintf(name, sizeof(name), "%s/case%03d.bin", dir.c_str(), idx);
    FILE* f = fopen(name, "wb");
    fwrite(p, 1, bytes, f);
    fclose(f);
}

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    int idx = 0;
    // adaln: N, D, code, scale code, shift code, subtract_mean, scale_group, shift_group, misalign
    struct A { int N, D, code, sc, hc; bool mean; int sg, hg, mis; };
    const A acases[] = {
        {7, 6144, 2, 2, 2, false, 7, 7, 0},   {9, 6144, 2, 2, 2, false, 1, 9, 0},
        {5, 256, 1, 1, 1, false, 5, 5, 0},    {3, 3072, 2, 2, 2, true, 3, 1, 0},
        {11, 512, 1, 1, 1, true, 1, 11, 0},   {4, 1000, 2, 2, 2, false, 4, 4, 0},
        {4, 6144, 2, 0, 0, false, 4, 4, 0},   {4, 6144, 0, 0, 0, false, 4, 4, 0},
        {3, 512, 2, 2, 2, false, 3, 3, 2},    {17, 1536, 2, 2, 2, false, 17, 17, 0},
    };
    for (const A& c : acases) {
        auto x = make_code(c.code, (size_t)c.N * c.D, 3.0f);
        auto s = make_code(c.sc, (size_t)((c.N - 1) / c.sg + 1) * c.D, 0.5f);
        auto h = make_code(c.hc, (size_t)((c.N - 1) / c.hg + 1) * c.D, 0.5f);
        std::vector<uint8_t> out((size_t)c.N * c.D * esize(c.code) + 64, 0xCD);
        launch_adaln_kernel(x.data() + c.mis, s.data(), h.data(), out.data() + c.mis, c.N, c.D, c.sg, c.hg,
                            1e-6f, c.code, c.sc, c.hc, c.mean, nullptr);
        fprintf(stderr, "adaln case %d N=%d D=%d grid=%u block=%u\n", idx, c.N, c.D, emu::gdim.x, emu::bdim.x);
        shadow_dump(dir, idx, out.data() + c.mis, (size_t)c.N * c.D, esize(c.code));
        dump(dir, idx++, out.data(), out.size());
        emu::shadow.clear();
    }

    // rms_rope: q (and k at q + k_off in the same buffer, as a qkv slice) read through
    // strides (sx_b, sx_d1, sx_d2, 1); outputs contiguous, or in place. freqs is a
    // contiguous (fb, fd1, fd2, rot / 2, 2, 2) table.
    struct R {
        int64_t batch, dim1, dim2, head_dim, rot, sx_b, sx_d1, sx_d2, k_off;
        bool in_place;
        int64_t fb, fd1, fd2;
        int x_code, f_code, s_code;
        bool split;
        int mis;
    };
    const int64_t N = 301, H = 6, M = 203, G = 7;
    const R rcases[] = {
        // Krea 2: BHND view of (1, N, H * 128), q alone, fp32 freqs, interleaved
        {1, H, N, 128, 0, N * H * 128, 128, H * 128, -1, false, 1, 1, N, 2, 0, 2, false, 0},
        // the same with k beside q
        {1, H, N, 128, 0, N * 2 * H * 128, 128, 2 * H * 128, H * 128, false, 1, 1, N, 2, 0, 2, false, 0},
        // MiniMax H3: BNHD slice of qkv, in place, bf16 freqs, split-half rot 96
        {1, M, G, 128, 96, M * 3 * G * 128, 3 * G * 128, 128, G * 128, true, 1, M, 1, 2, 2, 2, true, 0},
        // fp16 x and freqs, fp32 weights, split-half full rot, batch 2, BNHD
        {2, M, G, 128, 0, M * G * 128, G * 128, 128, -1, false, 1, M, 1, 1, 1, 0, true, 0},
        // bf16 freqs per batch, interleaved partial rot 64, BHND with k
        {2, 5, 11, 128, 64, 2 * 5 * 11 * 128, 11 * 128, 128, 5 * 11 * 128, false, 2, 1, 11, 2, 2, 2, false, 0},
        // freqs broadcast over both dims, fp32 freqs, split-half rot 96, with k
        {1, 5, 9, 128, 96, 2 * 5 * 9 * 128, 9 * 128, 128, 5 * 9 * 128, false, 1, 1, 1, 2, 0, 2, true, 0},
        // one rotated pair, interleaved; one-lane halves, split-half
        {1, 3, 13, 128, 2, 3 * 13 * 128, 13 * 128, 128, -1, false, 1, 1, 13, 1, 0, 1, false, 0},
        {1, 3, 13, 128, 8, 3 * 13 * 128, 13 * 128, 128, -1, false, 1, 1, 13, 2, 2, 2, true, 0},
        // fallbacks: head_dim 64, per-head freqs, misaligned q, fp32 x, rot 100 split-half
        {1, 4, 9, 64, 0, 4 * 9 * 64, 9 * 64, 64, -1, false, 1, 1, 9, 2, 0, 2, false, 0},
        {1, 4, 9, 128, 0, 4 * 9 * 128, 9 * 128, 128, -1, false, 1, 4, 9, 2, 0, 2, false, 0},
        {1, 4, 9, 128, 0, 4 * 9 * 128, 9 * 128, 128, -1, false, 1, 1, 9, 2, 0, 2, false, 1},
        {1, 4, 9, 128, 0, 4 * 9 * 128, 9 * 128, 128, -1, false, 1, 1, 9, 0, 0, 0, false, 0},
        {1, 4, 9, 128, 100, 4 * 9 * 128, 9 * 128, 128, -1, false, 1, 1, 9, 2, 2, 2, true, 0},
    };
    for (const R& c : rcases) {
        const int64_t rot = c.rot ? c.rot : c.head_dim;
        const int64_t span = (c.batch - 1) * c.sx_b + (c.dim1 - 1) * c.sx_d1 + (c.dim2 - 1) * c.sx_d2 +
                             c.head_dim + (c.k_off > 0 ? c.k_off : 0) + c.mis;
        auto x = make_code(c.x_code, (size_t)span, 3.0f);
        const int64_t np = rot / 2;
        auto f = make_code(c.f_code, (size_t)(c.fb * c.fd1 * c.fd2 * np * 4), 1.0f);
        auto qs = make_code(c.s_code, (size_t)c.head_dim, 0.5f);
        auto ks = make_code(c.s_code, (size_t)c.head_dim, 0.5f);
        const int es = esize(c.x_code);
        const size_t n_out = (size_t)(c.batch * c.dim1 * c.dim2 * c.head_dim);
        std::vector<uint8_t> qo(n_out * es + 64, 0xCD), ko(n_out * es + 64, 0xCD);
        uint8_t* q = x.data() + c.mis * es;
        uint8_t* k = c.k_off >= 0 ? q + c.k_off * es : nullptr;
        const int64_t so_b = c.in_place ? c.sx_b : c.dim1 * c.dim2 * c.head_dim;
        const int64_t so_d1 = c.in_place ? c.sx_d1 : c.dim2 * c.head_dim;
        const int64_t so_d2 = c.in_place ? c.sx_d2 : c.head_dim;
        launch_rms_rope_kernel(q, k, f.data(), qs.data(), ks.data(), c.in_place ? q : qo.data(),
                               c.in_place ? k : ko.data(), c.batch, c.dim1, c.dim2, c.head_dim, c.rot,
                               c.fb, c.fd1, c.fd2, c.sx_b, c.sx_d1, c.sx_d2, 1, so_b, so_d1, so_d2, 1,
                               c.fd1 * c.fd2 * np * 4, c.fd2 * np * 4, np * 4, 4, 2, 1, c.x_code, c.f_code,
                               c.s_code, 1e-6f, c.split, nullptr);
        fprintf(stderr, "rms_rope case %d dims=%lldx%lldx%lld D=%lld rot=%lld grid=%u block=%u\n", idx,
                (long long)c.batch, (long long)c.dim1, (long long)c.dim2, (long long)c.head_dim,
                (long long)rot, emu::gdim.x, emu::bdim.x);
        if (c.in_place) {
            shadow_dump(dir, idx, x.data(), x.size() / es, es);
            dump(dir, idx++, x.data(), x.size());
        } else {
            shadow_dump(dir, idx, qo.data(), n_out, es);
            dump(dir, idx++, qo.data(), qo.size());
            shadow_dump(dir, idx, ko.data(), n_out, es);
            dump(dir, idx++, ko.data(), ko.size());
        }
        emu::shadow.clear();
    }
    return 0;
}
