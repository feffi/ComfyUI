"""Copy a kitchen HIP source tree and turn kernel<<<g, b, s, st>>>(args) into emu::launch(kernel, g, b, s, st, args)."""
import re, shutil, sys, pathlib
src, dst = map(pathlib.Path, sys.argv[1:3])
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst)
SHIPPED = {
    "rms_rope.hip": [
        ("sq_q += v * v;", "sq_q = __builtin_fmaf(v, v, sq_q);"),
        ("sq_k += w * w;", "sq_k = __builtin_fmaf(w, w, sq_k);"),
        ("rsqrtf(red_q[0] * inv_d + epsilon)", "rsqrtf(__builtin_fmaf(red_q[0], inv_d, epsilon))"),
        ("rsqrtf(red_k[0] * inv_d + epsilon)", "rsqrtf(__builtin_fmaf(red_k[0], inv_d, epsilon))"),
        ("load_in(q, ia_in, x_code) * rrms_q * sa", "rrms_q * sa * load_in(q, ia_in, x_code)"),
        ("load_in(q, ib_in, x_code) * rrms_q * sb", "rrms_q * sb * load_in(q, ib_in, x_code)"),
        ("load_in(k, ia_in, x_code) * rrms_k * ta", "rrms_k * ta * load_in(k, ia_in, x_code)"),
        ("load_in(k, ib_in, x_code) * rrms_k * tb", "rrms_k * tb * load_in(k, ib_in, x_code)"),
    ],
    "rope_math.h": [("return f_a * x_a + f_b * x_b;", "return __builtin_fmaf(f_b, x_b, f_a * x_a);")],
    "adaln.hip": [
        ("sq += d * d;", "sq = __builtin_fmaf(d, d, sq);"),
        ("rsqrtf(red_sq[0] * inv_d + eps)", "rsqrtf(__builtin_fmaf(red_sq[0], inv_d, eps))"),
        ("store_out<T>(out, xbase + i, norm * (1.0f + s) + h);", "store_out<T>(out, xbase + i, __builtin_fmaf(norm, 1.0f + s, h));"),
    ],
}
# Record the fp32 value of every narrowing store, so orderings that differ by an ulp
# show up before the bf16/fp16 rounding hides them. The fast path packs its stores
# in registers, so its hook records the destination.
SHADOW = {
    "adaln.hip": [
        ("store_out<__half>(__half* p, int64_t i, float v) {", "store_out<__half>(__half* p, int64_t i, float v) {\n    emu::rec(p, i, v);"),
        ("store_out<__bf16>(__bf16* p, int64_t i, float v) {", "store_out<__bf16>(__bf16* p, int64_t i, float v) {\n    emu::rec(p, i, v);"),
    ],
    "rope_math.h": [
        ("rope_store<__half>(__half* p, int64_t i, float v) {", "rope_store<__half>(__half* p, int64_t i, float v) {\n    emu::rec(p, i, v);"),
        ("rope_store<__bf16>(__bf16* p, int64_t i, float v) {", "rope_store<__bf16>(__bf16* p, int64_t i, float v) {\n    emu::rec(p, i, v);"),
    ],
}
SHADOW_OPT = {
    "adaln.hip": [("store_out<T>(res, j, __builtin_fmaf(norm, 1.0f + s[j], h[j]));",
                   "{ const float sv_ = __builtin_fmaf(norm, 1.0f + s[j], h[j]); emu::rec(out + xbase + o, j, sv_); store_out<T>(res, j, sv_); }")],
    "rms_rope.hip": [("for (int i = 0; i < 4; ++i) rope_store<T>(packed, i, res[i]);",
                      "for (int i = 0; i < 4; ++i) { emu::rec(out, i, res[i]); rope_store<T>(packed, i, res[i]); }")],
}
pat = re.compile(r"([A-Za-z_][\w:]*\s*(?:<[^;{}()]*?>)?)\s*<<<\s*(.*?)>>>\s*\(", re.S)
for p in dst.rglob("*"):
    if p.suffix in (".h", ".hip", ".cpp"):
        t = orig = p.read_text()
        t = t.replace('asm("v_cvt_f16_f32 %0, %1\\n\\tv_cvt_f32_f16 %0, %0" : "=v"(r) : "v"(x));', 'r = (float)(_Float16)x;')
        t = re.sub(r"extern __shared__ ([\w ]+?) (\w+)\[\];", r"__shared__ \1 \2[65536];", t)
        # rms_rope_kernel and rope_combine as 0.2.36's gfx1201 code object evaluates
        # them (read off its ISA), so the host model rounds like the shipped binary.
        for a, b in SHIPPED.get(p.name, []) + SHADOW.get(p.name, []):
            if t.count(a) != 1: sys.exit(f"{p.name}: expected one '{a}'")
            t = t.replace(a, b)
        for a, b in SHADOW_OPT.get(p.name, []):
            if t.count(a) > 1: sys.exit(f"{p.name}: expected at most one '{a}'")
            t = t.replace(a, b)
        n = pat.sub(lambda m: f"emu::launch({m.group(1).strip()}, {m.group(2).strip()}, ", t)
        if n != orig: p.write_text(n)
