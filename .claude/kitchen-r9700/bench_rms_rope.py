# ruff: noqa: T201
"""rms_rope microbenchmark for comfy-kitchen's HIP kernels (patch 0006).

Shapes: Krea 2 at 1280x2048 as the model calls it (rms_rope with q 48 heads and k 12
heads as BHND views of the projections, fp32 freqs, so two launches), MiniMax H3 as it
calls it (rms_rope_split_half_ in place on BNHD slices of the qkv buffer, 56 heads,
rot_dim 96, bf16 freqs), then partial, odd and fallback shapes.

    python bench_rms_rope.py --device 1 --save-ref C:\\kt\\rope_0236.pt   # installed kitchen
    set PYTHONPATH=C:\\kt\\kitchen-patched
    python bench_rms_rope.py --device 1 --ref C:\\kt\\rope_0236.pt        # patched kitchen

GB/s counts what a launch has to move once (q/k read and written, freqs read) against
the ~600 GB/s practical peak. "fast" mirrors the launcher's test for 0006's head_dim 128
path per launch (q, then k); "-" runs rms_rope_kernel. The reference stores a SHA-256 of
every output plus a sample, so --ref reports bit identity; "err64" is the max |error| of
the first 64 tokens against an fp64 evaluation, for both builds.
"""

import argparse
import json
import os

import torch

from bench_fp8_gemm import digest, time_ms

PEAK_GBS = 600.0
BF16, FP16, FP32 = torch.bfloat16, torch.float16, torch.float32


def rotation_table(lead, pairs, dtype, device, g):
    """(..., pairs, 2, 2) rotation matrices [[cos, -sin], [sin, cos]] of random angles."""
    ang = torch.randn(*lead, pairs, device=device, generator=g, dtype=torch.float64) * 3
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack([c, -s, s, c], dim=-1).reshape(*lead, pairs, 2, 2).to(dtype)


def weight(dtype, device, g, d=128):
    return (1 + 0.2 * torch.randn(d, device=device, generator=g)).to(dtype)


def bhnd(t, heads, dtype, device, g, d=128):
    """Krea 2's layout: (1, heads, t, d) view of a (1, t, heads * d) projection."""
    x = torch.randn(1, t, heads * d, device=device, generator=g).mul_(3).to(dtype)
    return x.unflatten(-1, (heads, d)).transpose(1, 2)


def cases():
    out = []

    def krea2(t, x_dtype=BF16, f_dtype=FP32, per_head=False, d=128):
        def make(dev, g):
            q, k = bhnd(t, 48, x_dtype, dev, g, d), bhnd(t, 12, x_dtype, dev, g, d)
            freqs = rotation_table((1, 48 if per_head else 1, t), d // 2, f_dtype, dev, g)
            if per_head:  # k's 12 heads read the first 12 heads' table
                return dict(q=q, k=k, fq=freqs, fk=freqs[:, :12], qw=weight(x_dtype, dev, g, d),
                            kw=weight(x_dtype, dev, g, d))
            return dict(q=q, k=k, fq=freqs, fk=freqs, qw=weight(x_dtype, dev, g, d),
                        kw=weight(x_dtype, dev, g, d))

        def call(ck, a):
            if a["fq"] is a["fk"]:
                return ck.rms_rope(a["q"], a["k"], a["fq"], a["qw"], a["kw"], 1e-5)
            return (ck.rms_rope1(a["q"], a["fq"], a["qw"], 1e-5), ck.rms_rope1(a["k"], a["fk"], a["kw"], 1e-5))

        return dict(make=make, call=call, split=False, rot=0, eps=1e-5, tok_dim=2, inplace=False)

    def h3(s, heads=56, rot=96, f_dtype=BF16, x_dtype=BF16, inplace=True, d=128):
        def make(dev, g):
            inner = heads * d
            qkv = torch.randn(1, s, 3 * inner, device=dev, generator=g).mul_(3).to(x_dtype)
            q = qkv[..., :inner].view(1, s, heads, d)
            k = qkv[..., inner:2 * inner].view(1, s, heads, d)
            freqs = rotation_table((1, s, 1), rot // 2, f_dtype, dev, g)
            return dict(q=q, k=k, fq=freqs, fk=freqs, qw=weight(x_dtype, dev, g, d), kw=weight(x_dtype, dev, g, d))

        def call(ck, a):
            fn = ck.rms_rope_split_half_ if inplace else ck.rms_rope_split_half
            return fn(a["q"], a["k"], a["fq"], a["qw"], a["kw"], epsilon=1e-6, rot_dim=rot)

        return dict(make=make, call=call, split=True, rot=rot, eps=1e-6, tok_dim=1, inplace=inplace)

    for t in (10264, 10347):
        out.append((f"krea2 T={t}", krea2(t)))
    out.append(("h3 S=39520", h3(39520)))
    out += [
        ("krea2 T=77", krea2(77)),
        ("krea2 T=1", krea2(1)),
        ("krea2 T=4103 fp16, fp16 freqs", krea2(4103, FP16, FP16)),
        ("h3 S=1031 H=7 rot 128 bf16 freqs", h3(1031, heads=7, rot=128)),
        ("h3 S=1031 H=7 rot 96 fp32 freqs", h3(1031, heads=7, f_dtype=FP32, inplace=False)),
        ("h3 S=33 H=56 rot 8", h3(33, rot=8)),
        ("fallback: head_dim 64", krea2(4103, d=64)),
        ("fallback: per-head freqs", krea2(4103, per_head=True)),
        ("fallback: fp32 x", krea2(1031, FP32, FP32)),
        ("fallback: rot 100", h3(1031, heads=7, rot=100)),
    ]
    return out


def fast_path(x, freqs, split, rot):
    """The launcher's test for the head_dim 128 path, on the tensor a launch sees."""
    b, d1, d2, d = x.shape
    if d != 128 or x.dtype not in (FP16, BF16) or x.stride(-1) != 1:
        return False
    if freqs.shape[1] != 1 and freqs.shape[2] != 1:
        return False
    if split and (rot or d) % 8:
        return False
    if x.data_ptr() % 8:
        return False
    return all(n == 1 or s % 4 == 0 for n, s in zip((b, d1, d2), x.stride()[:3], strict=True))


def ref64(x, freqs, w, eps, split, rot):
    x = x.double()
    d = x.shape[-1]
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.double()
    rot = rot or d
    f = freqs.double()
    sl_a, sl_b = (slice(0, rot // 2), slice(rot // 2, rot)) if split else (slice(0, rot, 2), slice(1, rot, 2))
    a, b = n[..., sl_a], n[..., sl_b]
    out = n.clone()
    out[..., sl_a] = f[..., 0, 0] * a + f[..., 0, 1] * b
    out[..., sl_b] = f[..., 1, 0] * a + f[..., 1, 1] * b
    return out


def run(args):
    import comfy_kitchen
    from comfy_kitchen.backends import hip as ck

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    info = {"torch": torch.__version__, "hip": torch.version.hip,
            "kitchen": getattr(comfy_kitchen, "__version__", "?"),
            "kitchen_path": os.path.dirname(comfy_kitchen.__file__),
            "gpu": f"{props.name} {getattr(props, 'gcnArchName', '')}"}
    ref = torch.load(args.ref) if args.ref else None
    save, rows = {}, []
    for i, (name, c) in enumerate(cases()):
        if args.only and args.only not in name:
            continue
        g = torch.Generator(device=device).manual_seed(2000 + i)
        a = c["make"](device, g)
        tok = c["tok_dim"]
        # fp64 reference on the first 64 tokens of the untouched inputs
        sub = {key: a[key].narrow(tok, 0, min(64, a[key].shape[tok])) for key in ("q", "k")}
        fsub = {key: a[key].narrow(tok, 0, min(64, a[key].shape[tok])) for key in ("fq", "fk")}
        want = (ref64(sub["q"], fsub["fq"], a["qw"], c["eps"], c["split"], c["rot"]),
                ref64(sub["k"], fsub["fk"], a["kw"], c["eps"], c["split"], c["rot"]))
        fast = [fast_path(a[x], a[f], c["split"], c["rot"]) for x, f in (("q", "fq"), ("k", "fk"))]
        q_out, k_out = c["call"](ck, a)
        err = max((o.narrow(tok, 0, w.shape[tok]).double() - w).abs().max().item()
                  for o, w in ((q_out, want[0]), (k_out, want[1])))
        row_digest = digest(torch.cat([q_out.flatten(), k_out.flatten()]))
        sample = torch.cat([q_out.flatten()[::97], k_out.flatten()[::97]]).float().cpu()

        moved = sum(2 * t.numel() * t.element_size() for t in (a["q"], a["k"]))
        moved += a["fq"].numel() * a["fq"].element_size()
        if a["fk"] is not a["fq"]:
            moved += a["fk"].numel() * a["fk"].element_size()
        ms = time_ms(lambda a=a, c=c: c["call"](ck, a), args.warmup, args.iters)
        row = {"case": name, "ms": ms[0], "minmax": list(ms[1:]), "gbs": moved / ms[0] / 1e6,
               "fast": "".join("q" if f else "-" for f in fast[:1]) + "".join("k" if f else "-" for f in fast[1:]),
               "err64": err}
        save[name] = {"sha256": row_digest, "sample": sample}
        if ref is not None and name in ref:
            row["identical"] = ref[name]["sha256"] == row_digest
            row["maxdiff_vs_ref"] = (ref[name]["sample"] - sample).abs().max().item()
        rows.append(row)
        a = q_out = k_out = None
        torch.cuda.empty_cache()
    if args.save_ref:
        torch.save(save, args.save_ref)
    return info, rows


def print_table(info, rows):
    print(json.dumps(info))
    print(f"{'case':<36}{'ms':>9}{'GB/s':>8}{'%peak':>7}{'fast':>6}{'err64':>10}{'identical':>11}{'maxdiff':>9}")
    for r in rows:
        print(f"{r['case']:<36}{r['ms']:>9.3f}{r['gbs']:>8.0f}{100 * r['gbs'] / PEAK_GBS:>7.0f}{r['fast']:>6}"
              f"{r['err64']:>10.3g}{str(r.get('identical', '')):>11}{str(r.get('maxdiff_vs_ref', '')):>9}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--only", help="substring of the case name, e.g. krea2 or h3")
    p.add_argument("--save-ref", help="write output hashes and samples of this kitchen build")
    p.add_argument("--ref", help="compare against a file written by --save-ref")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    with torch.inference_mode():
        info, rows = run(args)
    if args.json:
        print(json.dumps({"info": info, "rows": rows}))
    else:
        print_table(info, rows)


if __name__ == "__main__":
    main()
