"""HIP graph probe for gfx1201 on Windows: the capture pattern ComfyUI uses for text encoder decode.

Mirrors comfy/model_prefetch.py: one torch.cuda.CUDAGraph per decoder layer, captured on a
side stream with capture_error_mode="thread_local", after warming the BLAS handle and
workspace on that stream. Layer shapes default to Qwen3-VL-4B (Krea 2 text encoder):
hidden 2560, 32 q heads / 8 kv heads, head_dim 128, MLP 9728, 36 layers.

Reports per decode token: eager time, graph-replay time, eager time with ComfyUI's
two offload streams waiting on the compute stream every layer (the pattern behind
ROCm/legacy-rocm-build#6685), the max difference between replay and eager outputs,
and a soak of --soak tokens to catch hangs / TDR resets on the installed driver.

  .venv-rocm-100\\Scripts\\python.exe hipgraph_probe.py --device cuda:1
  .venv-rocm-100\\Scripts\\python.exe hipgraph_probe.py --hidden 4096 --mlp 12288 --layers 36   (Qwen3-VL-8B)
  set GPU_MAX_HW_QUEUES=2 & python hipgraph_probe.py                                           (#6685 check)
"""
import argparse
import time

import torch
import torch.nn.functional as F


def make_layer(a, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(0)
    def w(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(device=device, dtype=dtype)
    q, kv = a.heads * a.head_dim, a.kv_heads * a.head_dim
    return {"qkv": w(q + 2 * kv, a.hidden), "o": w(a.hidden, q), "gate_up": w(2 * a.mlp, a.hidden), "down": w(a.hidden, a.mlp),
            "n1": torch.ones(a.hidden, device=device, dtype=dtype), "n2": torch.ones(a.hidden, device=device, dtype=dtype),
            "k": w(1, a.kv_heads, a.ctx, a.head_dim), "v": w(1, a.kv_heads, a.ctx, a.head_dim)}


def layer_forward(a, p, x):
    h = F.rms_norm(x, (a.hidden,), p["n1"], 1e-6)
    q, k, v = F.linear(h, p["qkv"]).split([a.heads * a.head_dim, a.kv_heads * a.head_dim, a.kv_heads * a.head_dim], dim=-1)
    q = q.view(1, 1, a.heads, a.head_dim).transpose(1, 2)
    keys = torch.cat([p["k"], k.view(1, 1, a.kv_heads, a.head_dim).transpose(1, 2)], dim=2)
    vals = torch.cat([p["v"], v.view(1, 1, a.kv_heads, a.head_dim).transpose(1, 2)], dim=2)
    o = F.scaled_dot_product_attention(q, keys, vals, enable_gqa=True).transpose(1, 2).reshape(1, 1, -1)
    x = x + F.linear(o, p["o"])
    gate, up = F.linear(F.rms_norm(x, (a.hidden,), p["n2"], 1e-6), p["gate_up"]).chunk(2, dim=-1)
    return x + F.linear(F.silu(gate) * up, p["down"])


def timed(fn, n, device):
    torch.cuda.synchronize(device)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--mlp", type=int, default=9728)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--ctx", type=int, default=512, help="KV length already in the cache")
    ap.add_argument("--tokens", type=int, default=50, help="timed decode tokens per mode")
    ap.add_argument("--soak", type=int, default=5000, help="graph-replay tokens for the stability soak (0 = skip)")
    a = ap.parse_args()

    device = torch.device(a.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print(f"torch {torch.__version__} hip {torch.version.hip} device {props.name} {getattr(props, 'gcnArchName', '')}")
    dtype = torch.bfloat16
    layers = [make_layer(a, device, dtype) for _ in range(a.layers)]
    buf = torch.randn(1, 1, a.hidden, device=device, dtype=dtype)
    x0 = buf.clone()

    def eager_token():
        x = buf
        for p in layers:
            x = layer_forward(a, p, x)
        return x

    offload = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    def eager_token_with_offload_waits():
        x = buf
        for i, p in enumerate(layers):
            offload[i % 2].wait_stream(torch.cuda.current_stream(device))  # as comfy.model_management.get_offload_stream
            x = layer_forward(a, p, x)
        return x

    ref = eager_token()
    t_eager = timed(eager_token, a.tokens, device)
    t_waits = timed(eager_token_with_offload_waits, a.tokens, device)

    # capture: side stream, BLAS handle/workspace warmed on it first (comfy/model_prefetch.py:189-199)
    capture = torch.cuda.Stream(device=device)
    with torch.cuda.stream(capture):
        torch.cuda.current_blas_handle()
        one = torch.empty((2, 2), device=device)
        torch.addmm(one[0], one, one)
        for _ in range(2):  # warm-up on the capture stream, as GRAPH_WARMED_MODULES does
            eager_token()
    torch.cuda.current_stream(device).wait_stream(capture)
    graphs, io = [], []
    t_cap = time.perf_counter()
    for p in layers:
        xin = torch.empty_like(buf)
        graph = torch.cuda.CUDAGraph()
        capture.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.graph(graph, stream=capture, capture_error_mode="thread_local"):
            xout = layer_forward(a, p, xin)
        graphs.append(graph)
        io.append((xin, xout))
    torch.cuda.synchronize(device)
    t_cap = time.perf_counter() - t_cap

    def graph_token():
        io[0][0].copy_(buf)
        for i, graph in enumerate(graphs):
            graph.replay()
            if i + 1 < len(graphs):
                io[i + 1][0].copy_(io[i][1])
        return io[-1][1]

    out = graph_token().clone()
    t_graph = timed(graph_token, a.tokens, device)
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"layers {a.layers} hidden {a.hidden} ctx {a.ctx}: capture {t_cap:.2f} s for {len(graphs)} graphs")
    print(f"per token  eager {t_eager * 1e3:8.3f} ms   eager+offload waits {t_waits * 1e3:8.3f} ms   graph replay {t_graph * 1e3:8.3f} ms   "
          f"speedup {t_eager / t_graph:.2f}x   max|replay-eager| {diff:.3e}")
    assert torch.equal(buf, x0)

    if a.soak:
        t = time.perf_counter()
        for i in range(a.soak):
            graph_token()
            if i % 500 == 499:
                torch.cuda.synchronize(device)
                print(f"  soak {i + 1}/{a.soak} ok, {(time.perf_counter() - t) / (i + 1) * 1e3:.3f} ms/token", flush=True)
        torch.cuda.synchronize(device)
        last = graph_token().float()
        print(f"soak done: {a.soak} tokens, finite={bool(torch.isfinite(last).all())}, max|replay-eager| {(last - ref.float()).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
