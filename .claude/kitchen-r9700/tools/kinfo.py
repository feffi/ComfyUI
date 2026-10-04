"""Per-kernel resources and main-loop instruction mix from a clang -S AMDGPU file."""
import re, subprocess, sys, collections

def demangle(n):
    d = subprocess.run(["llvm-cxxfilt-20", n], capture_output=True, text=True).stdout.strip()
    return d.replace("comfy::hip_backend::", "").replace("unsigned char const*", "u8*")

def classify(ins):
    op = ins.split()[0]
    if op.startswith("v_wmma"): return "wmma"
    if op.startswith("ds_load") or op.startswith("ds_read"): return "ds_load"
    if op.startswith("ds_store") or op.startswith("ds_write"): return "ds_store"
    if op.startswith("ds_"): return "ds_other"
    if op.startswith("global_load") or op.startswith("buffer_load"): return "vmem_load"
    if op.startswith("global_store") or op.startswith("buffer_store"): return "vmem_store"
    if op in ("s_barrier", "s_barrier_signal", "s_barrier_wait"): return "barrier"
    if op.startswith("s_wait") or op.startswith("s_delay") or op == "s_nop": return "wait/nop"
    if op.startswith("s_cbranch") or op == "s_branch": return "branch"
    if op.startswith("v_"): return "valu"
    if op.startswith("s_"): return "salu"
    return "other"

def kernels(path):
    s = open(path).read().split("\n")
    out = {}
    cur = None
    for line in s:
        m = re.match(r"^(_Z\S+):\s", line + " ")
        if m and not line.startswith("\t"):
            cur = m.group(1); out[cur] = []; continue
        if cur and line.startswith("\t.end_amdhsa_kernel"): cur = None; continue
        if cur and line.startswith(".Lfunc_end"):
            cur = None; continue
        if cur: out[cur].append(line.rstrip())
    sets = {k: v for k, v in re.findall(r"\.set (\S+), (?:max\()?(\d+)", "\n".join(s))}
    lds = dict(re.findall(r"\.amdhsa_kernel (\S+)\n\t\t\.amdhsa_group_segment_fixed_size (\d+)", "\n".join(s)))
    return out, sets, lds

def loop_mix(body):
    labels = {}
    insns = []
    for line in body:
        t = line.strip()
        if not t or t.startswith(";") or t.startswith("//"): continue
        if re.match(r"^\.LBB\S+:", t): labels[t.split(":")[0]] = len(insns); continue
        if t.startswith("."): continue
        insns.append(t.split("//")[0].strip())
    best = None
    for i, ins in enumerate(insns):
        m = re.match(r"s_(cbranch_\w+|branch)\s+(\.LBB\S+)", ins)
        if m and m.group(2) in labels and labels[m.group(2)] <= i:
            lo = labels[m.group(2)]
            n = sum(1 for x in insns[lo:i+1] if x.startswith("v_wmma"))
            if n and (best is None or n > best[2]): best = (lo, i, n)
    if not best: return None
    c = collections.Counter(classify(x) for x in insns[best[0]:best[1]+1])
    return c

if __name__ == "__main__":
    path = sys.argv[1]; filt = sys.argv[2] if len(sys.argv) > 2 else ""
    ks, sets, lds = kernels(path)
    for k, body in ks.items():
        d = demangle(k)
        if filt not in d: continue
        v = sets.get(k + ".num_vgpr"); sg = sets.get(k + ".num_sgpr")
        c = loop_mix(body)
        print(f"{d[:160]}\n  vgpr={v} sgpr={sg} lds={lds.get(k)}")
        if c: print("  loop:", dict(sorted(c.items(), key=lambda x: -x[1])))
