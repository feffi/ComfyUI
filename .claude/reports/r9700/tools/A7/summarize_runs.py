"""Median over several comfy_startup_profile.py runs.

usage: python summarize_runs.py startup_warm_1.json startup_warm_2.json startup_warm_3.json
"""
import json
import statistics
import sys
from collections import defaultdict


def main():
    runs = [json.load(open(p, encoding="utf-8")) for p in sys.argv[1:]]
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    print("runs: {}".format(len(runs)))
    for key in ("interpreter_start_ms", "gui_line_ms_since_script", "process_total_ms", "process_cpu_ms", "defender_cpu_s_delta"):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if vals:
            print("{:28s} median {:9.1f}   runs {}".format(key, med(vals), vals))

    def table(title, rows_of_run, min_ms):
        acc = defaultdict(list)
        for rows in rows_of_run:
            for name, dur in rows:
                acc[name].append(dur)
        print("\n" + title)
        for name, vals in sorted(acc.items(), key=lambda kv: -med(kv[1])):
            if med(vals) >= min_ms:
                print("  {:9.1f}  {}".format(med(vals), name))

    table("phases (median ms):", [[(p[0], p[2]) for p in r["phases"]] for r in runs], 1)
    table("first imports, inclusive (median ms, nested entries overlap):", [[(i[0], i[2]) for i in r["imports"]] for r in runs], 30)
    table("DLL / extension loads (median ms):", [[(d[0] + " " + d[1].split("site-packages")[-1], d[3]) for d in r.get("dlls", [])] for r in runs], 10)
    table("node module loads (median ms):", [[(n[0] + "/" + n[1].replace("\\", "/").rstrip("/").split("/")[-1], n[2]) for n in r["node_loads"]] for r in runs], 20)


if __name__ == "__main__":
    main()
