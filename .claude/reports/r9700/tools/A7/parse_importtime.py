"""Summarize a `python -X importtime` log (stderr of ComfyUI).

usage: python parse_importtime.py importtime.log [N]

Prints the N largest imports by cumulative time (only modules imported at
nesting level 1 below a top-level package, so the numbers do not double count),
the N largest by self time, and self time summed per top-level package.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^import time:\s+(\d+) \|\s+(\d+) \|( *)(\S.*)$")


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.match(line.rstrip("\n"))
            if m:
                rows.append((int(m.group(1)), int(m.group(2)), len(m.group(3)) // 2, m.group(4).strip()))

    total_cum = sum(r[1] for r in rows if r[2] == 0)
    print("sum of top-level cumulative import time: {:.0f} ms ({} imports)".format(total_cum / 1000, len(rows)))

    print("\nlargest top-level imports (cumulative, ms):")
    for self_us, cum_us, depth, name in sorted([r for r in rows if r[2] == 0], key=lambda r: -r[1])[:n]:
        print("{:9.1f}  {}".format(cum_us / 1000, name))

    print("\nlargest cumulative at any depth (ms, nested entries overlap):")
    for self_us, cum_us, depth, name in sorted(rows, key=lambda r: -r[1])[:n]:
        print("{:9.1f}  {}{}".format(cum_us / 1000, "  " * depth, name))

    print("\nlargest self time (ms):")
    for self_us, cum_us, depth, name in sorted(rows, key=lambda r: -r[0])[:n]:
        print("{:9.1f}  {}".format(self_us / 1000, name))

    per_pkg = defaultdict(int)
    for self_us, cum_us, depth, name in rows:
        per_pkg[name.split(".")[0]] += self_us
    print("\nself time per top-level package (ms):")
    for pkg, us in sorted(per_pkg.items(), key=lambda kv: -kv[1])[:n]:
        print("{:9.1f}  {}".format(us / 1000, pkg))


if __name__ == "__main__":
    main()
