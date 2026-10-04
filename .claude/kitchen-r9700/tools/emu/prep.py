"""Copy a kitchen HIP source tree and turn kernel<<<g, b, s, st>>>(args) into emu::launch(kernel, g, b, s, st, args)."""
import re, shutil, sys, pathlib
src, dst = map(pathlib.Path, sys.argv[1:3])
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst)
pat = re.compile(r"([A-Za-z_][\w:]*\s*(?:<[^;{}()]*?>)?)\s*<<<\s*(.*?)>>>\s*\(", re.S)
for p in dst.rglob("*"):
    if p.suffix in (".h", ".hip", ".cpp"):
        t = p.read_text()
        n = pat.sub(lambda m: f"emu::launch({m.group(1).strip()}, {m.group(2).strip()}, ", t)
        if n != t: p.write_text(n)
