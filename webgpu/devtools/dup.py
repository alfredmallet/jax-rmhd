# token-normalized clone detector: report any run of >= N normalized non-trivial lines
# that appears more than once across the shared core and the two app scripts.
import re, sys, hashlib
from collections import defaultdict
N = 10
files = sys.argv[1:]
norm_lines = {}
for f in files:
    out = []
    for i, raw in enumerate(open(f, encoding="utf-8"), 1):
        s = raw.strip()
        if not s or s.startswith("//"): continue
        s = re.sub(r"\s+", " ", s)
        if len(s) < 8: continue            # braces, "}," etc.
        out.append((i, s))
    norm_lines[f] = out
index = defaultdict(list)
for f, ls in norm_lines.items():
    for k in range(len(ls) - N + 1):
        h = hashlib.md5("\n".join(x[1] for x in ls[k:k+N]).encode()).hexdigest()
        index[h].append((f, ls[k][0], ls[k+N-1][0]))
clones = {h: v for h, v in index.items() if len(v) > 1}
# collapse overlapping reports
seen = set(); reported = 0
for h, v in sorted(clones.items(), key=lambda kv: -len(kv[1])):
    key = tuple(sorted((f, a // N) for f, a, b in v))
    if key in seen: continue
    seen.add(key); reported += 1
    print("CLONE (%d lines x %d):" % (N, len(v)), "; ".join("%s:%d-%d" % (f.split("/")[-1], a, b) for f, a, b in v))
print("total distinct %d-line clone groups: %d" % (N, reported))
