import sys
def load(p):
    d={}; k=None; buf=[]
    for line in open(p, encoding="utf-8"):
        if line.startswith("########## "):
            if k: d[k]="".join(buf)
            k=line.strip().strip("#").strip(); buf=[]
        else: buf.append(line)
    if k: d[k]="".join(buf)
    return d
a=load(sys.argv[1]); b=load(sys.argv[2])
names=sorted(set(a)|set(b))
same=0; diff={}
for n in names:
    if a.get(n)==b.get(n): same+=1
    else:
        kern=n.split(" :: ")[1]
        diff.setdefault(kern,[]).append(n.split(" :: ")[0])
print(f"{sys.argv[1]} vs {sys.argv[2]}: {same}/{len(names)} identical")
for k,v in sorted(diff.items()):
    print("  DIFFERS:", k, "at", ",".join(v))
