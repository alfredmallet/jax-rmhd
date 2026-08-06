// Emit every generated WGSL kernel of an app page, at every resolution preset and at
// the self-test grid, for byte-diffing against a pre-phase baseline.
// Usage: node dumpwgsl2.js <dir> <page> "" <out.txt>   (kdiff.py diffs two dumps)
"use strict";
const fs = require("fs");
const [dir, page, demo, out] = process.argv.slice(2);
const env = require("./stubenv")(dir, page, demo);
const is3d = env.is3d;
const presets = is3d ? [[64, 32], [128, 32], [128, 64], [256, 64]] : [[128], [256], [512]];
const chunks = [];
function dump(label, P) {
  const S = env.run(`function(P){
    const gr = makeGrid(P);
    const g = Object.assign(${is3d ? "{ nx: P.nx, ny: P.ny, nz: P.nz }" : "{ nx: P.nx, ny: P.ny }"}, gr);
    return buildShaders(g);
  }`, P);
  for (const k of Object.keys(S).sort()) chunks.push("########## " + label + " :: " + k + " ##########\n" + S[k] + "\n");
}
for (const pr of presets) {
  const P = is3d
    ? { nx: pr[0], ny: pr[0], nz: pr[1], Lx: 2*Math.PI, Ly: 2*Math.PI, Lz: 2*Math.PI,
        diss: 2.2e-11, hyper: 4, zdiss: 3.1e-4, fshell: [1,3] }
    : { nx: pr[0], ny: pr[0], Lx: 2*Math.PI, Ly: 2*Math.PI, diss: 1e-13, hyper: 4, fshell: [1,3] };
  dump(pr.join("x"), P);
}
const R = env.run("function(){ return REFVEC; }");
const PS = is3d
  ? { nx: R.nx, ny: R.ny, nz: R.nz, Lx: R.Lx, Ly: R.Ly, Lz: R.Lz, diss: R.diss, hyper: R.hyper,
      zdiss: R.z_diss_k, fshell: R.fshell }
  : { nx: R.nx, ny: R.ny, Lx: R.Lx, Ly: R.Ly, diss: R.diss, hyper: R.hyper, fshell: R.fshell };
dump("selftest", PS);
fs.writeFileSync(out, chunks.join(""));
console.log(page + " -> " + out + "  (" + chunks.length + " kernels, " + fs.statSync(out).size + " bytes)");
process.exit(0);
