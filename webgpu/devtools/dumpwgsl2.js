// Emit every generated WGSL kernel of an app page, at every resolution preset and at
// the self-test grid, for byte-diffing against a pre-phase baseline.
// Usage: node dumpwgsl2.js <dir> <page> "" <out.txt> ['{"pm":10,...}']
//   (kdiff.py diffs two dumps)
//
// The optional 5th argument is a JSON object merged into every parameter set, for
// dumping a NON-default solver: the 2D per-field dissipation (`{"pm":10}`, Pm = nu/eta),
// the maintained equilibrium flux (`{"eqsrc":true}`) and the rectangular boxes
// (`{"ny":128,"Lx":12.566370614359172,"Ly":6.283185307179586}`) of REFINE_PLAN J / J2 are
// compile-time constants, so they have their own kernel text. Diff two such dumps against
// each other (the labels are the same) to see exactly which kernels a knob reaches; leave
// it off for the phase baseline, which it does not change by a byte.
"use strict";
const fs = require("fs");
const [dir, page, demo, out] = process.argv.slice(2);
const ovr = process.argv[6] ? JSON.parse(process.argv[6]) : null;
const env = require("./stubenv")(dir, page, demo);
const is3d = env.is3d;
// every resolution the page's own selRes offers -- the dump is only a pin on the text the
// app can actually emit, and a kernel that differs only at the longest line (1024 in 2D,
// nz = 128/256 in 3D) is invisible to a dump that stops short of it
const presets = is3d ? [[64, 32], [128, 32], [128, 64], [256, 64], [64, 128], [64, 256]]
                     : [[128], [256], [512], [1024]];
const chunks = [];
function dump(label, P) {
  if (ovr) Object.assign(P, ovr);
  const S = env.run(`function(P){
    const gr = makeGrid(P);
    const g = Object.assign(${is3d ? "{ nx: P.nx, ny: P.ny, nz: P.nz }"
                                   : "{ nx: P.nx, ny: P.ny, Lx: P.Lx, Ly: P.Ly, pm: P.pm, eqsrc: P.eqsrc }"}, gr);
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
