// parse every generated WGSL kernel with wgsl_reflect (no GPU here; this is the
// closest thing to a WGSL compile check the sandbox can run)
//
// wgsl_reflect is a PARSER, not Tint: it accepts WGSL *reserved words* as identifiers
// (bit us 2026-08-08: `fn selfnormScale(target: ...)` parsed clean here and killed the
// pipeline on every real device -- "target" is reserved). So this script additionally
// scans every declared identifier (let/var/const/fn/params) against the WGSL spec's
// reserved-word list and fails on a hit.
// wgsl_reflect from wherever it is installed (WGSL_REFLECT=<path> overrides, the same
// idiom names.mjs uses for acorn -- checkiso.js points it at devtools/node_modules)
import fs from 'fs';
import path from 'path';
const { WgslReflect } = await import(
  process.env.WGSL_REFLECT ? path.resolve(process.env.WGSL_REFLECT)
                           : '/tmp/chk/node_modules/wgsl_reflect/wgsl_reflect.module.js');

const RESERVED = new Set(`NULL Self abstract active alignas alignof as asm asm_fragment
async attribute auto await become binding_array cast catch class co_await co_return
co_yield coherent column_major common compile compile_fragment concept const_cast
consteval constexpr constinit crate debugger decltype delete demote demote_to_helper do
dynamic_cast enum explicit export extends extern external fallthrough filter final
finally friend from fxgroup get goto groupshared highp impl implements import inline
instanceof interface layout lowp macro macro_rules match mediump meta mod module move
mut mutable namespace new nil noexcept noinline nointerpolation noperspective null
nullptr of operator package packoffset partition pass patch pixelfragment precise
precision premerge priv protected pub public readonly ref regardless register
reinterpret_cast require resource restrict self set shared sizeof smooth snorm static
static_assert static_cast std subroutine super target template this thread_local throw
trait try type typedef typeid typename typeof union unless unorm unsafe unsized use
using varying virtual volatile wgsl where with writeonly yield`.split(/\s+/));

const DECL = /\b(?:let|var|const)\s+([A-Za-z_]\w*)|\bfn\s+([A-Za-z_]\w*)\s*\(([^)]*)\)/g;
const PARAM = /([A-Za-z_]\w*)\s*:/g;
function reservedHits(code) {
  const hits = new Set();
  for (const m of code.matchAll(DECL)) {
    const names = [m[1], m[2]];
    if (m[3]) for (const p of m[3].matchAll(PARAM)) names.push(p[1]);
    for (const nm of names) if (nm && RESERVED.has(nm)) hits.add(nm);
  }
  return [...hits];
}

let n = 0, bad = 0;
for (const f of process.argv.slice(2)) {
  const txt = fs.readFileSync(f, 'utf8');
  const parts = txt.split(/^########## (.*) ##########$/m);
  for (let i = 1; i < parts.length; i += 2) {
    const name = parts[i], code = parts[i + 1];
    n++;
    try { new WgslReflect(code); }
    catch (e) { bad++; console.log('PARSE FAIL ' + name + ': ' + e.message); }
    const hits = reservedHits(code);
    if (hits.length) { bad++; console.log('RESERVED-WORD FAIL ' + name + ': ' + hits.join(', ')); }
  }
}
console.log(n + ' kernels parsed, ' + bad + ' failures');
process.exit(bad ? 1 : 0);
