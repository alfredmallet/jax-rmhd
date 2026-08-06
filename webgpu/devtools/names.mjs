// (1) two classic scripts share the global lexical scope: a top-level let/const/class/
//     function declared in BOTH common.js|physics.js and an app is a load-time
//     SyntaxError. (2) every identifier an app reads must be local, shared, or a
//     standard/browser global -- catches typos across the shared-core boundary.
import fs from 'fs';
import { createRequire } from 'module';
// acorn from wherever it is installed (npm i acorn; ACORN=<path> overrides)
const { parse } = await import(process.env.ACORN || 'acorn');
const DIR = process.argv[2] || new URL('..', import.meta.url).pathname;
const BUILTIN = new Set(['Math','JSON','Object','Array','String','Number','Boolean','Map','Set','Error',
  'Promise','Float32Array','Float64Array','Uint32Array','Int32Array','Uint8ClampedArray','isFinite','isNaN',
  'parseInt','parseFloat','console','document','window','navigator','performance','requestAnimationFrame',
  'setTimeout','clearTimeout','setInterval','Path2D','GPUBufferUsage','GPUTextureUsage','GPUMapMode',
  'globalThis','undefined','NaN','Infinity','arguments','structuredClone','Symbol','Date','RegExp',
  'TextEncoder','TextDecoder','ArrayBuffer','DataView','Uint8Array','fetch','alert','location','URLSearchParams']);
function topNames(src) {
  const ast = parse(src, { ecmaVersion: 2022 });
  const s = new Set();
  for (const n of ast.body) {
    if (n.type === 'FunctionDeclaration' || n.type === 'ClassDeclaration') s.add(n.id.name);
    else if (n.type === 'VariableDeclaration') for (const d of n.declarations) {
      if (d.id.type === 'Identifier') s.add(d.id.name);
      else if (d.id.type === 'ArrayPattern') for (const e of d.id.elements) if (e) s.add(e.name);
    }
  }
  return s;
}
const read = f => fs.readFileSync(f, 'utf8');
const inline = f => { const t = read(f); return t.slice(t.indexOf('<script>') + 8, t.lastIndexOf('</script>')); };
const common = topNames(read(DIR + '/common.js'));
const phys = topNames(read(DIR + '/physics.js'));
const shared = new Set([...common, ...phys]);
const both = [...common].filter(n => phys.has(n));
let bad = both.length;
console.log('common.js: ' + common.size + ' top-level names, physics.js: ' + phys.size +
            (both.length ? '  COLLIDE: ' + both.join(', ') : '  (no collision)'));
for (const page of ['rmhd2d.html', 'rmhd3d.html']) {
  const src = inline(DIR + '/' + page);
  const local = topNames(src);
  const dup = [...local].filter(n => shared.has(n));
  if (dup.length) { console.log('  ' + page + ' REDECLARES: ' + dup.join(', ')); bad += dup.length; }
  const ast = parse(src, { ecmaVersion: 2022 });
  const seen = new Set();
  (function walk(n, parent, key) {
    if (!n || typeof n.type !== 'string') return;
    if (n.type === 'Identifier') {
      const isProp = parent && ((parent.type === 'MemberExpression' && key === 'property' && !parent.computed) ||
        (parent.type === 'Property' && key === 'key' && !parent.computed) ||
        (parent.type === 'MethodDefinition' && key === 'key') ||
        (parent.type === 'PropertyDefinition' && key === 'key'));
      if (!isProp) seen.add(n.name);
      return;
    }
    for (const k of Object.keys(n)) {
      const v = n[k];
      if (Array.isArray(v)) v.forEach(c => walk(c, n, k));
      else if (v && typeof v === 'object' && typeof v.type === 'string') walk(v, n, k);
    }
  })(ast, null, null);
  // every identifier bound ANYWHERE in the file (declarators incl. patterns, params,
  // function/class names, catch clauses) -- exact, from the AST
  const boundAnywhere = new Set();
  (function bind(n) {
    if (!n || typeof n.type !== 'string') return;
    const pat = p => {
      if (!p) return;
      if (p.type === 'Identifier') boundAnywhere.add(p.name);
      else if (p.type === 'ObjectPattern') p.properties.forEach(q => pat(q.value || q.argument));
      else if (p.type === 'ArrayPattern') p.elements.forEach(pat);
      else if (p.type === 'AssignmentPattern') pat(p.left);
      else if (p.type === 'RestElement') pat(p.argument);
    };
    if (n.type === 'VariableDeclarator') pat(n.id);
    if (n.type === 'FunctionDeclaration' || n.type === 'FunctionExpression' ||
        n.type === 'ArrowFunctionExpression') { if (n.id) boundAnywhere.add(n.id.name); n.params.forEach(pat); }
    if (n.type === 'ClassDeclaration' || n.type === 'ClassExpression') { if (n.id) boundAnywhere.add(n.id.name); }
    if (n.type === 'CatchClause') pat(n.param);
    for (const k of Object.keys(n)) {
      const v = n[k];
      if (Array.isArray(v)) v.forEach(bind);
      else if (v && typeof v === 'object' && typeof v.type === 'string') bind(v);
    }
  })(ast);
  const unknown = [...seen].filter(n => !local.has(n) && !shared.has(n) && !BUILTIN.has(n) && !boundAnywhere.has(n));
  console.log('  ' + page + ': ' + local.size + ' top-level names, ' +
              (unknown.length ? 'UNRESOLVED: ' + unknown.join(', ') : 'every free identifier resolves'));
  bad += unknown.length;
}
process.exit(bad ? 1 : 0);
