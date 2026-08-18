// (1) classic scripts share the global lexical scope: a top-level let/const/class/
//     function declared in BOTH a shared file (common.js|physics.js|solver2d.js) and an
//     app is a load-time SyntaxError. (2) every identifier an app reads must be local,
//     shared by the files THAT page loads, or a standard/browser global -- catches typos
//     across the shared-core boundary. (3) solver2d.js is checked as a page of its own,
//     against common.js + physics.js alone: it may never reach into rmhd2d.html's inline
//     script, which is what makes it loadable by a second page.
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
// the shared classic scripts, and which of them each unit is loaded on top of. A page's
// list is the one its markup carries; solver2d.js is a unit too, and its list is what
// makes it reusable -- common.js and physics.js only, never a page's inline script.
const SHARED = ['common.js', 'physics.js', 'solver2d.js'];
const LOADS = { 'rmhd2d.html': ['common.js', 'physics.js', 'solver2d.js'],
                'rmhd3d.html': ['common.js', 'physics.js'],
                'solver2d.js': ['common.js', 'physics.js'] };
const top = {};
for (const f of SHARED) top[f] = topNames(read(DIR + '/' + f));
let bad = 0;
for (let a = 0; a < SHARED.length; a++) for (let b = a + 1; b < SHARED.length; b++) {
  const hit = [...top[SHARED[a]]].filter(n => top[SHARED[b]].has(n));
  bad += hit.length;
  console.log(SHARED[a] + ': ' + top[SHARED[a]].size + ' top-level names, ' + SHARED[b] + ': ' +
              top[SHARED[b]].size + (hit.length ? '  COLLIDE: ' + hit.join(', ') : '  (no collision)'));
}
for (const page of Object.keys(LOADS)) {
  const src = page.endsWith('.html') ? inline(DIR + '/' + page) : read(DIR + '/' + page);
  const shared = new Set(LOADS[page].flatMap(f => [...top[f]]));
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
