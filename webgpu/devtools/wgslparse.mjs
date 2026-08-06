// parse every generated WGSL kernel with wgsl_reflect (no GPU here; this is the
// closest thing to a WGSL compile check the sandbox can run)
import { WgslReflect } from '/tmp/chk/node_modules/wgsl_reflect/wgsl_reflect.module.js';
import fs from 'fs';
let n = 0, bad = 0;
for (const f of process.argv.slice(2)) {
  const txt = fs.readFileSync(f, 'utf8');
  const parts = txt.split(/^########## (.*) ##########$/m);
  for (let i = 1; i < parts.length; i += 2) {
    const name = parts[i], code = parts[i + 1];
    n++;
    try { new WgslReflect(code); }
    catch (e) { bad++; console.log('PARSE FAIL ' + name + ': ' + e.message); }
  }
}
console.log(n + ' kernels parsed, ' + bad + ' failures');
process.exit(bad ? 1 : 0);
