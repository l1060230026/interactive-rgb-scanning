import { build } from 'esbuild';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('.', import.meta.url));
await mkdir(root + 'dist', {recursive:true});
await build({entryPoints:[root+'viewer.js'],bundle:true,format:'esm',minify:true,
  legalComments:'eof',outfile:root+'dist/viewer.js',target:['es2020']});
await build({entryPoints:[root+'phone-points.js'],bundle:true,format:'esm',minify:true,
  legalComments:'eof',outfile:root+'dist/phone-points.js',target:['es2020']});
for (const file of ['index.html','viewer.css']) await copyFile(root+file, root+'dist/'+file);
const licenses = await Promise.all(['@mkkellogg/gaussian-splats-3d/LICENSE','three/LICENSE'].map(async path => {
  try {return await readFile(root+'node_modules/'+path,'utf8');}
  catch (_) {return path.startsWith('@mkkellogg') ? 'GaussianSplats3D 0.4.7 — MIT License; Copyright Mark Kellogg. https://github.com/mkkellogg/GaussianSplats3D/blob/main/LICENSE' : '';}
}));
await writeFile(root+'dist/THIRD-PARTY.txt',licenses.join('\n\n'));
