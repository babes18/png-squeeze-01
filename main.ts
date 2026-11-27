// png-squeeze-01: squeeze 8 screenshots into the smallest lossless PNGs.
//
// Pipeline per image: decode the stored input PNG, screen the exact forms the
// kit's scale compressor tries (palette by count / luminance / nearest-neighbour
// walk, grey, grey+alpha, RGB, RGBA, tRNS key) with zlib-6, run the optimal
// parsing DEFLATE (deflate.mjs) on the winner, keep whichever of the optimal
// stream or zlib-9 is smaller, and self-check the written PNG by decoding it
// with the kit's decoder and comparing RGBA byte for byte. On any mismatch it
// falls back to the scale-style stream, and throws rather than emit a wrong
// image. Deterministic: fixed work only, no clock, no randomness, no
// environment reads beyond the input/output directories below.
//
// Usage (same as the kit starter):
//   MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deflateSync } from 'node:zlib';
import { decodePng } from './png.mjs';
import { pngOfForm, screenAll } from './forms.mjs';
import { deflateOptimal } from './deflate.mjs';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

function equalBytes(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

for (const id of readdirSync(I).sort()) {
  const input = readFileSync(join(I, id, 'image.png'));
  const img = decodePng(input);
  if (!img.ok) throw new Error(`${id}: input does not decode: ${img.why}`);
  const candidates = screenAll(img.rgba, img.width, img.height);
  const best = candidates[0];
  for (let k = 1; k < candidates.length; k++) candidates[k].data = null;
  // rowLen is one filtered row's length (filter byte included): the same
  // position in the previous row, a distance the optimal parse checks directly.
  const opt = deflateOptimal(best.data, { rowLen: img.width * best.form.bpp + 1 });
  const z9 = deflateSync(best.data, { level: 9 });
  const stream = opt.length <= z9.length ? opt : new Uint8Array(z9.buffer, z9.byteOffset, z9.length);
  let png = pngOfForm(img.width, img.height, best.form, stream);
  const check = decodePng(png, { width: img.width, height: img.height });
  if (!check.ok || !equalBytes(check.rgba, img.rgba)) {
    // The optimal stream (or the form) failed the self-check: fall back to the
    // scale-style stream on the same form, and throw rather than emit wrong bytes.
    png = pngOfForm(img.width, img.height, best.form, deflateSync(best.data, { level: 6 }));
    const check2 = decodePng(png, { width: img.width, height: img.height });
    if (!check2.ok || !equalBytes(check2.rgba, img.rgba)) {
      throw new Error(`${id}: self-check failed (${check2.ok ? 'pixels differ' : check2.why})`);
    }
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'image.png'), png);
}
