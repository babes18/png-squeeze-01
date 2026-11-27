# png-squeeze-01

[![make.fun bounty](https://img.shields.io/badge/make.fun-png--squeeze--01-blue)](https://make.fun/bounty/png-squeeze-01-run-20261127)

Lossless PNG squeeze for the [make.fun bounty](https://make.fun/bounty/png-squeeze-01-run-20261127):
8 screenshots (1280×720 RGBA) compressed as small as possible, every pixel exact.

## How it works

- **Forms** (`forms.mjs`, adapted from the kit's own `scale.mjs`, MIT): palette
  (by count, by luminance, and by a nearest-neighbour walk through colour
  space), grey, grey+alpha, RGB, RGBA, and a tRNS key for single-transparent-
  value images — each screened with zlib-6 over fixed / min-sum / fewest-pairs
  / brute-force (per-row, zlib-weighed) filters.
- **Deflate** (`deflate.mjs`, our own reduced Zopfli): greedy parse seeds the
  cost model, then forward-DP optimal parsing with iterative cost refinement,
  a 4-pair match cache, per-block Huffman trees (dynamic or fixed, whichever
  is smaller), block-split emission (a 128 KiB split is tried), and a
  stored-block fallback. Pure JS, no dependencies, no clock, no randomness —
  two runs write byte-identical outputs.
- **PNG** (`png.mjs`, the kit checker's own codec, MIT): decode, filter, and
  encode with only IHDR / PLTE / tRNS / IDAT / IEND.
- Every output is self-checked by decoding it with the kit decoder and
  comparing RGBA byte for byte before it is written.

## Run

```sh
MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<output> node main.ts
```

Node 22, no install, 60 s for the set of 8 on 1 CPU / 512 MB.
