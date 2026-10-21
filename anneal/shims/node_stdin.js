'use strict';
// Preloaded before every JavaScript and TypeScript program the gate runs (node --require node_stdin.js main.js).
// fs.readFileSync(0) and fs.readFileSync('/dev/stdin') return the whole of standard input, as on a Linux judge:
// Windows has no /dev/stdin, and a single read of fd 0 on a pipe can stop early there. A second read returns
// nothing, as it would after the input is used up. Everything else is unchanged.
const fs = require('fs');

const original = fs.readFileSync;
let consumed = false;

function readAllStdin() {
  if (consumed) return Buffer.alloc(0);
  consumed = true;
  const chunks = [];
  const buf = Buffer.alloc(1 << 16);
  for (;;) {
    let n;
    try {
      n = fs.readSync(0, buf, 0, buf.length, null);
    } catch (e) {
      if (e.code === 'EOF') break;
      if (e.code === 'EAGAIN') continue;
      throw e;
    }
    if (n === 0) break;
    chunks.push(Buffer.from(buf.subarray(0, n)));
  }
  return Buffer.concat(chunks);
}

fs.readFileSync = function readFileSync(path, options) {
  if (path === 0 || path === '/dev/stdin' || path === '/dev/fd/0' || path === '/proc/self/fd/0') {
    const data = readAllStdin();
    const encoding = typeof options === 'string' ? options : options && options.encoding;
    return encoding ? data.toString(encoding) : data;
  }
  return original.apply(this, arguments);
};
require('module').syncBuiltinESMExports();
