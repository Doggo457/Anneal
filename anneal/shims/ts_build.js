'use strict';
// node ts_build.js <typescript package folder> <in.ts> <out.js>
// TypeScript to CommonJS JavaScript with transpileModule: no type check (types never change what a program does),
// syntax errors fail the build.
const fs = require('fs');
const path = require('path');

const [tsDir, inFile, outFile] = process.argv.slice(2);
const ts = require(path.join(tsDir, 'lib', 'typescript.js'));
const out = ts.transpileModule(fs.readFileSync(inFile, 'utf8'), {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.CommonJS,
    esModuleInterop: true,
  },
  fileName: path.basename(inFile),
  reportDiagnostics: true,
});
const errors = (out.diagnostics || []).filter((d) => d.category === ts.DiagnosticCategory.Error);
if (errors.length) {
  for (const d of errors) {
    const where = d.file && d.start !== undefined ? d.file.getLineAndCharacterOfPosition(d.start) : null;
    const at = where ? `${inFile}(${where.line + 1},${where.character + 1}): ` : '';
    process.stderr.write(`${at}error TS${d.code}: ${ts.flattenDiagnosticMessageText(d.messageText, '\n')}\n`);
  }
  process.exit(1);
}
fs.writeFileSync(outFile, out.outputText);
