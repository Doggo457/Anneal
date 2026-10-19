const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ deviceScaleFactor: 2.5 });
  await p.goto('file://' + process.argv[2]);
  await p.waitForTimeout(300);
  await (await p.$('#fig')).screenshot({ path: process.argv[3] });
  await b.close();
})();
