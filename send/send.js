// send.js (stdin / --file 対応)
const fs = require("node:fs");

function readAllStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => (data += chunk));
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

async function main() {
  let cmd = null;

  if (process.argv[2] === "--file") {
    const p = process.argv[3];
    if (!p) {
      console.error("Usage: node send.js --file cmd.txt");
      process.exit(1);
    }
    cmd = fs.readFileSync(p, "utf8");
  } else if (process.argv[2]) {
    cmd = process.argv[2];
  } else {
    cmd = await readAllStdin();
  }

  cmd = cmd.trimEnd(); // 末尾改行だけ消す（中の改行はそのまま）

  const url = "stars-2026-hp.xyz/";
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ input: cmd }),
  });

  if (!res.ok) {
    console.error("HTTP", res.status, await res.text());
    process.exit(1);
  }

  const data = await res.json();
  console.log(JSON.stringify(data, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});