async function test(input) {
  const res = await fetch("https://stars-2026-hp.xyz/", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ input }),
  });
  const data = await res.json();
  return data.output?.[0]?.value || "NO";
}

async function main() {
  const chars = ["&","*",",","-","0","5","9","<","C","D","F","M","P","Q","W","X","[","f","j","k","l","n","o","u","w","y","z","{"];

  // Test double chars to see variation
  const responses = new Map();
  for (const c of chars) {
    const val = await test(c + c);
    if (!responses.has(val)) responses.set(val, []);
    responses.get(val).push(c + c);
  }
  console.log("Unique responses for double chars:", responses.size);
  for (const [resp, inputs] of responses) {
    console.log("  Response:", JSON.stringify(resp));
    console.log("  Inputs (" + inputs.length + "):", inputs.slice(0, 15).join(", "));
  }

  // Also test some specific multi-char combos
  const combos = ["Dlu", "Xn&", "PXz", "n9u", "zPQ", "uPX", "wn-"];
  for (const combo of combos) {
    const val = await test(combo);
    console.log("  " + combo + " -> " + JSON.stringify(val));
  }
}

main().catch(e => console.error(e));
