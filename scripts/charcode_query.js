// Query the server to find the character code of each server character.
// Strategy: [print [format [output.number [pop (X) [z.1 z.2 z.1] 0] end]]]
// This pops the first char from the 1-char string "(X)" and outputs its numeric code.

async function query(input) {
  const res = await fetch("https://stars-2026-hp.xyz/", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ input }),
  });
  const data = await res.json();
  return data;
}

// Number decoding: server uses 5-based system
// Dlu=0, Xn&=1, PXz=2, n9u=3, zPQ=4, uPX=5
// uPXn&=6, uPXPXz=7, uPXn9u=8, uPXzlQ=9
// wn-=10, wn-Xn&=11, ..., wn-uPXzlQ=19
// PXzn-wn-=20, ..., PXzn-wn-zPQ=24
// Negative: MPQ prefix means -(x+1)
const NUM_MAP = {
  "Dlu": 0, "Xn&": 1, "PXz": 2, "n9u": 3, "zPQ": 4,
  "uPX": 5, "uPXn&": 6, "uPXPXz": 7, "uPXn9u": 8, "uPXzlQ": 9,
  "wn-": 10, "wn-Xn&": 11, "wn-PXz": 12, "wn-n9u": 13, "wn-zPQ": 14,
  "wn-uPX": 15, "wn-uPXn&": 16, "wn-uPXPXz": 17, "wn-uPXn9u": 18, "wn-uPXzlQ": 19,
  "PXzn-wn-": 20, "PXzn-wn-Xn&": 21, "PXzn-wn-PXz": 22, "PXzn-wn-n9u": 23, "PXzn-wn-zPQ": 24,
  "PXzn-wn-uPX": 25, "PXzn-wn-uPXn&": 26, "PXzn-wn-uPXPXz": 27, "PXzn-wn-uPXn9u": 28, "PXzn-wn-uPXzlQ": 29,
  "n9un-wn-": 30,
  "QPz": 100, "X9Q": 1000,
};

function parseNumber(s) {
  // Try direct lookup
  if (NUM_MAP[s] !== undefined) return NUM_MAP[s];
  // Try negative: MPQ prefix
  if (s.startsWith("MPQ")) {
    const rest = s.slice(3);
    const val = NUM_MAP[rest];
    if (val !== undefined) return -(val + 1);
  }
  return "UNKNOWN:" + s;
}

async function main() {
  // Characters to test - all 28 valid server characters
  const chars = ["&","*",",","-","0","5","9","<","C","D","F","M","P","Q","W","X","[","f","j","k","l","n","o","u","w","y","z","{"];

  // The command template:
  // [print [format [output.number [pop (CHAR) [z.1 z.2 z.1] 0] end]]]
  // = cQnWPX clXlM cQPPuMnXfkPXzP c-PuuPX eCHAR8 czfXn& zfPXz zfXn&( Dlu( PQWnX(((

  const results = {};

  for (const ch of chars) {
    const cmd = `cQnWPX clXlM cQPPuMnXfkPXzP c-PuuPX e${ch}8 czfXn& zfPXz zfXn&( Dlu( PQWnX(((`;

    console.log(`Querying char '${ch}': ${cmd}`);

    try {
      const data = await query(cmd);
      const responseStr = data.output?.[0]?.value || JSON.stringify(data);
      console.log(`  Response: ${responseStr}`);

      // Try to extract number from response
      // Expected format: cMPQP cNUMBER((  = [result [NUMBER]]
      const match = responseStr.match(/cMPQP c(.+?)\(\(/);
      if (match) {
        const numStr = match[1].trim();
        const code = parseNumber(numStr);
        results[ch] = { code, raw: numStr };
        console.log(`  => Code: ${code}`);
      } else {
        results[ch] = { code: "PARSE_ERROR", raw: responseStr };
        console.log(`  => Could not parse`);
      }
    } catch (e) {
      console.log(`  Error: ${e.message}`);
      results[ch] = { code: "ERROR", raw: e.message };
    }

    // Small delay to not overload server
    await new Promise(r => setTimeout(r, 300));
  }

  console.log("\n\n=== CHARACTER CODE TABLE ===");
  console.log("Server Char | Code | Raw");
  console.log("-".repeat(50));

  // Sort by code
  const sorted = Object.entries(results).sort((a, b) => {
    const ca = typeof a[1].code === 'number' ? a[1].code : 999;
    const cb = typeof b[1].code === 'number' ? b[1].code : 999;
    return ca - cb;
  });

  for (const [ch, info] of sorted) {
    console.log(`    ${ch}       |  ${String(info.code).padStart(3)}  | ${info.raw}`);
  }

  // Also create reverse mapping (code -> char)
  console.log("\n=== REVERSE MAPPING (Code -> Char) ===");
  const reverseMap = {};
  for (const [ch, info] of Object.entries(results)) {
    if (typeof info.code === 'number') {
      reverseMap[info.code] = ch;
    }
  }

  for (let i = 0; i <= 30; i++) {
    if (reverseMap[i]) {
      console.log(`  ${String(i).padStart(2)} -> '${reverseMap[i]}'`);
    }
  }

  // Decode the question text
  const questionCodes = [2, 8, 16, 5, 6, 20, 6, 19, 8, 22, 8, 19, 22, 21, 8, 24, 22, 23, 8, 19, 17, 1, 8, 7, 5, 17, 1, 22, 5, 8, 19, 6, 13];
  console.log("\n=== DECODED QUESTION TEXT ===");
  let decoded = "";
  for (const code of questionCodes) {
    if (reverseMap[code]) {
      decoded += reverseMap[code];
    } else {
      decoded += `[${code}]`;
    }
  }
  console.log(decoded);

  // Also decode error message
  const errorCodes = [19, 20, 22, 9, 21];
  let errorDecoded = "";
  for (const code of errorCodes) {
    if (reverseMap[code]) {
      errorDecoded += reverseMap[code];
    } else {
      errorDecoded += `[${code}]`;
    }
  }
  console.log(`Error message decoded: ${errorDecoded} (should be M5PWz)`);
}

main().catch(e => console.error(e));
