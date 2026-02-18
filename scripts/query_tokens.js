// Query the server for unknown tokens in the question text
async function query(input) {
  const res = await fetch("https://stars-2026-hp.xyz/", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ input }),
  });
  const data = await res.json();
  return data.output?.map(o => o.value).join(" | ") || JSON.stringify(data);
}

async function main() {
  // The decoded question text (data chain order, with '-' as separator):
  // ,lz | QPwnQu | wnz | kPX | MPz | P | zlWlQD | <
  // Known: ,lz=for, wnz=what, kPX=is, zlWlQD=definition
  // Unknown: QPwnQu, MPz, P, <

  // Test 1: Use [what TOKEN] to query meanings
  const tokens = ["QPwnQu", "MPz", "P", "<", ",lz", "wnz", "kPX", "zlWlQD"];

  console.log("=== [what TOKEN] queries ===");
  for (const tok of tokens) {
    const cmd = `cwnz ${tok}(`;
    const resp = await query(cmd);
    console.log(`[what ${tok}] => ${resp}`);
    await new Promise(r => setTimeout(r, 300));
  }

  // Test 2: Try printing the whole question string as output
  console.log("\n=== Print question string ===");
  const questionStr = ",lz-QPwnQu-wnz-kPX-MPz-P-zlWlQD-<";
  const printCmd = `cQnWPX clXlM cQPPuMnXfz9XP5 e${questionStr}8 PQWnX(((`;
  const printResp = await query(printCmd);
  console.log(`Print question: ${printResp}`);

  // Test 3: Try different tokenizations of the question
  console.log("\n=== Alternative tokenizations ===");

  // Maybe QPwnQu is two tokens: QPw + nQu, or QP + wnQu
  const altTokens = ["QPw", "nQu", "QP", "wnQu", "QPwn", "Qu"];
  for (const tok of altTokens) {
    const cmd = `cwnz ${tok}(`;
    const resp = await query(cmd);
    console.log(`[what ${tok}] => ${resp}`);
    await new Promise(r => setTimeout(r, 300));
  }

  // Test 4: Try printing known strings to verify encoding
  console.log("\n=== Verification ===");
  // Print "wrong" = M5PWz
  const verifyCmd = `cQnWPX clXlM cQPPuMnXfz9XP5 eM5PWz8 PQWnX(((`;
  const verifyResp = await query(verifyCmd);
  console.log(`Print "M5PWz": ${verifyResp}`);

  // Print "definition" = zlWlQD
  const defCmd = `cQnWPX clXlM cQPPuMnXfz9XP5 ezlWlQD8 PQWnX(((`;
  const defResp = await query(defCmd);
  console.log(`Print "zlWlQD": ${defResp}`);

  // Test 5: Try [encode TOKEN] to see graph encoding
  console.log("\n=== Encode queries ===");
  for (const tok of ["QPwnQu", "MPz"]) {
    const cmd = `czlWPX ${tok}(`;
    const resp = await query(cmd);
    console.log(`[encode ${tok}] => ${resp}`);
    await new Promise(r => setTimeout(r, 300));
  }

  // Test 6: Try the question as a command by wrapping in [view]
  console.log("\n=== View-like queries ===");
  // Maybe the question text IS a chapter title or query?
  // Try: [view (chapter ,lz QPwnQu wnz kPX MPz P zlWlQD <)]
  const viewCmd = `czMnX e-nQPz ,lz QPwnQu wnz kPX MPz P zlWlQD <8(`;
  const viewResp = await query(viewCmd);
  console.log(`View as chapter: ${viewResp}`);

  // Test 7: Try QPwnQu as a chapter identifier
  console.log("\n=== Chapter queries ===");
  for (const chap of ["QPwnQu", ",lz QPwnQu wnz kPX MPz P zlWlQD <"]) {
    const cmd = `czMnX e-nQPz ${chap}8(`;
    const resp = await query(cmd);
    console.log(`[view (chapter ${chap})] => ${resp}`);
    await new Promise(r => setTimeout(r, 300));
  }
}

main().catch(e => console.error(e));
