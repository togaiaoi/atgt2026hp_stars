// Query the server's navigator system to understand the question
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
  const queries = [
    // Navigator page
    ["View navigator", "czMnX ePXMn-nQ8("],
    // View about QPwnQu
    ["View about QPwnQu", "czMnX eXnkl QPwnQu8("],
    // View about MPz-P
    ["View about MPz-P", "czMnX eXnkl MPz-P8("],
    // Try [what (PXMn-nQ)]
    ["What PXMn-nQ", "cwnz ePXMn-nQ8("],
    // View chapter PXMn-nQ
    ["View chapter PXMn-nQ", "czMnX e-nQPz PXMn-nQ8("],
    // Try answering with various problem IDs
    ["Answer PXMn-nQ with QfDlu (Q.0)", "cznQPX ePXMn-nQ8 QfDlu("],
    ["Answer PXMn-nQ with QfnQ& (Q.3)", "cznQPX ePXMn-nQ8 QfnQ&("],
    // Try view about QnzF0lX (command)
    ["View about command", "czMnX eXnkl QnzF0lX8("],
    // Try view about QfnQ& (Q.3)
    ["View about Q.3", "czMnX eXnkl QfnQ&8("],
    // Print the navigator question text as string
    ["Print navigator", "cQnWPX clXlM cQPPuMnXfz9XP5 ePXMn-nQ8 PQWnX((("],
  ];

  for (const [desc, cmd] of queries) {
    console.log(`\n--- ${desc} ---`);
    console.log(`CMD: ${cmd}`);
    try {
      const resp = await query(cmd);
      console.log(`RESP: ${resp}`);
    } catch(e) {
      console.log(`ERR: ${e.message}`);
    }
    await new Promise(r => setTimeout(r, 300));
  }

  // Also try: the actual answer to Q.3 is 4 (zPQ)
  // If the key is the answer to the current problem...
  console.log("\n\n=== Trying problem answers as navigator answers ===");
  const answers = [
    ["Q.0 ans: 2", "cznQPX ePXMn-nQ8 PXz("],
    ["Q.1 ans: 3", "cznQPX ePXMn-nQ8 n9u("],
    ["Q.2 ans: 9", "cznQPX ePXMn-nQ8 uPXzlQ("],
    ["Q.3 ans: 4", "cznQPX ePXMn-nQ8 zPQ("],
    ["Q.3 ID: QfnQ&", "cznQPX ePXMn-nQ8 QfnQ&("],
    // Try answer with question ID in (question ...) format
    ["Ans fmt: Q.0", "cznQPX ePX{nQ QfDlu8 PXz("],
    ["Ans fmt: Q.3", "cznQPX ePX{nQ Qfn9u8 zPQ("],
  ];

  for (const [desc, cmd] of answers) {
    console.log(`\n--- ${desc} ---`);
    console.log(`CMD: ${cmd}`);
    try {
      const resp = await query(cmd);
      console.log(`RESP: ${resp}`);
    } catch(e) {
      console.log(`ERR: ${e.message}`);
    }
    await new Promise(r => setTimeout(r, 300));
  }
}

main().catch(e => console.error(e));
