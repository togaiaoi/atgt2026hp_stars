# Stars パズル ハンドオーバー文書

## パズルの概要
「あんたがた2026HP Stars」は、`stars.txt`（約400MB）のSKIコンビネータ計算プログラムを解析するパズル。プログラムを評価すると質問文を出力し、正しい「鍵文字列」を入力すると画像が生成される。その画像から最終回答を読み取るのがゴール。

## 現在の到達地点

### 完了していること
1. **SKI評価器の構築**: Rust製のグラフリダクション評価器を `ski_eval_rs/` に構築済み。arenaベース、遅延評価、fuel制限あり。
2. **I/Oプロトコルの解明**: pair1(pair1(p1,p2), Q) 構造でChurch数タグを使う。出力(p1=1)、入力(p1=2)、停止(p1=0)。
3. **I/Oフローの確認**: 空入力時 → 質問出力(33文字) → 鍵入力要求 → "wrong"出力(5文字) → 停止
4. **文字列デコード成功**: Convention B（pair_fst=value, pair_snd=rest、逆順push）で文字列を取得。
5. **全文字コード対応表の取得**: charcode_query.jsでサーバーに直接問い合わせ、0-24の全コード+負コード3個を解明。
6. **質問文の完全解読**: `,lz-QPwnQu-wnz-kPX-MPz-P-zlWlQD-<` = "for navigation, what is current definition?"
7. **ナビゲーター調査**: `[view (navigator)]` で "current definition = QfnQ&" と回答。
8. **鍵文字列の確定と検証**: **QfnQ& (コード [5,0,17,5,3])** が正解。
   - プログラムが "wrong" の代わりに鍵を5回エコーし、画像データ(p2=2)を出力。

### 残り1ステップ
1. **画像レンダリング** — diamond構造のデコードにOOM問題あり。5引数セレクタで再試行が必要。
2. **最終回答を画像から読み取る**

## 鍵検証の詳細

### I/Oフロー（鍵QfnQ&入力時）
```
Step 1: 出力・文字列 (p1=1, p2=1) — 33文字の質問文
Step 2: 入力・文字列 (p1=2, p2=1) — 鍵 [5,0,17,5,3] を入力
Step 3: 出力・文字列 (p1=1, p2=1) — 鍵を5回繰り返し (25文字)
Step 4: 出力・画像   (p1=1, p2=2) — 画像データ出力 ← ★ここ
Step 5: 停止 (p1=0, p2=0)
```

### 鍵文字列の構築方法（Rust）
```rust
let mut str_node = make_false(&mut arena); // nil
for &code in key_codes.iter().rev() {      // 逆順pushが重要
    let ch_num = make_scott_num(&mut arena, code);
    str_node = make_pair(&mut arena, ch_num, str_node); // pair_fst=value
}
```

### 実行コマンド
```bash
cd ski_eval_rs && PATH="/c/Users/mizuki/.cargo/bin:$PATH" cargo build --release
./ski_eval_rs/target/release/ski-eval.exe very_large_txt/stars_compact.txt \
  --fuel 5000000000 --decode io --key 5,0,17,5,3 --img images/trykey --grid 64
```

## 画像レンダリングの課題

### 画像データの構造
画像データは **diamond（Church-encoded 5-tuple）**:
```
diamond(COND)(QA)(QB)(QC)(QD) = λf. f(COND)(QA)(QB)(QC)(QD)
```
- COND: ブール値（葉の場合のピクセル色）
- QA=NW, QB=NE, QC=SW, QD=SE: 4象限の子ノード

### pair1セレクタでは失敗する理由
pair1のK/KIセレクタを適用すると、5引数のうち余分な引数が互いに適用されて巨大な計算が発生しOOMになる。

### 正しい5引数セレクタ（数学的に検証済み）
```
sel_0 = S(KK)(S(KK)(S(KK)(S(KK)(I))))  → COND を抽出
sel_1 = K(S(KK)(S(KK)(S(KK)(I))))      → QA (NW) を抽出
sel_2 = K(K(S(KK)(S(KK)(I))))          → QB (NE) を抽出
sel_3 = K(K(K(S(KK)(I))))              → QC (SW) を抽出
sel_4 = K(K(K(K(I))))                  → QD (SE) を抽出
```
使い方: `data(sel_i)` で i番目のフィールドを取得。

### 追加で必要なこと
- Rust評価器にこの5引数セレクタを使ったレンダラを実装する
- アリーナサイズ上限を追加する（現在46GBでOOMクラッシュ）
- 小さい解像度（4x4, 8x8）から試して正しくレンダリングできるか確認

## 全文字コード対応表
```
コード →  表示文字
  0   →  f        10  →  F        20  →  W
  1   →  w        11  →  *        21  →  M
  2   →  <        12  →  y        22  →  P
  3   →  &        13  →  ,        14  →  C
  5   →  Q        15  →  {        24  →  X
  6   →  l        16  →  D
  7   →  u        17  →  n        負のコード:
  8   →  -        18  →  0        -1  →  [
  9   →  5        19  →  z        -2  →  o
  4   →  9        23  →  k        -3  →  j
```

## 主要ファイル
| ファイル | 説明 |
|---------|------|
| `ski_eval_rs/src/main.rs` | Rust SKI評価器（~3500行） |
| `very_large_txt/stars_compact.txt` | コンパクト形式のプログラム |
| `CLAUDE.md` | 詳細な技術情報・仕様・確定事項 |
| `reference/couldbewrong/reference.md` | プレイヤー解析（言語仕様・問題解答集） |
| `scripts/navigator_query.js` | サーバーナビゲータークエリスクリプト |
| `scripts/charcode_query.js` | 文字コード問い合わせスクリプト |
| `send/send.js` | サーバー提出スクリプト（他プレイヤー作） |
| `images/` | レンダリング出力（PGM形式） |
