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
9. **低解像度レンダリング成功**:
   - 4x4: depth 9-25 全て完了（全depthでパターンあり）
   - 8x8: depth 9-15 完了（depth 16でOOM — 中間depthのレンダリング汚染が原因）
   - PNG変換済み: `png/` フォルダに4x4全depth・8x8 d9-15のスケール画像+コンポジットシート

### 完了: デコーダー（left_x）解析
10. **left_x構造解明**: 153K charsの完全評価済みquadtree。7つのY-combinator再帰セクション
    - COND(32%), NW(16%), NE(0.4%), SW(0.05%=座標反射), SE(51%=メイン)
    - 末尾定数: `step(step(step(data, 2048), 256), 32)` で24段ズーム生成
    - 各セクションが `result(0)(0)(0)(前セクション)(1)` でチェーン
11. **演算子テスト**: arg0=等値系, arg1=XOR系, arg3=比較条件
12. **elem[0-2]差分**: >99.97%共有、判別子(K/K(K(I))/I)と副フラグのみ異なる
13. **ブルートフォース限界確認**: 8x8 depth16で1.4B nodes OOM、ネイティブ必須

### 完了: Codex解析結果（再帰セクション=ビットシリアル加算器）
14. **再帰関数の構造**: 6引数関数（x88 x89 x90 x91 x92 x93）
    - x91 = carry/borrow state（1ビット）
    - x92, x93 = 2つの数値引数（pair2/nilとしてdestructure）
    - 再帰呼出: `x88 x97 x102 x181 x98 x103`
15. **演算子の正体（codex確認）**:
    - arg0(a,b) = if a==b { a } else { 0 }（等値パス）
    - arg1(a,b) = a ^ b（XOR）
    - arg3(x1,x2) = LSB条件分岐: `λx1.λx2. x1(λx7. x7 false true x2) x2`
16. **全体構造**: bit-serial adder（wrapping_add相当）

### 128x128レンダリング: OOM問題（未解決）
17. **Rustタイルレンダリング**: 実装完了済み（snapshot/restore方式）
    - `arena.nodes.clone().into_boxed_slice()` でスナップショット
    - タイル毎にスナップショットから復元 → 8x8ピクセル評価
18. **OOM発生**: aggressive GC後のarenaが123M slots（32M live + 91M free）
    - snapshotがALL 123M nodesをclone → ~1.5GB
    - 元arenaと合わせてメモリ割当失敗（"memory allocation of 2147483648 bytes failed"）
19. **必要な修正: Compacting GC**
    - GC後にlive nodesをarena先頭に詰め直し、参照を付け替え、Vecを縮小
    - 123M → 32M nodes（1.5GB → 384MB）でsnapshot可能に

### 残りのステップ
1. **Compacting GC実装** — arenaのliveノード圧縮でsnapshot用メモリ削減
2. **128x128レンダリング** — compacting GC後に全17 depth (9-25) を128x128で
3. **（代替）ネイティブデコーダーのRust実装** — 型付きIRベースの高速デコード
4. **最終回答を画像から読み取る**

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

## 128x128レンダリング方針（snapshot/restore）

### 問題点
- 128x128は四分木depth 7レベル分（16384ピクセル）のナビゲーションが必要
- render_shared（共有遅延評価）では中間評価結果がliveノードとして蓄積 → OOM
- 8x8でもdepth 16でOOM（~1.88B nodes、レンダリング汚染の蓄積が原因）
- checkpoint/restoreはundo logが巨大（数GB〜数十GB）で非実用的
- checkpoint中はpath compressionが無効化され、大幅に遅くなる

### 採用するアプローチ: Vec snapshot/restore
**arenaのnodesベクタを丸ごとcloneし、タイル毎にmemcpy復元する**

#### メリット
- undo log不要（0バイト）
- path compression有効のまま（高速評価）
- 実装が極めてシンプル（clone + 代入）
- 復元コスト: ~384MB memcpy ≈ 50-100ms（タイル評価の秒単位に比べ無視可能）

#### メモリ見積もり（Compacting GC適用後）
- ベースarena（ズーム後GC→compact済み）: ~32M nodes = ~384MB
- snapshot用バッファ: ~384MB
- タイル評価中のピーク: ~650M nodes = ~7.8GB
- **合計ピーク: ~8.5GB**（32GBに余裕で収まる）

#### ⚠ 現状の問題: Compacting GC未実装
- aggressive GC後もarenaは123M slots（32M live + 91M dead）
- snapshot clone = 123M × 12B = ~1.5GB → メモリ割当失敗
- **Compacting GC実装が必須**: liveノードをarena先頭に移動、全参照を更新、Vec縮小

#### アルゴリズム
```
1. I/O Step 4で画像データ(data)を取得
2. dataからルート子ノード(TL,TR,BL,BR)を抽出
3. 7段のズーム（中央象限選択）で対象depthのサブツリーに到達
4. aggressive GC → compacting GC → ~32M live nodes, Vec縮小
5. snapshot = arena.nodes.clone()  // ~384MB
6. for each depth in 9..=25:
     for tile_row in 0..16:
       for tile_col in 0..16:
         a. snapshotから復元: arena.nodes = snapshot.clone()
         b. ズームルートから4レベルナビゲーション → タイルのサブルートへ
         c. サブルートから3レベルのピクセル評価 → 8x8ピクセル
         d. 結果を128x128バッファに書き込み
     画像を保存
```

#### 補足
- 各タイルは独立（前のタイルの評価結果を引き継がない）
- snapshotからの復元でpath compressionの汚染もリセットされる
- 256タイル × 50ms復元 = ~13秒のオーバーヘッド（許容範囲）
- 各depthでズームの再計算が必要だが、ズーム自体は軽量（~90Mノード増）

### 代替案（不採用）
| アプローチ | 不採用理由 |
|-----------|-----------|
| checkpoint/restore | undo log ~5-50GB、path compression無効で遅い |
| OS fork (COW) | Windows非対応 |
| 深度別独立プロセス | 17回のプロセス起動+パース、タイル分割との組合せが複雑 |
| Arena forking | 32M nodes × 4^7 = メモリ不足 |
| 構造パターンマッチ | 実装難度高、エンコーディング解析が必要 |

## 画像データの構造

### diamond（Church-encoded 5-tuple）
```
diamond(COND)(QA)(QB)(QC)(QD) = λf. f(COND)(QA)(QB)(QC)(QD)
```
- COND: ブール値（葉の場合のピクセル色）
- QA=NW, QB=NE, QC=SW, QD=SE: 4象限の子ノード

### 5引数セレクタ（数学的に検証済み・セルフテスト合格）
```
sel_0 = S(KK)(S(KK)(S(KK)(S(KK)(I))))  → COND を抽出
sel_1 = K(S(KK)(S(KK)(S(KK)(I))))      → QA (NW) を抽出
sel_2 = K(K(S(KK)(S(KK)(I))))          → QB (NE) を抽出
sel_3 = K(K(K(S(KK)(I))))              → QC (SW) を抽出
sel_4 = K(K(K(K(I))))                  → QD (SE) を抽出
```
使い方: `data(sel_i)` で i番目のフィールドを取得。

### レンダリング構成
- GMヒント: depth 1-8は2^(depth-1)解像度、depth 9-25は中央1/2を128x128で
- 中央ズーム: 各ズームステップでNWのSE、NEのSW、SWのNE、SEのNWを選択
  - zoom_tl = data(sel_1)(sel_4) = NW.SE
  - zoom_tr = data(sel_2)(sel_3) = NE.SW
  - zoom_bl = data(sel_3)(sel_2) = SW.NE
  - zoom_br = data(sel_4)(sel_1) = SE.NW

### 既存レンダリング結果

#### 4x4（depth 9-25全て完了）
| Depth | Probe | B/W | パターン |
|-------|-------|-----|---------|
| 9 | FFFF | 13/3 | .#.. .#.. ..#. .... |
| 10 | TFFT | 5/11 | #... #### ###. .### |
| 11 | TTTT | 0/16 | all white |
| 12 | TTTT | 1/15 | #### #### #.## #### |
| 13 | TTFT | 6/10 | .### ..## ..## #.## |
| 14 | FTFT | 9/7 | .### ..## ...# ...# |
| 15 | FTFF | 12/4 | ...# .... ..#. #..# |
| 16 | FFFT | 12/4 | ...# .... .### .... |
| 17 | FFTT | 10/6 | .... ..#. ..## .### |
| 18 | FTFT | 8/8 | ...# ...# .### .### |
| 19 | FFTT | 11/5 | ...# .... #... #.## |
| 20 | FFFF | 15/1 | .... .... .#.. .... |
| 21 | FFTT | 10/6 | .... .... ###. .### |
| 22 | FFTT | 10/6 | .... .... .##. #### |
| 23 | FFTT | 6/10 | ...# ..## .### #### |
| 24 | FTTT | 4/12 | ..## ..## #### #### |
| 25 | FTTT | 4/12 | ..## ..## #### #### |

#### 8x8（depth 9-15完了）
- PGM画像: `images/zoom8x8_depth{9-15}.pgm`
- PNG画像: `png/zoom8x8_depth{9-15}_8x8.png`（128x128にスケール済み）
- コンポジットシート: `png/composite_8x8_depths9-15.png`

## Codex解析: 再帰セクションの詳細

### 再帰関数の6引数
```
x88(x89, x90, x91, x92, x93)
  x89, x90: sign extension bits
  x91: carry/borrow state (1-bit)
  x92: 第1数値引数 (pair2/nil destructure)
  x93: 第2数値引数 (pair2/nil destructure)
```

### 演算の概要
- bit-serial adder: 2つの2の補数ビット列を1ビットずつ処理
- carry伝搬あり
- wrapping_add相当の演算

### Rustでの擬似コード
```rust
fn section_add_i64(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}
```

### top-level構造
- x3（最外関数）がdiamond 5-tupleの各フィールドに適用
- 最終段: `step(step(step(data, 2048), 256), 32)` で24段のズームレベル

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
| `ski_eval_rs/src/main.rs` | Rust SKI評価器（~5800行） |
| `extracted/x_arg4.txt` | デコーダー関数（52K compact SKI + lambda decompile） |
| `extracted/x_arg3.txt` | 比較/条件演算子（57 bytes） |
| `extracted/left_x.txt` | left_x全体（61K compact SKI） |
| `extracted/data_items/elem_*.txt` | item_09の各要素（elem0-2: ~301K, elem3-5: small） |
| `very_large_txt/stars_compact.txt` | コンパクト形式のプログラム |
| `CLAUDE.md` | 詳細な技術情報・仕様・確定事項 |
| `reference/couldbewrong/reference.md` | プレイヤー解析（言語仕様・問題解答集） |
| `scripts/navigator_query.js` | サーバーナビゲータークエリスクリプト |
| `scripts/charcode_query.js` | 文字コード問い合わせスクリプト |
| `send/send.js` | サーバー提出スクリプト（他プレイヤー作） |
| `images/` | レンダリング出力（PGM形式） |
| `png/` | PNG形式の画像（スケール済み+コンポジット） |
