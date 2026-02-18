# あんたがた2026HP Stars パズル解析プロジェクト

## このファイルについて
重要な情報を随時更新する。コンテキストが失われた場合に備え、発見事項・進捗・判断根拠をここに記録する。

## プロジェクト概要
- `stars.txt` (~400MB) はSKIコンビネータ計算のプログラム
- 正しく評価すると画像を生成し、その画像から最終回答を読み取る
- コンパクト形式: `k`=S, `X`=K, `D`=I, `-`=application (後置記法)

## 信頼度レベル
- `reference/hint_old.md`, `reference/hint.md` → **GMヒント（完全に信頼可能）**
- `reference/reference.md` → **プレイヤー解析（間違っている可能性あり）**
- ただし、reference.md内の数値3のコンパクトエンコーディングはサーバーに受理済みなので信頼できる

## SKIコンビネータ規則
- S f g x → f x (g x)
- K x y → x
- I x → x
- edge_mark (l): l A B C → A C (B (K C)) — stars.txtの原始プリミティブ。S/K/IはすべてlのツリーとしてDEFINED

## プログラム構造
- `result = S(SI(KA))(Y)` — ここでAはLEFT部、Yはデータ部
- LEFT = S-spineの5引数: `LEFT(Y) = arg0(Y)(arg1(Y))(arg2(Y))(arg3(Y))(arg4(Y))`

## エンコーディング仕様

### 2種類のpair
1. **pair1（1引数 Scott pair / タプル）**: `S(SI(KA))(KB)`
   - pair1(handler) = handler(A)(B)
   - pair1_fst: pair1(K) = A
   - pair1_snd: pair1(KI) = B
   - **用途**: I/O命令のタプル構造

2. **pair2（2引数 Scott pair / リスト用）**: `S(KK)(S(SI(KA))(KB))`
   - pair2(f)(g) = f(A)(B) — fがpairハンドラ、gがnilハンドラ
   - pair_fst: pair2(K)(dummy) = A
   - pair_snd: pair2(KI)(dummy) = B
   - **用途**: リスト（文字列、数値ビット列）のcons cell

### 真偽値
- true = S(KK)I → true(x)(y) = x
- false = KI → false(x)(y) = y
- nil = false = KI

### 数値 (2の補数、ビット列)
- ヒント: 「整数は、まずその整数を2の補数表現でビット列に変換し、各ビットを真偽値にすることで真偽値列にしている」
- pair(bit, rest_bits) のチェーン。pair_fst=bit、pair_snd=rest。
  - 0 = pair(false, nil)
  - 1 = pair(true, pair(false, nil))
  - 6 = pair(false, pair(true, pair(true, pair(false, nil))))
- 終端: pair(false, nil) — 最後のfalse bitとnilの組
- **注意**: 数値のビット列は pair_fst=bit, pair_snd=rest の順序。文字列リストの cons(prev, val) とは逆順。
- decode_scott_num: pair_fst をbitとして読む → 動作確認済み（値6のデコード成功）
- decode_integer: 同じ convention（pair_fst=bit, pair_snd=rest）で2の補数として解釈

### 文字列リスト
- ヒント: 「文字列は文字コードの列として表現されている」
- cons(prev_list, char_code) = pair2(prev_list, char_code)
  - pair_fst = prev_list（残りのリスト）
  - pair_snd = char_code（文字コード = 整数）
- nil (KI=false) でリスト終端
- **重要**: 文字列リストはpair_snd=value、数値ビット列はpair_fst=bit。convention が異なる。

### Church数（I/Oタグ専用）
- Church 0 = KI
- Church 1 = S(S(KS)(S(KK)I))(KI)
- I/Oタグ (p1, p2) にのみ使用。整数とは別のエンコーディング。

## I/O プロトコル（検証済み）

### I/O命令の構造
```
IO命令 = pair1(pair1(p1, p2), Q)
```
- p1, p2: Church数（I/Oタグ）
- Q: データ/継続

### I/Oタグの意味
| p1 | 意味 | p2の意味 | Qの内容 |
|----|------|----------|---------|
| 0  | 停止 | 0        | 0 (全てゼロ) |
| 1  | 出力 | 0=整数, 1=文字列, 2=画像 | pair1(data, continuation) |
| 2  | 入力 | 0=整数, 1=文字列 | λx.continuation(x) |

### 発見済みのI/Oフロー
空文字列入力時の動作:
1. **Step 1**: 出力・文字列 (p1=1, p2=1) — 33要素のリスト（質問文）
2. **Step 2**: 入力・文字列 (p1=2, p2=1) — 鍵文字列の入力要求
3. **Step 3**: 出力・文字列 (p1=1, p2=1) — 5要素のリスト（「間違い」メッセージ）
4. **Step 4**: 停止 (p1=0, p2=0)

正しい鍵を入力すれば:
- 鍵データの出力 → 画像データのデコード → 画像出力 → 最終回答

### 画像（無限四分木）
- ヒント: 「画像は無限四分木として表記。(画像 真偽値b e左上 e右上 e左下 e右下)」
- λ式: pair1(bool_b, NW, NE, SW, SE) = 5要素タプル
- bool_b = 現在の色（白/黒）
- NW/NE/SW/SE = 各象限のサブ画像
- インタプリタは指定解像度まで木を掘り、そこの色を出力

## 現在の問題点（Session 7時点）

### 文字列デコード
- 33要素のリストは正しくトラバースできる
- しかし31/33の文字コードが KI (nil/false) → 整数0に相当
- position 1 のみ Scott数として6にデコード成功
- position 0 は複雑な構造で decode_scott_num 失敗
- **仮説**: 文字コードの取り出し方が間違っている可能性。あるいはリスト構造の解釈が誤っている。

### 次に試すべきこと
1. decode_integer（2の補数デコーダ）を文字コードに適用
2. 文字コードの生のSKI構造をもっと詳細に調査
3. リスト要素のpair_fst/pair_snd の入れ替えを試す

## 発見したバグ（修正済み）
1. **pair encoding**: 1引数 → 2引数 に修正
2. **make_scott_num(0)**: KI → pair(false, nil) に修正
3. **数値終端**: bare nil → pair(false, nil) に修正
4. **pair extraction**: 1引数適用 → 2引数適用に修正
5. **decode_church_num**: 部分式のwhnf強制が欠落 → 追加して修正

## ファイル構成
- `very_large_txt/stars_compact.txt` — コンパクト形式 (30,485,221 bytes)
- `ski_eval_rs/src/main.rs` — Rust SKI評価器（メイン、~3100行）
- `extracted/left_x.txt` — LEFT部のコンパクト表現
- `scripts/` — Python解析スクリプト群
- `reference/hint-new.md` — **最新のGMヒント（最重要参照先）**
- `reference/` — その他ヒント・解析文書
- `images/` — レンダリング出力

## ビルドと実行
```bash
cd ski_eval_rs && PATH="/c/Users/mizuki/.cargo/bin:$PATH" cargo build --release
```
実行:
```bash
./ski_eval_rs/target/release/ski-eval.exe very_large_txt/stars_compact.txt --fuel 2000000000 --decode io --img images/io_test
```
主なdecodeモード: render, render2, examine, describe, num, bool, list, stream, structure, walk1, io

## 進捗
- [x] Step 1-4: 圧縮、パース、ラムダ変換、演算子抽出
- [x] Rust SKI評価器構築
- [x] pair/数値エンコーディングのバグ修正
- [x] セルフテスト検証合格
- [x] 1-arg pair抽出 (walk1モード) 検証
- [x] I/Oインタプリタの基本構造実装
- [x] Church数デコード修正・動作確認
- [x] I/Oフロー発見（出力→入力→出力→停止）
- [ ] 文字列デコードの修正 ← **現在作業中**
- [ ] 質問文の解読
- [ ] 鍵文字列の特定
- [ ] 正しい鍵で画像出力
- [ ] 画像レンダリング
- [ ] 最終回答導出
