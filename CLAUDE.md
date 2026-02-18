# あんたがた2026HP Stars パズル解析プロジェクト

## このファイルについて
重要な情報を随時更新する。コンテキストが失われた場合に備え、発見事項・進捗・判断根拠をここに記録する。

## プロジェクト概要
- `stars.txt` (~400MB) はSKIコンビネータ計算のプログラム
- 正しく評価すると: 質問文出力 → 鍵文字列入力 → 正誤判定 → (正解なら)画像出力
- 画像から最終回答を読み取るのがゴール
- コンパクト形式: `k`=S, `X`=K, `D`=I, `-`=application (後置記法/スタックマシン)

## 信頼度レベル
- `reference/hint-new.md` → **最新GMヒント（最重要・完全に信頼可能）**
- `reference/hint_old.md`, `reference/hint.md` → **旧GMヒント（信頼可能）**
- `reference/reference.md` → **プレイヤー解析（間違っている可能性あり）**
- 外部情報（他プレイヤーから得た情報）は検証の上で採用

## ファイル構成
```
very_large_txt/stars_compact.txt  — コンパクト形式 (30,485,221 bytes)
ski_eval_rs/src/main.rs           — Rust SKI評価器（メイン、~3200行）
analysis_charcode.md              — 文字コード解析メモ
scripts/                          — Python/JS解析スクリプト群
  ski_eval.py                     — Python版SKI評価器
  step4e_test_operators.py        — 演算子テスト
  test_server.js                  — サーバーテスト用
send/send.js                      — 他プレイヤーのサーバー提出スクリプト
extracted/left_x.txt              — LEFT部のコンパクト表現
reference/hint-new.md             — 最新GMヒント
reference/                        — その他ヒント・解析文書
images/                           — レンダリング出力
```

## ビルドと実行
```bash
# ビルド
cd ski_eval_rs && PATH="/c/Users/mizuki/.cargo/bin:$PATH" cargo build --release

# 実行（decodeモードを指定）
./ski_eval_rs/target/release/ski-eval.exe very_large_txt/stars_compact.txt --fuel 2000000000 --decode <MODE> --img images/output
```

### 主なdecodeモード
| モード | 説明 |
|--------|------|
| io | I/Oインタプリタ（質問出力→入力→応答→停止） |
| keyfind | タイミングサイドチャネル攻撃（鍵文字総当たり） |
| render, render2 | 画像レンダリング |
| examine, describe | 構造調査 |
| num, bool, list | 各種デコード |
| stream, structure, walk1 | 詳細構造解析 |

---

## SKIコンビネータ規則
- S f g x → f x (g x)
- K x y → x
- I x → x
- edge_mark (l): l A B C → A C (B (K C)) — stars.txtの原始プリミティブ

## エンコーディング仕様

### 2種類のpair
1. **pair1（1引数 Scott pair / タプル）**: `S(SI(KA))(KB)`
   - pair1(handler) = handler(A)(B)
   - pair1_fst: pair1(K) = A, pair1_snd: pair1(KI) = B
   - **用途**: I/O命令のタプル構造

2. **pair2（2引数 Scott pair / リスト用）**: `S(KK)(S(SI(KA))(KB))`
   - pair2(f)(g) = f(A)(B) — fがpairハンドラ、gがnilハンドラ
   - pair_fst: pair2(K)(dummy) = A, pair_snd: pair2(KI)(dummy) = B
   - **用途**: リスト（文字列、数値ビット列）のcons cell

### 真偽値
- true = S(KK)I → true(x)(y) = x
- false = KI → false(x)(y) = y
- nil = false = KI

### 数値 (2の補数、ビット列)
- pair(bit, rest_bits) のチェーン。pair_fst=bit、pair_snd=rest
- 0 = pair(false, nil), 1 = pair(true, pair(false, nil))
- 6 = pair(false, pair(true, pair(true, pair(false, nil))))
- 終端: pair(false, nil)

### 文字列リスト（Convention B — 実証済み）
- cons(prev_list, char_code) = pair2(prev_list, char_code)
  - **pair_fst = prev_list（残りのリスト）、pair_snd = char_code（文字コード）**
- nil (KI=false) でリスト終端
- ⚠ ヒントの記述とは逆順（pair_fst=value ではなく pair_fst=rest）だが、実際にこれで動作確認済み

### Church数（I/Oタグ専用）
- Church 0 = KI, Church 1 = S(S(KS)(S(KK)I))(KI)
- I/Oタグ (p1, p2) にのみ使用。整数とは別のエンコーディング。

---

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

### 確認済みI/Oフロー
空文字列入力時:
1. **Step 1**: 出力・文字列 (p1=1, p2=1) — 33文字の質問文
2. **Step 2**: 入力・文字列 (p1=2, p2=1) — 鍵文字列の入力要求
3. **Step 3**: 出力・文字列 (p1=1, p2=1) — 5文字のメッセージ（"wrong"）
4. **Step 4**: 停止 (p1=0, p2=0)

正しい鍵を入力すれば:
- 鍵データの出力 → 画像データのデコード → 画像出力 → 最終回答

---

## サーバー情報

### API
- URL: `https://stars-2026-hp.xyz/`
- Method: POST
- Content-Type: application/json
- Body: `{ "input": "<文字列>" }`
- Response: `{ "output": [{ "type": "string"|"image", "value": "..." }] }`

### サーバーの有効文字セット（28文字）
```
& * , - 0 5 9 < C D F M P Q W X [ f j k l n o u w y z {
```
- これら以外のASCII文字（a-z通常、A-Z通常、数字等）は無視される
- **全ての非空の有効入力に対して同一レスポンスが返る**（ブルートフォース不可）

### サーバー応答の例
```
空入力      → "cwnz eDlu8("
"M5PWz"     → "cQn9zn eM5PWz QPW9-8("
任意の有効文字 → "cQn9zn eM5PWz QPW9-8("  （M5PWzと同じ）
```

### サーバー表示文字とプログラム内部コードの対応
サーバーは独自の文字セット（上記28文字）でSKI式を表示する。
`M5PWz` は `wrong` のエンコード表現。

---

## 文字コード解析

### 確定した文字対応（5文字）
| 内部コード | 表示文字 | 英語 | 根拠 |
|-----------|---------|------|------|
| 9         | W       | n    | wrong の4文字目 |
| 19        | M       | w    | wrong の1文字目 |
| 20        | 5       | r    | wrong の2文字目 |
| 21        | z       | g    | wrong の5文字目 |
| 22        | P       | o    | wrong の3文字目 |

### 質問文（33文字、内部コード列）
```
2, 8, 16, 5, 6, 20, 6, 19, 8, 22, 8, 19, 22, 21, 8, 24, 22, 23, 8, 19, 17, 1, 8, 7, 5, 17, 1, 22, 5, 8, 19, 6, 13
```

### エラーメッセージ（5文字） = "wrong"
```
19(=w), 20(=r), 22(=o), 9(=n), 21(=g)
```

### 既知の文字で部分復号した質問文
8=スペースと仮定した場合:
```
? | ??_r_w | o | wog | ?o? | w?? | ?_r??o_r | w_?
```
（パターン: 1-6-1-3-3-3-6-3 文字の8単語）

### 未確定コード（質問文中に出現するもの）
1, 2, 5, 6, 7, 8, 13, 16, 17, 23, 24

### 未使用コード
3, 4, 10, 11, 12, 14, 15, 18

### 整数0-24のコンパクトSKI表現（外部情報・別の記法）
```
Dlu=0, Xn&=1, PXz=2, n9u=3, zPQ=4, uPX=5
uPXn&=6, uPXPXz=7, uPXn9u=8, uPXzlQ=9
wn-=10, wn-Xn&=11, ..., wn-uPXzlQ=19
PXzn-wn-=20, ..., PXzn-wn-zPQ=24
```
- 5進法ベースの構造（0-4基本、5×n+mで拡張）
- この記法はサーバー表示文字を使った表現

---

## 外部解析データ（他プレイヤー情報）

### グラフルート問1
```
クエリ: czMnX e-nQPz lfXn& QP5959PX QnQ Qn&n-QPznk8(
全文:   cMPQP czMn5 e-nQPz lfXn& QP5959PX QnQ Qn&n-QPznk8( czlWlQD cePX,P5 QPwF9zPX QnQ PX9M kPX QP5959PX QnQ &n-QPznk8 ezPX -nW Qnkn<lW uPXl-FXnuz QP5959PX QnQ Qn&n-QPznk8 cezPX -nW kn<lW 5959PX Qn&n-QPznk QPP9znX lf&n{QPX8
```
- サーバー表示文字で書かれた全質問テキスト
- サーバーAPI応答よりかなり長い → サーバーの返すテキストは省略/集約されている可能性
- 「8(」で終端するパターンが繰り返される
- 「c」で各出力ブロックが始まるパターン

---

## 試行済みアプローチと結果

### タイミングサイドチャネル攻撃（keyfind モード）
- 実装: I/O Step2(入力)のQ(λ)に各文字コード(1-24)を適用し、ステップ数で正しい文字を判別
- **結果: 失敗** — 遅延評価のため、Q(input)が即座にWHNFに達してしまう（~25000ステップ）
- 改良版: Q(input)の結果からタグ・Q2・データ・出力文字列の先頭文字まで強制評価
- 改良版の結果は未検証

### サーバーブルートフォース
- 全28有効文字の単一・二重・三重文字を試行
- **結果: 失敗** — 全ての非空有効入力に対して同一レスポンス

### 画像（無限四分木）
- ヒント: pair1(bool_b, NW, NE, SW, SE) = 5要素タプル
- インタプリタは指定解像度まで木を掘り、そこの色を出力
- 画像レンダリングコードは実装済みだが、正しい鍵がないため未使用

---

## 発見したバグ（修正済み）
1. pair encoding: 1引数 → 2引数 に修正
2. make_scott_num(0): KI → pair(false, nil) に修正
3. 数値終端: bare nil → pair(false, nil) に修正
4. pair extraction: 1引数適用 → 2引数適用に修正
5. decode_church_num: 部分式のwhnf強制が欠落 → 追加して修正

---

## 進捗チェックリスト
- [x] Step 1-4: 圧縮、パース、ラムダ変換、演算子抽出
- [x] Rust SKI評価器構築（グラフリダクション、fuel制限、arenaベース）
- [x] pair/数値エンコーディングのバグ修正
- [x] セルフテスト検証合格
- [x] I/Oインタプリタ実装・Church数デコード
- [x] I/Oフロー発見（出力→入力→出力→停止）
- [x] Convention B発見（pair_fst=rest, pair_snd=value）
- [x] 文字列デコード成功（33文字の質問文 + 5文字のエラーメッセージ）
- [x] 5文字の文字コード対応判明（wrong = M5PWz = コード19,20,22,9,21）
- [x] サーバーAPI調査（有効文字28種、全入力で同一応答）
- [ ] **全文字コード対応の解明**（残り19文字）
- [ ] **質問文の完全解読**
- [ ] **鍵文字列の特定**
- [ ] 正しい鍵で画像出力
- [ ] 画像レンダリング
- [ ] 最終回答導出
