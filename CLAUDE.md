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
- `result(N)(m)(z)` → ピクセル値 (N=解像度, m=行, z=列)

## Scott エンコーディング（検証済み）
### pair (2引数 Scott pair)
```
pair(A, B) = S(KK)(S(SI(KA))(KB))
compact: kXX--kkD-X<A>---X<B>---
```
- pair(f)(g) = f(A)(B)  — fがpairハンドラ、gがnilハンドラ
- **重要**: 2引数必要！ 1引数エンコーディング S(SI(KA))(KB) は間違い

### nil (false)
```
nil = KI
compact: XD-
```
- nil(f)(g) = g

### true
```
true = S(KK)I
compact: kXX--D-
```
- true(x)(y) = x

### false
```
false = KI
compact: XD-
```
- false(x)(y) = y

### 数値 (LSB-first binary pair chain)
- 0 = pair(false, nil) — **nilそのものではない！**
- 1 = pair(true, pair(false, nil))
- 2 = pair(false, pair(true, pair(false, nil)))
- 3 = pair(true, pair(true, pair(false, nil)))
- 数値3のcompact (サーバー検証済み): `kXX--kkD-XkXX--D----XkXX--kkD-XkXX--D----XkXX--kkD-XXD----XXD----------` (71 chars)

### pair展開（2引数適用が必要）
- fst(p): p(K)(dummy) = A
- snd(p): p(KI)(dummy) = B

## 発見したバグ（Session 3で発見）
1. **pair encoding**: `S(SI(Ka))(Kb)` (1引数) → `S(KK)(S(SI(Ka))(Kb))` (2引数) に修正必要
2. **make_scott_num(0)**: `KI` (nil) → `pair(false, nil)` に修正必要
3. **数値終端**: bare nil → `pair(false, nil)` に修正必要
4. **pair extraction**: 1引数適用 → 2引数適用に修正必要

## ファイル構成
- `very_large_txt/stars_compact.txt` — コンパクト形式 (30,485,221 bytes)
- `ski_eval_rs/src/main.rs` — Rust SKI評価器（メイン）
- `extracted/left_x.txt` — LEFT部のコンパクト表現
- `scripts/` — Python解析スクリプト群
- `reference/` — ヒント・解析文書
- `images/` — レンダリング出力

## ビルド
```
cd ski_eval_rs && cargo build --release
```
実行:
```
./target/release/ski-eval <compact-file> --decode <mode> --fuel N --var N --grid N --img PATH
```
主なdecodeモード: render, render2, examine, describe, num, bool, list, stream, structure

## 進捗
- [x] Step 1-4: 圧縮、パース、ラムダ変換、演算子抽出
- [x] Rust SKI評価器構築
- [ ] pair/数値エンコーディングのバグ修正 ← **現在作業中**
- [ ] セルフテストで検証
- [ ] 正しい画像レンダリング
- [ ] 最終回答導出
