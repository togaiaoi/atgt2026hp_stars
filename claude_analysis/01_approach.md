# 解法アプローチ

## 最も有望な方針

GMのヒントから、2つの方針がある。現実的にはハイブリッドが最善。

### 方針B（推奨・まず試す）: SKIインタプリタで実行

stars.txtは約4億文字のSKI式。直接実行は非現実的だが、
**遅延評価**と**コンビネータの最適化**を使えば可能かもしれない。

ただし400MBの式を愚直に簡約するのは計算量的に厳しい可能性が高い。

### 方針A（本命）: プログラムを復元して読解

1. **Step 1: l/- → コンパクト (k/D/X/-)**
   - SKIコンビネータパターンを検出して圧縮
   - K: `lll--lll--l-l-l-l-l--` (21文字) → `X`
   - I: `ll-l-lll--lll--l-l-l-l-l---` (27文字) → `D`
   - S: `llll-ll-ll---llll-ll-------l-l-` (31文字) → `k`

2. **Step 2: コンパクト → SKI木**
   - スタックアルゴリズムでパース
   - `k`→S, `D`→I, `X`→K をプッシュ、`-`でpop2つ→適用

3. **Step 3: SKI → ラムダ式（reverse bracket abstraction）**
   - unbracket(x, I) => x
   - unbracket(x, (K e)) => e
   - unbracket(x, (S e1 e2)) => (unbracket(x,e1) unbracket(x,e2))
   - revert(e): unbracketを試し、成功→λx.revert(e')、失敗→再帰的にrevert

4. **Step 4: 構造分離**
   - トップレベルの `(LEFT RIGHT)` を分離
   - LEFT (~152K文字): 上位適用器 `((S X)(K Y))` — RIGHTに適用して `X RIGHT Y` を生成
   - RIGHT (~30.3M文字): `((B^6 H) DATA_CHAIN)` — メインプログラムH + データ
   - さらにRIGHT内部を分離: H (デコーダープログラム, ~30M文字) と DATA_CHAIN (25アイテム, 307K文字)

5. **Step 5: デコーダー部の復元**
   - 対象: LEFT内のX/Y、およびRIGHT内のH（メインプログラム）
   - ラムダ式→元の言語の演算子を同定
   - GMヒントで明示された復元対象:
     - 真偽値 (true/false)
     - 列/スタック (push/pop/empty_stack)
     - 整数 (Scottエンコーディング、チャーチではない)
     - 再帰 (recursive)
     - 画像出力の5引数記号 (diamond)
   - 加えて算術・比較・条件演算子 (add, subtract, multiply, equals, greater, if, etc.)

6. **Step 6: データ部の解釈**
   - GMヒント: 「データ部全体は画像データで暗号化されている」
   - DATA_CHAINの25アイテムそれぞれの役割を特定（画像本体、鍵、ヘッダ、パラメータ等）
   - デコーダープログラムがDATA_CHAINをどう処理するかを理解した上でデコード

## 実装の優先順位

### Phase 1: 圧縮 (l/- → k/D/X/-)
- 400MBファイルを~30MBに圧縮
- これは既にreference.mdで完了報告あり

### Phase 2: パースと構造分離
- 30MBのコンパクト文字列をパース
- トップレベルのLEFTとRIGHTを分離
- DATA_CHAINの25アイテムを抽出

### Phase 3: reverse bracket abstraction
- デコーダー部（~152K文字）をラムダ式に変換
- ラムダ式を読解可能な形に

### Phase 4: プログラム理解
- 復元されたプログラムが何をしているか理解
- データ部をどう処理するかを把握

### Phase 5: 実行 or 手動デコード
- 理解に基づいてデータを処理
- 最終回答を得る

## 技術的課題

1. **メモリ**: 400MBファイルの処理。ストリーム処理が必要
2. **SKIパターンマッチ**: l/-文字列中でSKIコンビネータを正しく認識する
   - 単純な文字列置換ではなく、木構造としてのマッチが必要
3. **reverse bracket abstraction**: 大きなSKI式に対する効率的な逆変換
4. **演算子同定**: ラムダ式から元の演算子（add, if, etc.）を認識する
