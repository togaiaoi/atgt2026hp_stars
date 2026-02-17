# エンコーディング早見表

## SKIコンビネータ基礎

| コンビネータ | 規則 | コンパクト | 完全graph.symbol (l/-) |
|-------------|------|-----------|----------------------|
| I (identity) | I x = x | `D` | `ll-l-lll--lll--l-l-l-l-l---` (27文字) |
| K (constant) | K x y = x | `X` | `lll--lll--l-l-l-l-l--` (21文字) |
| S (substitution) | S f g x = (f x)(g x) | `k` | `llll-ll-ll---llll-ll-------l-l-` (31文字) |
| edge_mark | l A B C = (AC)(B(KC)) | `l` | `l` (1文字) |

## stars.txt符号化

### postorder符号化
```
encode(leaf)       → そのleafの文字
encode([case X Y]) → encode(X) + encode(Y) + '-'
```

### デコード（スタックアルゴリズム）
1. 左から右へ読む
2. leaf文字 → スタックにpush
3. `-` → pop Y, pop X, push (X Y)
4. 最後にスタックに残った1つが結果

## Scottエンコーディング（チャーチではない！）

### 真偽値
| 値 | λ式 | SKI表現 | コンパクト | 備考 |
|----|-----|---------|-----------|------|
| true | λx.λy.x | K | `X` | 外延的にはKと同値 |
| false / nil | λx.λy.y | KI | `XD-` | |

**重要**: stars.txtのデータ内で実際に使われるエンコーディングは以下:
- `true` = `S(KK)I` = コンパクト `kXX--D-` （Kとは外延的に同値だが、stars.txt内ではこの形で出現）
- `false` = `KI` = コンパクト `XD-`

パターンマッチ時は `kXX--D-` (S(KK)I) で照合すること。`X` (K) でマッチしようとすると復元を誤る。

### ペア
```
pair(A, B) = S(KK)(S(SI(KA))(KB))
コンパクト: kXX--kkD-X<A>---X<B>---
```

### 数値（2進数、LSB-first）
```
0 = pair(false, nil)  = pair(KI, KI)
1 = pair(true,  nil)  = pair(S(KK)I, KI)
2 = pair(false, pair(true, nil))
3 = pair(true,  pair(true, nil))
n = pair(n%2==1 ? true : false, encode(n>>1))
```

### デコード
ペアチェインからビットを読む: bit0 + 2*bit1 + 4*bit2 + ...

## bracket abstraction (λ式 ↔ SKI)

### 順方向 (λ→SKI)
```
convert((e1 e2)) => (convert(e1) convert(e2))
convert(x)       => x
convert(λx.e)    => bracket[x](convert(e))

bracket[x](x)       => I
bracket[x](e)        => (K e)     (xがe中に出現しない)
bracket[x]((e1 e2))  => (S bracket[x](e1) bracket[x](e2))
```

### 逆方向 (SKI→λ) ← 今回重要！
```
unbracket(x, I)         => x
unbracket(x, (K e))     => e
unbracket(x, (S e1 e2)) => (unbracket(x,e1) unbracket(x,e2))
unbracket(x, その他)    => エラー

revert(e):
  unbracket(x, e) を試す
  → 成功: λx. revert(e') を返す
  → 失敗:
    定数や変数ならそのまま返す
    (e1 e2) なら (revert(e1) revert(e2))
    λx.e' なら λx.revert(e')
```

## 既知の演算子のSKI対応

スプレッドシートより:
| 演算子 | SKI |
|--------|-----|
| not | `SSI-KKI----KSKK--I---` |
| true (S(KK)I) | `SKK--I-` |
| false (KI) | `KI-` |
| format | `I` |
| function [x] [call x] | `I` |
| recursive | `SSSKS--SKK--I---KSI-I----SSKS--SKK--I---KSI-I----` (Y変種) |
| push a b | `SKK--SSI-Ka---Kb---` |
