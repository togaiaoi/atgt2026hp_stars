・bracket abstractionについて

λ式からSKIコンビネータ式への変換には、以下の手順を用いているぞ。 λ式eに対してconvert(e)を計算すると、eと等価なはたらきをするSKIコンビネータ式に変換されるぞ。

convert((e1 e2)) => (convert(e1)  convert(e2))
convert(x) => x
convert(λx. e) => bracket[x](convert(e))
bracket[x](x) => I
bracket[x](e) => (K e)　ただしxがe中に現れないとき
bracket[x]((e1 e2)) => (S bracket[x](e1) bracket[x](e2))

たとえば、`λx.λy.x`は、convert(λx.λy.x) => bracket[x](convert(λy.x)) => bracket[x]
(bracket[y](x)) => bracket[x](Kx) => (S bracket[x](K) bracket[x](x)) => (S(KK)I) のように変換されるぞ。本奴での表記だとSKK--I-だな。

この操作は、以下の手順で逆変換を行うことができるぞ。 revertがconvertの逆操作で、unbracketがbracketの逆操作だ。

unbracket(x,I) => x
unbracket(x,(K e)) => e
unbracket(x,(S e1 e2)) => (unbracket(x,e1) unbracket(x,e2))
unbtacket(x, それ以外の場合) => エラーを起こす
revert(e) => まずunbracket(x,e) を試す。成功してe'が返ってきたなら、λx. revert(e') を返す。エラー
が起きたときは、「まず、定数や変数ならやめる。 そうでないとき、eが(e1 e2)なら(revert(e1) reverte(e2))
を、eがλx.e'ならλx.revert(e')を返す」

例えば、  S(S(KS)(S(KK)I))(KI)、本奴の表記だとSSKS--SKK--I---KI--を復元しようとすると、
revert(S(S(KS)(S(KK)I))(KI)) => λx.revert(S(Kx)I) => λx.λy.revert(x y)となって、このrevertに
は失敗するので、λx.λy.(revert(x) revert(y)) => λx.λy.x y と、ちゃんと復元できるぞ。

・補足(bracket abstractionに関してはちゃんと理解してほしいけど、ここは読まな
くてもいいぞ)

・λx.λy.xはtrueを、λx.λy.x yはチャーチ数1をλ式で表現したものだぞ。本奴でも
それぞれ真偽値や入出力を木に変換する際に使われているぞ。
・bracket abstractionの逆変換ができることの証明はこのようになっているぞ。
『任意のλ式e1,e2について、 convert(e1) = convert(e1) ならば 変数名をつけ
かえることでe1 = e2になる』
【証明】
まず、 bracket[x](e1) = bracket[x](e2) なら e1 = e2である。これは、
bracketによる変換後の木の一番左側の葉が3つの場合分けでどれも違うので、どの場合
分け由来かが復元できるため。　次に、 convertの一意性に関して。同じ場合分け由
来なら変換規則に関する帰納法からいえる。　違う場合分けで由来で一致する可能性
があるのはconvert((e1 e2)) = convert(λx. e)の場合のみだが、
それぞれの変換後の木の一番左の葉に注目すると、 convert((e1 e2))の方がどうして
もconvert(λx. e)より一段以上深くなってしまうので、この場合はありえない。
