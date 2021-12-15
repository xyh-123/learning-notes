

深度学习基本概念
===

![image-20211209144950533](assess/image-20211209144950533.png)

![image-20211209145106676](assess/image-20211209145106676.png)

<img src="assess/image-20211209151533264.png" alt="image-20211209151533264" style="zoom:80%;" />

<img src="assess/image-20211209151555225.png" alt="image-20211209151555225" style="zoom: 50%;" /><img src="assess/image-20211209151620590.png" alt="image-20211209151620590" style="zoom:50%;" />
$$
L=\frac{1}{N}\sum_{i=1}^{n}e_n
$$
$e=|y-\hat{y}|$ , L is mean absolute error (MAE)

$e=(y-\hat{y})^2$ , L is mean square error (MSE)

如果$y,\hat{y}$都是概率分布的话，Loss使用Cross-entropy

 ![image-20211209151648416](assess/image-20211209151648416.png)

<img src="assess/image-20211209151702812.png" alt="image-20211209151702812" style="zoom: 67%;" /><img src="assess/image-20211209151716149.png" alt="image-20211209151716149" style="zoom: 67%;" />

![image-20211209151756575](assess/image-20211209151756575.png)

![image-20211209151854688](assess/image-20211209151854688.png)

![image-20211209151944895](assess/image-20211209151944895.png)

<img src="assess/image-20211209153043177.png" alt="image-20211209153043177" style="zoom: 67%;" /><img src="assess/image-20211209153124202.png" alt="image-20211209153124202" style="zoom: 67%;" />



<img src="assess/image-20211209153419737.png" alt="image-20211209153419737" style="zoom:80%;" />

<img src="assess/image-20211209153606296.png" alt="image-20211209153606296" style="zoom:80%;" />

<img src="assess/image-20211209153638231.png" alt="image-20211209153638231" style="zoom:80%;" />

![image-20211209154348641](assess/image-20211209154348641.png)

![image-20211209164856584](assess/image-20211209164856584.png)

![image-20211209164942386](assess/image-20211209164942386.png)



<img src="assess/image-20211209165115329.png" alt="image-20211209165115329" style="zoom: 50%;" /><img src="assess/image-20211209165147007.png" alt="image-20211209165147007" style="zoom: 50%;" />

<img src="assess/image-20211209165339832.png" alt="image-20211209165339832" style="zoom:50%;" /><img src="assess/image-20211209165319448.png" alt="image-20211209165319448" style="zoom:50%;" />

<img src="assess/image-20211209165419807.png" alt="image-20211209165419807" style="zoom:50%;" /><img src="assess/image-20211209165436268.png" alt="image-20211209165436268" style="zoom:50%;" />

<img src="assess/image-20211209165856073.png" alt="image-20211209165856073" style="zoom:80%;" />

<img src="assess/image-20211209165907228.png" alt="image-20211209165907228" style="zoom:80%;" />

<img src="assess/image-20211209165959977.png" alt="image-20211209165959977" style="zoom:80%;" />

![image-20211209170145728](assess/image-20211209170145728.png)



![image-20211209170347964](assess/image-20211209170347964.png)

![image-20211209170509066](assess/image-20211209170509066.png)

<img src="assess/image-20211209170619661.png" alt="image-20211209170619661" style="zoom:50%;" /><img src="assess/image-20211209170631247.png" alt="image-20211209170631247" style="zoom:50%;" />

![image-20211209172728987](assess/image-20211209172728987.png)

<img src="assess/image-20211209172958582.png" alt="image-20211209172958582" style="zoom:80%;" />

<img src="assess/image-20211209185959018.png" alt="image-20211209185959018" style="zoom:80%;" />

<img src="assess/image-20211209190108789.png" alt="image-20211209190108789" style="zoom:80%;" />

![image-20211209190156960](assess/image-20211209190156960.png)

![image-20211209190221690](assess/image-20211209190221690.png)

<img src="assess/image-20211209190244742.png" alt="image-20211209190244742" style="zoom:80%;" />

<img src="assess/image-20211209190420365.png" alt="image-20211209190420365" style="zoom:80%;" />

![image-20211209190613214](assess/image-20211209190613214.png)

![image-20211209191105560](assess/image-20211209191105560.png)

过拟合解决方法
---



<img src="assess/image-20211209191127676.png" alt="image-20211209191127676" style="zoom: 50%;" /><img src="assess/image-20211209191140667.png" alt="image-20211209191140667" style="zoom: 50%;" />

![image-20211209191228323](assess/image-20211209191228323.png)



![image-20211209191524604](assess/image-20211209191524604.png)

<img src="assess/image-20211209191600835.png" alt="image-20211209191600835" style="zoom:50%;" /><img src="assess/image-20211209191615289.png" alt="image-20211209191615289" style="zoom:50%;" />

![image-20211209192438003](assess/image-20211209192438003.png)

N折交叉验证
---

![image-20211209192652845](assess/image-20211209192652845.png)

<img src="assess/image-20211209193604372.png" alt="image-20211209193604372" style="zoom:80%;" />





<img src="assess/image-20211209193835326.png" alt="image-20211209193835326" style="zoom:80%;" />

![image-20211209194241458](assess/image-20211209194241458.png)

<img src="assess/image-20211209194324723.png" alt="image-20211209194324723" style="zoom:80%;" />

![image-20211209194547155](assess/image-20211209194547155.png)

`eigen values 特征值`

<img src="assess/image-20211209195000571.png" alt="image-20211209195000571" style="zoom:67%;" /><img src="assess/image-20211209195019428.png" alt="image-20211209195019428" style="zoom:67%;" />

![image-20211209195529677](assess/image-20211209195529677.png)

<img src="assess/image-20211209195650045.png" alt="image-20211209195650045" style="zoom:80%;" />

这种方法在真正操作中不长使用，因为$H$比较难算

<img src="assess/image-20211209200157150.png" alt="image-20211209200157150" style="zoom:80%;" />



<img src="assess/image-20211209200346168.png" alt="image-20211209200346168" style="zoom:80%;" />

![image-20211209200421908](assess/image-20211209200421908.png)

Batch and Momentum
---

### Batch

<img src="assess/image-20211209200558921.png" alt="image-20211209200558921" style="zoom:80%;" />

<img src="assess/image-20211209200652006.png" alt="image-20211209200652006" style="zoom:80%;" />

<img src="assess/image-20211209201123943.png" alt="image-20211209201123943" style="zoom:80%;" />

<img src="assess/image-20211209201354676.png" alt="image-20211209201354676" style="zoom:80%;" />

![image-20211209201509595](assess/image-20211209201509595.png)

<img src="assess/image-20211209205414607.png" alt="image-20211209205414607" style="zoom:80%;" />

<img src="assess/image-20211209205732604.png" alt="image-20211209205732604" style="zoom:80%;" />

<img src="assess/image-20211209205818165.png" alt="image-20211209205818165" style="zoom:80%;" />

<img src="assess/image-20211209205850532.png" alt="image-20211209205850532" style="zoom:80%;" />

### Momentum

<img src="assess/image-20211209205945208.png" alt="image-20211209205945208" style="zoom:80%;" />

<img src="assess/image-20211209210018591.png" alt="image-20211209210018591" style="zoom: 67%;" /><img src="assess/image-20211209210212798.png" alt="image-20211209210212798" style="zoom: 50%;" />

<img src="assess/image-20211209210311910.png" alt="image-20211209210311910" style="zoom:80%;" />

<img src="assess/image-20211209210427045.png" alt="image-20211209210427045" style="zoom:80%;" />

<img src="assess/image-20211209210538424.png" alt="image-20211209210538424" style="zoom:80%;" />

Adaptive Learning Rate
---

<img src="assess/image-20211209210813773.png" alt="image-20211209210813773" style="zoom:80%;" />

<img src="assess/image-20211209211205773.png" alt="image-20211209211205773" style="zoom:80%;" />

> 不同的参数需要不同的学习率

<img src="assess/image-20211209211502268.png" alt="image-20211209211502268" style="zoom:80%;" />

<img src="assess/image-20211209211543470.png" alt="image-20211209211543470" style="zoom:80%;" />

<img src="assess/image-20211209211856064.png" alt="image-20211209211856064" style="zoom:80%;" />

<img src="assess/image-20211209212138102.png" alt="image-20211209212138102" style="zoom:50%;" /><img src="assess/image-20211209212158981.png" alt="image-20211209212158981" style="zoom:50%;" />

<img src="assess/image-20211209212557212.png" alt="image-20211209212557212" style="zoom:80%;" />

<img src="assess/image-20211209212935464.png" alt="image-20211209212935464" style="zoom:80%;" />

![image-20211209213010497](assess/image-20211209213010497.png)

<img src="assess/image-20211209213213810.png" alt="image-20211209213213810" style="zoom:80%;" />

<img src="assess/image-20211209213426968.png" alt="image-20211209213426968" style="zoom:80%;" />



Classification
---

![image-20211209213936011](assess/image-20211209213936011.png)

<img src="assess/image-20211209214146375.png" alt="image-20211209214146375" style="zoom: 50%;" /><img src="assess/image-20211209214201994.png" alt="image-20211209214201994" style="zoom: 50%;" />

<img src="assess/image-20211209214505970.png" alt="image-20211209214505970" style="zoom:80%;" />

<img src="assess/image-20211209214654573.png" alt="image-20211209214654573" style="zoom:80%;" />

![image-20211209215011151](assess/image-20211209215011151.png)

<img src="assess/image-20211209215445190.png" alt="image-20211209215445190" style="zoom:80%;" />



Batch Normalization
---

> 批次标准化

<img src="assess/image-20211203102410240.png" alt="image-20211203102410240" style="zoom:50%;" /><img src="assess/image-20211203102507874.png" alt="image-20211203102507874" style="zoom:50%;" />

当$x_1$输入很小而$x_2$输入很大的时候，会出现很难训练的情况，而batch normalization可以解决这种情况



![image-20211203101857532](assess/image-20211203101857532.png)



所有同一个dimension上的数值，经过标准化(normalize)之后，均值(mean)为0，方差(variance)为1，这样做能让梯度下降的时候Loss收敛的更快。

![image-20211203103401358](assess/image-20211203103401358.png)

在$z^i$或者$a^i$处做标准化都可以，如果激活函数是sigmoid的话，在$z^i$处最比较好

<img src="assess/image-20211203103832718.png" alt="image-20211203103832718" style="zoom: 80%;" />

<img src="assess/image-20211203103855852.png" alt="image-20211203103855852" style="zoom: 80%;" />



只对batch里面的example，适用于batch size比较大的时候

![image-20211203104808883](assess/image-20211203104808883.png)

$β$初始值是0向量，$γ$初始值是全1的单位向量

### Testing/Inference

> 在测试的时候，不需要等batch size大小再开始测试，Pytorch已经解决了，不需要自己操作

![image-20211203110738724](assess/image-20211203110738724.png)

![image-20211209221145078](assess/image-20211209221145078.png)





• Batch Renormalization  https://arxiv.org/abs/1702.03275 

• Layer Normalization  https://arxiv.org/abs/1607.06450 

• Instance Normalization  https://arxiv.org/abs/1607.08022 

• Group Normalization  https://arxiv.org/abs/1803.08494 

• Weight Normalization  https://arxiv.org/abs/1602.07868 

• Spectrum Normalization  https://arxiv.org/abs/1705.10941







CNN
===

> 最常用在图像识别

![image-20211201225530090](assess/image-20211201225530090.png)

三个channels代表了R,G,B三个颜色，长宽代表像素数目

![image-20211201225546808](assess/image-20211201225546808.png)

![image-20211201230131822](assess/image-20211201230131822.png)

![image-20211201230231526](assess/image-20211201230231526.png)

同一个感知区(receptive field)可以由相同的神经元训练，感知区也可以重叠。

receptive field大小可以不同，同样receptive field也可以只考虑某些channel,receptive field也可以不是正方形，自己设定。

经典的receptive field：

看全部的channel，所以在描述receptive field的时候只需要描述高和宽就可以。高和宽合起来就叫做Kernel Size

![image-20211201231212122](assess/image-20211201231212122.png)



通常每一个receptive field有一组神经元去训练它，通常是64个

![image-20211201231601620](assess/image-20211201231601620.png)

![image-20211201231934114](assess/image-20211201231934114.png)

一些神经元共享参数(share parameters),因为输入不同，所以即使参数一样输出也不一样

![image-20211201232210795](assess/image-20211201232210795.png)

每一个receptive field只有一组参数，这些参数叫做filter

receptive field + parameter sharing 就是卷积层(convolutional layer)

![image-20211201232933946](assess/image-20211201232933946.png)

第二种理解

![image-20211201233101407](assess/image-20211201233101407.png)

![image-20211201233330208](assess/image-20211201233330208.png)

![image-20211201234138835](assess/image-20211201234138835.png)

Feature Map 可以看成是一张新的图片，第一个convolution的高度根据输入图片而定，彩色则为3，黑白则为1。第二个convolution的filter高度必须设为64，因为第一个convolution输出为64(根据第一层convolution的filter个数而定)个channel

![image-20211201234537971](assess/image-20211201234537971.png)

如果第二层依旧是3×3的filter，相当于提取原图像的5×5，所以当Network叠的越深，同样是3x3的大小的 Filter，看的范围越来越大，依然可以提取更大的pattern

![image-20211201235021054](assess/image-20211201235021054.png)

filter也有bias

Pooling
---

> 相当于缩小图片，减少运算量

<img src="assess/image-20211201235317001.png" alt="image-20211201235317001" style="zoom:50%;" /><img src="assess/image-20211201235332306.png" alt="image-20211201235332306" style="zoom:50%;" />

![image-20211201235442659](assess/image-20211201235442659.png)

![image-20211201235601784](assess/image-20211201235601784.png)



CNN不能处理图片放大和旋转的问题。![image-20211202100443449](assess/image-20211202100443449.png)

Self-attention(自注意力机制)
===

> 为了解决输入一排向量，且向量个数可变，如文本



![image-20211202100920463](assess/image-20211202100920463.png)

Word Embedding——给每一个词汇一个向量，而这个向量是带有语义信息的，一个句子就是一排长度不一的向量

![image-20211202101243665](assess/image-20211202101243665.png)

![image-20211202101550793](assess/image-20211202101550793.png)

![image-20211202101640541](assess/image-20211202101640541.png)

输入是一堆向量，输出有以下几种情况：

- 每一个向量对应一个输出，即输入输出大小一致，如POS Tagging(词性标注)

![image-20211202102033778](assess/image-20211202102033778.png)

![image-20211202103304366](assess/image-20211202103304366.png)

![image-20211202102919219](assess/image-20211202102919219.png)



经过Self-attention输出的vector都是考虑了一整组sequence得到了

![image-20211202103207590](assess/image-20211202103207590.png)





![image-20211202103647023](assess/image-20211202103647023.png)

self-attention的输入可以是原始的input也可以是隐藏层的输出，$a^n,b^n$都要表示一个vector

![image-20211202104123099](assess/image-20211202104123099.png)

$α$表示两个向量相关联的程度，求$α$最常用的方法为Dot-product,在transformer中也是用这个方法

![image-20211202104437089](assess/image-20211202104437089.png)



![image-20211202104946273](assess/image-20211202104946273.png)



![image-20211202105029973](assess/image-20211202105029973.png)

不一定非要用Soft-max,可以用其他的激活函数如relu等

![image-20211202105344181](assess/image-20211202105344181.png)

那一个向量attention score最大，即两个向量之间的关联性很强，$a'_{1,i}$的值越大 $b^1$的值就越接近$v^i$ 

![image-20211202105849674](assess/image-20211202105849674.png)



$b^1$到$b^4$不一定要依次产生，同时被计算出来

![image-20211202110123809](assess/image-20211202110123809.png)

![image-20211202111032912](assess/image-20211202111032912.png)

$W^q,W^k,W^v$都是需要学习参数，$I,Q,K,V$都是矩阵，$I$中有四个column，即$a^1,a^2,a^3,a^4$.

![image-20211202111739851](assess/image-20211202111739851.png)

 ![image-20211202111938003](assess/image-20211202111938003.png)

$A$进行softmax之后，在$A'$中每一column的和都为1

![image-20211202112412962](assess/image-20211202112412962.png)

$O$中的每一个column就是self-attention的输出

![image-20211202151257507](assess/image-20211202151257507.png)

$I$是self-attention的input

![image-20211206191254327](assess/image-20211206191254327.png)

Multi-head Self-attention
---

> 多相关性自注意力

<img src="assess/image-20211202152040461.png" alt="image-20211202152040461" style="zoom:50%;" /><img src="assess/image-20211202152119103.png" alt="image-20211202152119103" style="zoom:50%;" />

head的数目也是一个超参数，需要自己训练

> 前面的self-attention都没有考虑位置的信息，而很多应用场景需要将位置信息考虑进去

Positional Encoding
---

> 目前尚待研究

![image-20211202152626754](assess/image-20211202152626754.png)

Truncated self-attention
---

> 不考虑整个sequence，而是只考虑一小段的sequence

![image-20211202153455659](assess/image-20211202153455659.png)

![image-20211202153710663](assess/image-20211202153710663.png)

![image-20211202153757453](assess/image-20211202153757453.png)



![image-20211202153919462](assess/image-20211202153919462.png)

![image-20211202154057267](assess/image-20211202154057267.png)



self-attention弹性比较大，需要的数据量也更多，在小数据量的情况下，CNN表现比self-attention好

![image-20211202154511846](assess/image-20211202154511846.png)



- 一组向量对应一个输出

![image-20211202102240314](assess/image-20211202102240314.png)



- 不知道输出多少个向量，有机器自己决定

![image-20211202102346227](assess/image-20211202102346227.png)

目前self-attention可以替代RNN处理序列.RNN无法平行处理Vector，只能一次处理Vector，也很难考虑sequence中最前面的vector.

![image-20211202155003894](assess/image-20211202155003894.png)



Self-attention for Graph
---

![image-20211202160154537](assess/image-20211202160154537.png)

只计算有边关联的节点之间的attention score，没有关联的节点之间attention score设为0

self-attention的缺点是运算量很大，如何减少运算量是未来研究的重点

![image-20211202160604666](assess/image-20211202160604666.png)









Transformer
===

> transformer是一个sequence-to-sequence(seq2seq)的model

![image-20211202161207018](assess/image-20211202161207018.png)

![image-20211202161839451](assess/image-20211202161839451.png)



![image-20211202162857666](assess/image-20211202162857666.png)

![image-20211202162303333](assess/image-20211202162303333.png)

**seq2seq可以用于多标签分类**

![image-20211202162621858](assess/image-20211202162621858.png)

**seq2seq可以用于目标检测**

![image-20211202162807508](assess/image-20211202162807508.png)





![image-20211202163025365](assess/image-20211202163025365.png)

Encoder
---

![image-20211202163138603](assess/image-20211202163138603.png)

![image-20211202163509960](assess/image-20211202163509960.png)

在transformer中的self-attention有一点不同，使用了residual connection

![image-20211202164349559](assess/image-20211202164349559.png)



Layer Normalization做的事情比bacth Normalization更简单一点，对输入的向量计算mean(均值)和standard deviation(标准偏差)。对同一个feature，同一个example里面的不同dimension计算mean和standard deviation

batch Normalization对同一个dimension不同的feature，对不同的example，不同的feature的同一个dimension计算mean和standard deviation

![image-20211202165950906](assess/image-20211202165950906.png)

<img src="assess/image-20211206191638068.png" alt="image-20211206191638068" style="zoom: 67%;" />

<img src="assess/image-20211206191803914.png" alt="image-20211206191803914" style="zoom:80%;" />

![image-20211202170333748](assess/image-20211202170333748.png)

Decoder
---

### Autoregressive(AT)

> 自回归

![image-20211202170516622](assess/image-20211202170516622.png)



![image-20211202183651926](assess/image-20211202183651926.png)

在字符表中额外添加的符号，代表开始和结尾，如<bos>,<end>

![image-20211202184053038](assess/image-20211202184053038.png)



![image-20211202184149199](assess/image-20211202184149199.png)

![image-20211202184540952](assess/image-20211202184540952.png)



Masked Self-attention不考虑右边的vector,只考虑左边的vector。

<img src="assess/image-20211202184620371.png" alt="image-20211202184620371" style="zoom:50%;" /><img src="assess/image-20211202184347656.png" alt="image-20211202184347656" style="zoom:50%;" />

$b^1,b^2,b^3,b^4$是依次产生的，不是一次性产生

![image-20211202184857818](assess/image-20211202184857818.png)

![image-20211202190018030](assess/image-20211202190018030.png)

![image-20211202190048153](assess/image-20211202190048153.png)

### Non-autogressive(NAT)

![image-20211202190542965](assess/image-20211202190542965.png)

AT的performance通常比NAT要好

**Encoder与Decoder之间传递消息**
---

![image-20211202190845576](assess/image-20211202190845576.png)



![image-20211202191450372](assess/image-20211202191450372.png)



![image-20211202191632604](assess/image-20211202191632604.png)

![image-20211202230525680](assess/image-20211202230525680.png)



Training
---

> 以上都是训练好模型之后，怎样input一个sequence通过模型得到对应的输出，接下来是模型的训练过程

![image-20211202230730800](assess/image-20211202230730800.png)

![image-20211202231219805](assess/image-20211202231219805.png)

![image-20211202231748138](assess/image-20211202231748138.png)

### Train Tips

> seq2seq这种模型的训练技巧

从输入复制作语句为输出的一部分的模型，如Point Network

![image-20211202232252784](assess/image-20211202232252784.png)



![image-20211202232317052](assess/image-20211202232317052.png)



![image-20211202232339599](assess/image-20211202232339599.png)

### Guided Attention

![image-20211202233059415](assess/image-20211202233059415.png)

guided attention将限制放到training中

### Beam Search(束搜索)

> 如果任务非常的明确，如语音识别，Beam Search效果会有帮助，当需要机器有创造力的时候，没有明确答案输出的时候，需要在decoder里面加入随机性(加noise-噪音)

![image-20211202233612593](assess/image-20211202233612593.png)

![image-20211202234950483](assess/image-20211202234950483.png)

Blue Score是通过两个完整的句子算出来的，本身很复杂，不可以微分，也就不能做梯度下降，所以不能作为训练时的损失函数。

当遇到无法Optimize的Loss Function，用RL进行训练。将Loss Function当做RL的Reward(奖励函数)，将自己的Decoder当做是Agent

![image-20211202235730408](assess/image-20211202235730408.png)

如何解决exposure bias,给Decoder中加入一些错误的信息，这样的方法叫做Scheduled Sampling

![image-20211203000107109](assess/image-20211203000107109.png)

PixelRNN
===

![image-20211208211515976](assess/image-20211208211515976.png)

语音合成

![](assess/gift1.gif)

![image-20211208215146924](assess/image-20211208215146924.png)

<img src="assess/image-20211208215414946.png" alt="image-20211208215414946" style="zoom:50%;" /><img src="assess/image-20211208215726831.png" alt="image-20211208215726831" style="zoom:50%;" />

![image-20211208215827176](assess/image-20211208215827176.png)

<img src="assess/image-20211208220053686.png" alt="image-20211208220053686" style="zoom: 80%;" />

VAE
===

> 2013年提出

<img src="assess/image-20211208220446896.png" alt="image-20211208220446896" style="zoom: 80%;" />



![image-20211208220740084](assess/image-20211208220740084.png)

同时需要Minimize两项才行

![image-20211208221038709](assess/image-20211208221038709.png)

![image-20211208221535577](assess/image-20211208221535577.png)

<img src="assess/image-20211208221940424.png" alt="image-20211208221940424" style="zoom: 50%;" /><img src="assess/image-20211208222023895.png" alt="image-20211208222023895" style="zoom:50%;" />

![image-20211208222438607](assess/image-20211208222438607.png)

![image-20211209101750597](assess/image-20211209101750597.png)

![image-20211208222655648](assess/image-20211208222655648.png)

![image-20211208222848615](assess/image-20211208222848615.png)

![image-20211208223405895](assess/image-20211208223405895.png)

![image-20211208224014190](assess/image-20211208224014190.png)

![image-20211208224030503](assess/image-20211208224030503.png)

![image-20211208224747003](assess/image-20211208224747003.png)

![image-20211209095023562](assess/image-20211209095023562.png)

![image-20211209095534951](assess/image-20211209095534951.png)

![image-20211209095654577](assess/image-20211209095654577.png)

![image-20211209100105937](assess/image-20211209100105937.png)

![image-20211209100412624](assess/image-20211209100412624.png)

![image-20211209101548337](assess/image-20211209101548337.png)

Flow
===

> 

![image-20211209103919347](assess/image-20211209103919347.png)

![image-20211209104500646](assess/image-20211209104500646.png)

![image-20211209104651499](assess/image-20211209104651499.png)

Jacobian
---

![image-20211209105933543](assess/image-20211209105933543.png)

Determinant
---

![image-20211209110227053](assess/image-20211209110227053.png)

![image-20211209110423312](assess/image-20211209110423312.png)

Change of Variable Theorem
---

![image-20211209110720881](assess/image-20211209110720881.png)

![image-20211209111404470](assess/image-20211209111404470.png)

![img](file:///C:\Users\xyh\Documents\Tencent Files\447767881\Image\C2C\Image1\I18CJO$~$R%7O3}3}N4W5[L.png)

![image-20211209112207836](assess/image-20211209112207836.png)

$△x_{11}$表示$z_1$改变的时候$x_1$的改变量

$△x_{21}$表示$z_1$改变的时候$x_2$的改变量

$△x_{12}$表示$z_2$改变的时候$x_1$的改变量

$△x_{22}$表示$z_2$改变的时候$x_2$的改变量

![image-20211209133912562](assess/image-20211209133912562.png)

![image-20211209134205571](assess/image-20211209134205571.png)

![image-20211209134416620](assess/image-20211209134416620.png)

![image-20211209134513789](assess/image-20211209134513789.png)

![image-20211209134731445](assess/image-20211209134731445.png)

![image-20211209135037055](assess/image-20211209135037055.png)

![image-20211209135309823](assess/image-20211209135309823.png)



![image-20211209135427170](assess/image-20211209135427170.png)

![image-20211209140334199](assess/image-20211209140334199.png)

![image-20211209140635777](assess/image-20211209140635777.png)

当使用这个方法做图片生成的时候，通常有两种方法将图片分为两个部分，一种方法是将图片中纵轴和横轴加起来是偶数的复制，奇数做transform。第二种方法是对图片的某几个channels做复制，其他的作transform，每一层都不一样。这两种方法可以交替使用。

![image-20211209141359083](assess/image-20211209141359083.png)

![image-20211209141946544](assess/image-20211209141946544.png)

![image-20211209142035895](assess/image-20211209142035895.png)

![image-20211209142304066](assess/image-20211209142304066.png)

![image-20211209142532151](assess/image-20211209142532151.png)

**语音合成**

![image-20211209142632313](assess/image-20211209142632313.png)















GAN
===

> Generative Adversial NetWork 生成式对抗网络，在生成模型中表现比较好

各种GAN模型：https://github.com/hindupuravinash/the-gan-zoo



Generator
---

> 生成器

![image-20211203145452891](assess/image-20211203145452891.png)

$Z$是随机生成的，所以每一次的输入$Z$都不同，是从Distribution中取样出来的，Distribution可以是一个uniform distribution，Distribution形状自己定。

<img src="assess/image-20211203143222963.png" alt="image-20211203143222963" style="zoom:80%;" />

$Z$是Normal Distribution(正态分布)中sample出来的向量，维度自己定

Discriminator
---



![image-20211203150858064](assess/image-20211203150858064.png)

![image-20211203150940334](assess/image-20211203150940334.png)

<img src="assess/image-20211203151024789.png" alt="image-20211203151024789" style="zoom:80%;" />

<img src="assess/image-20211203151150442.png" alt="image-20211203151150442" style="zoom:80%;" />

Generator不断更新参数，调整输出，Discriminator不断鉴别Generator的输出与真实输出之间的差距，不断的重复，直至生成的东西与真实的东西越来越接近。



训练步骤：

- 1.初始化generator和discriminator
- 2.固定generator训练discriminator，用generator的输出和数据库中真实样本训练discriminator，使得discriminator能区分真实样本与生成样本
- 3.固定discriminator训练generator,调整generator的参数，直到discriminator输出的值越大越好，即生成样本越接近真实样本
- 反复重复步骤2和3



<img src="assess/image-20211203151901881.png" alt="image-20211203151901881" style="zoom: 50%;" /><img src="assess/image-20211203152206623.png" alt="image-20211203152206623" style="zoom:50%;" />

<img src="assess/image-20211203152827604.png" alt="image-20211203152827604" style="zoom:80%;" />

Theory
---

![image-20211203153322379](assess/image-20211203153322379.png)



<img src="assess/image-20211203153712994.png" alt="image-20211203153712994" style="zoom: 80%;" />

<img src="assess/image-20211203154129229.png" alt="image-20211203154129229" style="zoom:80%;" />

![image-20211203154520115](assess/image-20211203154520115.png)

WGAN
---

![image-20211203155315261](assess/image-20211203155315261.png)

![image-20211203155944261](assess/image-20211203155944261.png)

$D$必须是一个平滑的function

![image-20211203160624063](assess/image-20211203160624063.png)

SNGAN(Spectral Normalization GAN)

GAN for Sequence Generation
---



![image-20211203161802523](assess/image-20211203161802523.png)



Evaluation of Generation
---

![image-20211203164255159](assess/image-20211203164255159.png)

Mode Collapse 模型塌陷

> 目前没有很好的解决方法

一种解决方法是当训练到Mode Collapse的时候，模型训练停止，用之前最近一次没有出现Mode Collapse的模型

Mode Dropping 

![image-20211203164817359](assess/image-20211203164817359.png)

<img src="assess/image-20211203164940831.png" alt="image-20211203164940831" style="zoom:50%;" /><img src="assess/image-20211203165000319.png" alt="image-20211203165000319" style="zoom:50%;" />

<img src="assess/image-20211203165525879.png" alt="image-20211203165525879" style="zoom:80%;" />

![image-20211203170621851](assess/image-20211203170621851.png)

Conditional Generation
---

> 有条件的生成模型

<img src="assess/image-20211203170820648.png" alt="image-20211203170820648" style="zoom:80%;" />

![image-20211203171226860](assess/image-20211203171226860.png)

<img src="assess/image-20211203171432262.png" alt="image-20211203171432262" style="zoom:80%;" />

<img src="assess/image-20211203171624926.png" alt="image-20211203171624926" style="zoom:80%;" />

Learning from  Unpaired Data(无监督)
---

### Cycle GAN

<img src="assess/image-20211203172523106.png" alt="image-20211203172523106" style="zoom:50%;" /><img src="assess/image-20211203172538677.png" alt="image-20211203172538677" style="zoom:50%;" />

<img src="assess/image-20211203173311506.png" alt="image-20211203173311506" style="zoom:50%;" /><img src="assess/image-20211203173818315.png" alt="image-20211203173818315" style="zoom:50%;" />

第一个generator由Domian X生成Domian y中的图片，第一个generator将生成的Domian y中的图片还原为Domian X，还原与输入越接近越好

<img src="assess/image-20211203174036569.png" alt="image-20211203174036569" style="zoom:80%;" />

<img src="assess/image-20211203175045925.png" alt="image-20211203175045925" style="zoom:80%;" />



<img src="assess/image-20211203175111290.png" alt="image-20211203175111290" style="zoom:80%;" />



<img src="assess/image-20211203175311304.png" alt="image-20211203175311304" style="zoom:80%;" />



<img src="assess/image-20211203175326024.png" alt="image-20211203175326024" style="zoom: 80%;" />



Word Embeding
===

<img src="assess/image-20211204104108599.png" alt="image-20211204104108599" style="zoom:80%;" />

<img src="assess/image-20211204104302757.png" alt="image-20211204104302757" style="zoom: 80%;" />

<img src="assess/image-20211204104517660.png" alt="image-20211204104517660" style="zoom: 80%;" />

怎样利用上下文

<img src="assess/image-20211204105103258.png" alt="image-20211204105103258" style="zoom: 80%;" />



![image-20211204110219188](assess/image-20211204110219188.png)

<img src="assess/image-20211204110351320.png" alt="image-20211204110351320" style="zoom:80%;" />

<img src="assess/image-20211204110825855.png" alt="image-20211204110825855" style="zoom:50%;" /><img src="assess/image-20211204111006190.png" alt="image-20211204111006190" style="zoom:50%;" />

![image-20211204111416592](assess/image-20211204111416592.png)

![image-20211204111542966](assess/image-20211204111542966.png)



<img src="assess/image-20211204111643414.png" alt="image-20211204111643414" style="zoom:80%;" />

![image-20211204131858007](assess/image-20211204131858007.png)





**Self-Supervised Learning**(自监督学习)
===

<img src="assess/image-20211204135502732.png" alt="image-20211204135502732" style="zoom:80%;" />



**BERT**
---

> Bidirectional Encoder Representations from Transformers，有340M个参数
>
> 相当于Word Emebeding的升级版



![image-20211204140329033](assess/image-20211204140329033.png)



<img src="assess/image-20211204140952568.png" alt="image-20211204140952568" style="zoom: 50%;" /><img src="assess/image-20211204141223115.png" alt="image-20211204141223115" style="zoom: 50%;" />



![image-20211204141512208](assess/image-20211204141512208.png)

CLS：当机器看到CLS这个特殊的token的时候，就是要产生一个Embeding(向量)，这个Embeding代表了整个BERT输入的token的信息

SOP：如果两个相接的句子顺序，BERT需要判断Yes,如果两个句子顺序颠倒输入，BERT需要判断NO,SOP实现比较困难



![image-20211204141818779](assess/image-20211204141818779.png)

上游的任务属于无监督学习，不需要标注的训练。下游的任务**`需要有少量有标注`**的资料。整个Pre-train(找到一组好的初始化参数)加上Fine-tune就是半监督学习(semi-supervised leraning) 

产生Bert的过程，就是pre-train.测试self-supervised模型的能力，一般使用GLUE

<img src="assess/image-20211204142134226.png" alt="image-20211204142134226" style="zoom:80%;" />

![image-20211204161941899](assess/image-20211204161941899.png)



![image-20211204161958504](assess/image-20211204161958504.png)

### 输入sequence输出class

![image-20211204144012123](assess/image-20211204144012123.png)



![image-20211204144231938](assess/image-20211204144231938.png)

### 输入sequence输出相同长度的sequence

<img src="assess/image-20211204145015457.png" alt="image-20211204145015457" style="zoom:80%;" />

### 输入two sequence输出class

![image-20211204145210294](assess/image-20211204145210294.png)

### Extraction-based Question Answering

![image-20211204145620550](assess/image-20211204145620550.png)

<img src="assess/image-20211204150043892.png" alt="image-20211204150043892" style="zoom:50%;" /><img src="assess/image-20211204150105776.png" alt="image-20211204150105776" style="zoom:50%;" />

![image-20211204151726114](assess/image-20211204151726114.png)

Corrupted:将输入的句子损坏。以下两篇论文为Corrupted的方法。

![image-20211205110746610](assess/image-20211205110746610.png)

<img src="assess/image-20211205111343999.png" alt="image-20211205111343999" style="zoom:80%;" />

<img src="assess/image-20211205111432629.png" alt="image-20211205111432629" style="zoom:80%;" />

![image-20211205111826696](assess/image-20211205111826696.png)

<img src="assess/image-20211205111953584.png" alt="image-20211205111953584" style="zoom:50%;" /><img src="assess/image-20211205112112047.png" alt="image-20211205112112047" style="zoom:50%;" />

<img src="assess/image-20211205112451990.png" alt="image-20211205112451990" style="zoom:80%;" />

![image-20211205112837657](assess/image-20211205112837657.png)

两个句子是相邻的，则让这两个句子的embeding越相似











![image-20211204163359074](assess/image-20211204163359074.png)



SEP：特殊分割token，开始产生out sequence vector

![image-20211204164033830](assess/image-20211204164033830.png)

### Fine-tune

微调网络，就是用别人训练好的模型，加上我们自己的数据，来训练新的模型。好处在于不用完全重新训练模型，从而提高效率

![image-20211204170316403](assess/image-20211204170316403.png)

![image-20211204171035485](assess/image-20211204171035485.png)



<img src="assess/image-20211204171151631.png" alt="image-20211204171151631" style="zoom:50%;" /><img src="assess/image-20211204171352832.png" alt="image-20211204171352832" style="zoom:50%;" />

![image-20211205093614706](assess/image-20211205093614706.png)

![image-20211205094312623](assess/image-20211205094312623.png)

W1和W2表示那一层比较重要

### Multi-BERT

> 多语言BERT

![image-20211205141042785](assess/image-20211205141042785.png)

![image-20211205141135041](assess/image-20211205141135041.png)



![image-20211205140331498](assess/image-20211205140331498.png)

![image-20211205140526515](assess/image-20211205140526515.png)

![image-20211205140638784](assess/image-20211205140638784.png)

![image-20211205141550180](assess/image-20211205141550180.png)

Pre-train模型如何产生
---

![image-20211205100131215](assess/image-20211205100131215.png)

![image-20211205100232996](assess/image-20211205100232996.png)

![image-20211205100408769](assess/image-20211205100408769.png)

用self-attention需要注意限制attention的位置，只能attention左边包括本身的vector，不能attention右边的vector

![image-20211205103443900](assess/image-20211205103443900.png)

更好的Masking方法

<img src="assess/image-20211205103756710.png" alt="image-20211205103756710"  /> 



<img src="assess/image-20211205103941740.png" alt="image-20211205103941740" style="zoom:80%;" />

<img src="assess/image-20211205104129654.png" alt="image-20211205104129654" style="zoom:50%;" /><img src="assess/image-20211205104149822.png" alt="image-20211205104149822" style="zoom:50%;" />

### XLNet

<img src="assess/image-20211205110141513.png" alt="image-20211205110141513" style="zoom:50%;" /><img src="assess/image-20211205110201214.png" alt="image-20211205110201214" style="zoom:50%;" />







<img src="assess/image-20211204151837244.png" alt="image-20211204151837244" style="zoom:80%;" />

![image-20211204152516341](assess/image-20211204152516341.png)



<img src="assess/image-20211204152811615.png" alt="image-20211204152811615" style="zoom:50%;" /><img src="assess/image-20211204152830493.png" alt="image-20211204152830493" style="zoom:50%;" />



<img src="assess/image-20211205101402911.png" alt="image-20211205101402911" style="zoom:50%;" /><img src="assess/image-20211204153031320.png" alt="image-20211204153031320" style="zoom: 50%;" />



<img src="assess/image-20211204161432515.png" alt="image-20211204161432515" style="zoom:80%;" />

<img src="assess/image-20211204161550412.png" alt="image-20211204161550412" style="zoom:80%;" />

以下论文包含各种Pre-train方法

<img src="assess/image-20211205114850021.png" alt="image-20211205114850021" style="zoom:80%;" />

GPT
---

> 预测接下的token,具有生成的能力

![image-20211205142033348](assess/image-20211205142033348.png)

GPT类似于transformer的decoder,输入的时候不能一次性输入，预测第$i$个token时不能，模型看不到$i$及之后的token

<img src="assess/image-20211205145038188.png" alt="image-20211205145038188" style="zoom: 80%;" />

<img src="assess/image-20211205145110545.png" alt="image-20211205145110545" style="zoom: 80%;" />



Auto-Encoder
===

> 是self-supervised方法中的一种

![image-20211205160548280](assess/image-20211205160548280.png)

<img src="assess/image-20211205160758673.png" alt="image-20211205160758673" style="zoom:80%;" />

<center> <img src="%E4%B8%93%E7%94%A8%E5%90%8D%E8%AF%8D.assets/image-20211205161631447.png" alt="image-20211205161631447" style="zoom:80%;" /></center>



Bert相当于一个De-noising Encoder

![image-20211205161941405](assess/image-20211205161941405.png)

Feature Disentanglement
---

> 把Vector中纠缠在一起的重要信解开

<img src="assess/image-20211205162350024.png" alt="image-20211205162350024" style="zoom:67%;" /><img src="assess/image-20211205162413687.png" alt="image-20211205162413687" style="zoom:67%;" />

<img src="assess/image-20211205162742508.png" alt="image-20211205162742508" style="zoom:67%;" /><img src="assess/image-20211205162819535.png" alt="image-20211205162819535" style="zoom:67%;" />

Discrete Latent Representation
---

> 离散潜在表示

将Embeding用Binary或者one-hot表示

![image-20211205163338871](assess/image-20211205163338871.png)

![image-20211205163740975](assess/image-20211205163740975.png)

**用文本作为Embeding**

![image-20211205164327410](assess/image-20211205164327410.png)

<img src="assess/image-20211205164416442.png" alt="image-20211205164416442" style="zoom:80%;" />

Decoder可以用来做生成器，VAE

![image-20211205164554742](assess/image-20211205164554742.png)

Auto-Encoder可以用于压缩

![image-20211205164640029](assess/image-20211205164640029.png)

Anomaly Detection
---

> 异常检测，Auto-Encoder是解决这类问题中的一个方法

![image-20211205165031843](assess/image-20211205165031843.png)

<img src="assess/image-20211205165108784.png" alt="image-20211205165108784" style="zoom:80%;" />

![image-20211205165417204](assess/image-20211205165417204.png)

<img src="assess/image-20211205165558170.png" alt="image-20211205165558170" style="zoom:67%;" /><img src="assess/image-20211205165614108.png" alt="image-20211205165614108" style="zoom:67%;" />

![image-20211205165739039](assess/image-20211205165739039.png)

Adversarial Attrack
===

> 恶意攻击

![image-20211205170408785](assess/image-20211205170408785.png)

<img src="assess/image-20211205170637127.png" alt="image-20211205170637127" style="zoom: 50%;" /><img src="assess/image-20211205170806044.png" alt="image-20211205170806044" style="zoom: 50%;" />

![image-20211205172850457](assess/image-20211205172850457.png)

![image-20211205213947054](assess/image-20211205213947054.png)

![image-20211205214448910](assess/image-20211205214448910.png)

![image-20211205214723635](assess/image-20211205214723635.png)

![image-20211205215132708](assess/image-20211205215132708.png)

<img src="assess/image-20211206084634512.png" alt="image-20211206084634512" style="zoom:80%;" />

![image-20211205215358533](assess/image-20211205215358533.png)

![image-20211205215926520](assess/image-20211205215926520.png)

-ResNet-152表示除ResNet-152以外四个模型

<img src="assess/image-20211205220626166.png" alt="image-20211205220626166" style="zoom:80%;" />



Defense
---

> 防御

![image-20211206091034532](assess/image-20211206091034532.png)

![image-20211206091103620](assess/image-20211206091103620.png)

![image-20211206091450542](assess/image-20211206091450542.png)

![image-20211206091633976](assess/image-20211206091633976.png)

### 主动防御

![image-20211206091907651](assess/image-20211206091907651.png)

这种方法能够进行数据增强，使模型不容易过拟合

Explainable ML
===

> 机器学习的可解释性

深度学习的解释性能力比较差，Explainable没有明确的目标

![image-20211206094730031](assess/image-20211206094730031.png)



Local Explanation
---

![image-20211206094913801](assess/image-20211206094913801.png)

![image-20211206095038735](assess/image-20211206095038735.png)

![image-20211206095208367](assess/image-20211206095208367.png)

$|\frac{△e}{△x}|$表示pixels的重要性

<img src="assess/image-20211206101300674.png" alt="image-20211206101300674" style="zoom:67%;" /><img src="assess/image-20211206101332305.png" alt="image-20211206101332305" style="zoom:67%;" />

![image-20211206101406488](assess/image-20211206101406488.png)

![image-20211206101536554](assess/image-20211206101536554.png)

SmoothGrad：随机给输入的图片添加噪音，添加噪音的saliency maps,然后平均它们

![image-20211206101946997](assess/image-20211206101946997.png)

神经网络怎么处理输入的数据

<img src="assess/image-20211206102116525.png" alt="image-20211206102116525" style="zoom: 67%;" />

<img src="assess/image-20211206103501846.png" style="zoom:67%;" /><img src="assess/image-20211206103517442.png" alt="image-20211206103517442" style="zoom:67%;" />

![image-20211206105312798](assess/image-20211206105312798.png)

Transfer Learning
===

![image-20211206192053138](assess/image-20211206192053138.png)

![image-20211206192406148](assess/image-20211206192406148.png)



![image-20211206165855819](assess/image-20211206165855819.png)



![image-20211206161909878](assess/image-20211206161909878.png)

<img src="assess/image-20211206162045739.png" alt="image-20211206162045739" style="zoom:80%;" />

![image-20211206163311080](assess/image-20211206163311080.png)

**Conservative Training**

> 对模型进行微调的时候，做一些限制，防止对模型改动太大，出现过拟合现象

![image-20211206164002247](assess/image-20211206164002247.png)

**网络层迁移(Layer Transfer)**

![image-20211206164112636](assess/image-20211206164112636.png)

![image-20211206164438700](assess/image-20211206164438700.png)

![image-20211206164738738](assess/image-20211206164738738.png)

Multitask Learing
---

![image-20211206165336244](assess/image-20211206165336244.png)

![image-20211206165439363](assess/image-20211206165439363.png)



Zero-shot Learing
---

![image-20211206170716572](assess/image-20211206170716572.png)

<img src="assess/image-20211206171209829.png" alt="image-20211206171209829" style="zoom: 50%;" /><img src="assess/image-20211206171226206.png" alt="image-20211206171226206" style="zoom: 50%;" />

<img src="assess/image-20211206171424776.png" alt="image-20211206171424776" style="zoom: 50%;" /><img src="assess/image-20211206171714318.png" alt="image-20211206171714318" style="zoom:50%;" />













Domain Adaptation
===

> 领域自适应，可以看做是迁移学习的另一个环节

![image-20211206105628873](assess/image-20211206105628873.png)

![image-20211206110142139](assess/image-20211206110142139.png)

![image-20211206110417127](assess/image-20211206110417127.png)

![image-20211206110541406](assess/image-20211206110541406.png)

怎样找Feature Extractor，可以将一般的Classifier分成Feature Extractor和Label Predictor.

<img src="assess/image-20211206111228064.png" alt="image-20211206111228064" style="zoom: 80%;" />

目标是Source和Target的差异，

![image-20211206150646756](assess/image-20211206150646756.png)

Domain Classifier就是一个二元分类器。最原始的方法并不是最好的。

![image-20211206170325477](assess/image-20211206170325477.png)

![image-20211206155933109](assess/image-20211206155933109.png)

![image-20211206160143834](assess/image-20211206160143834.png)

如何解决source和target数据类别不一致的问题。

![image-20211206160216309](assess/image-20211206160216309.png)

![image-20211206160643832](assess/image-20211206160643832.png)

![image-20211206160756901](assess/image-20211206160756901.png)

RL
===

>  Reinforcement Learning，适用于收集标注资料很困难的时候，正确答案人类也不知道是什么的时候。

![image-20211206193706155](assess/image-20211206193706155.png)

![image-20211206194527500](assess/image-20211206194527500.png)

Environment在这个过程中通过Reward不断的告诉Actor,当前输入的Action是好的还是不好的。目标是找一个最大的Reward的总和。

<img src="assess/image-20211206195117824.png" alt="image-20211206195117824" style="zoom:80%;" />

<img src="assess/image-20211206194958093.png" alt="image-20211206194958093" style="zoom: 80%;" />

<img src="assess/image-20211206195018691.png" alt="image-20211206195018691" style="zoom: 80%;" />

![image-20211206195338860](assess/image-20211206195338860.png)

![image-20211206195348931](assess/image-20211206195348931.png)

![image-20211206195632264](assess/image-20211206195632264.png)

![image-20211206195915922](assess/image-20211206195915922.png)

最后采用sample的好处是，即使机器看到相同的observation,机器产生的Action也会不同。

<img src="assess/image-20211206201147795.png" alt="image-20211206201147795" style="zoom:50%;" /><img src="assess/image-20211206201221380.png" alt="image-20211206201221380" style="zoom:50%;" />

目标是Total reward(return)越大越好，$Loss=-return$

![image-20211206202218508](assess/image-20211206202218508.png)

Env和Reward一般都具有随机性，RL是一个随机性特别大的问题，同样的Network可能每次的输出都不同

Policy Gradient
---

![image-20211206203112300](assess/image-20211206203112300.png)

![image-20211206203321320](assess/image-20211206203321320.png)

<img src="assess/image-20211206203431732.png" alt="image-20211206203431732" style="zoom:67%;" /><img src="assess/image-20211206203443734.png" alt="image-20211206203443734" style="zoom:67%;" />

控制$A_N$让模型知道，哪个行为是希望的，哪个行为是强烈不希望的。不希望Actor做$a$动作，是指Actor可以做其他的事情，什么都不做也可以是一个action。在同一个环境中可以同时决定想做和不想做的动作，想做的动作分数高，不想做的动作分数低，分数的高低决定出现该动作的概率。

 ![image-20211206205807661](assess/image-20211206205807661.png)

![image-20211206210443157](assess/image-20211206210443157.png)



![image-20211206210842947](assess/image-20211206210842947.png)

![image-20211206211519666](assess/image-20211206211519666.png)

![image-20211206211815780](assess/image-20211206211815780.png)

不同的RL一般是修改$A$的定义,obtain data是收集数据的过程。

![image-20211206211824931](assess/image-20211206211824931.png)

每次更新完参数之后，就需要重新收集训练资料，因为上一歩收集的资料，不一定适合下次一定的参数更新

![image-20211206211925997](assess/image-20211206211925997.png)

![image-20211206212216101](assess/image-20211206212216101.png)

![image-20211206212950000](assess/image-20211206212950000.png)

![image-20211206213015135](assess/image-20211206213015135.png)

Off-policy-PPO
---

> Proximal Policy OPtimization



![image-20211207095206690](assess/image-20211207095206690.png)

![image-20211207095617174](assess/image-20211207095617174.png)

Sample的次数必须要多

![image-20211207100126509](assess/image-20211207100126509.png)

![image-20211207101030194](assess/image-20211207101030194.png)

![image-20211207101422893](assess/image-20211207101422893.png)

![image-20211207101954797](assess/image-20211207101954797.png)



$clip(x,1-ε,1+ε)$的含义是第一项小于第二项输出$1-ε$，第一项大于第三项输出$1+ε$



![image-20211207102615129](assess/image-20211207102615129.png)

![image-20211207102547987](assess/image-20211207102547987.png)

![image-20211207102753017](assess/image-20211207102753017.png)



Exploration
---

> 增加Actor产生Action的随机性

![image-20211206213319527](assess/image-20211206213319527.png)

Actor-Critic
---

> 评估Actor的好坏

![image-20211207104103889](assess/image-20211207104103889.png)

![image-20211207104258698](assess/image-20211207104258698.png)

MC需要完成整个游戏才行

![image-20211207104706816](assess/image-20211207104706816.png)

![image-20211207105142001](assess/image-20211207105142001.png)

<img src="assess/image-20211207105426779.png" alt="image-20211207105426779" style="zoom:67%;" /><img src="assess/image-20211207105440632.png" alt="image-20211207105440632" style="zoom:67%;" />

![image-20211207105851698](assess/image-20211207105851698.png)

![image-20211207111639817](assess/image-20211207111639817.png)



在$S_{t+1}$的observation下得到的Cumulative Reward的期望值为$V^θ(S_{t+1})$ , $V^θ(S_{t+1})+r_t$ 表示$S_t$这个位置采取动作$a_t$以后跳到$S_{t+1}$后得到的期望值。$r_t$表示$S_t$采取$a_t$这个动作后得到的reward。

$A_t=V^θ(S_{t+1})+r_t-V^θ(S_t)$,表示采取$a_t$这个确定的action获得的reward与不采取$a_t$,而是随机sample的action的reward之间期望值的差距。$A_t>0$表示$a_t$这个action比随机sample要好，反之$a_t$这个action比较差。这个方法叫做`Advantage Actor-Critic`

![image-20211207114012067](assess/image-20211207114012067.png)

DQN
---

https://youtu.be/o_g9JUMw1Oc

https://youtu.be/2-zGCx4iv_k

![image-20211207114357716](assess/image-20211207114357716.png)

Reward Shaping
---

> 定义额外的reward去引导actor

![image-20211207144249405](assess/image-20211207144249405.png)

![image-20211207145008089](assess/image-20211207145008089.png)

![image-20211207145547615](assess/image-20211207145547615.png)

No Reward: Learning from Demonstration
---

![image-20211207150128837](assess/image-20211207150128837.png)

![image-20211207150514611](assess/image-20211207150514611.png)

### Inverse RL(IRL)

> 常用于训练机械手臂

![image-20211207151001842](assess/image-20211207151001842.png)

![image-20211207151125611](assess/image-20211207151125611.png)

![image-20211207152427538](assess/image-20211207152427538.png)

![image-20211207152510064](assess/image-20211207152510064.png)

![image-20211207152827309](assess/image-20211207152827309.png)



Life Long Learning(LLL)
===

> 终生学习,也叫作Continuous Learning

![image-20211207153342069](assess/image-20211207153342069.png)

![image-20211207153451156](assess/image-20211207153451156.png)

<img src="assess/image-20211207153752821.png" alt="image-20211207153752821" style="zoom: 67%;" /><img src="assess/image-20211207153806186.png" alt="image-20211207153806186" style="zoom: 67%;" />

![image-20211207154516703](assess/image-20211207154516703.png)

![image-20211207154540095](assess/image-20211207154540095.png)

这种情况称为**Catastrophic Forgetting(灾难性遗忘**)

![image-20211207154856439](assess/image-20211207154856439.png)

<img src="assess/image-20211207155131244.png" alt="image-20211207155131244" style="zoom:80%;" />

![image-20211207155149261](assess/image-20211207155149261.png)

Evaluation
---

![image-20211207155414508](assess/image-20211207155414508.png)



![image-20211207160109221](assess/image-20211207160109221.png)

![image-20211207160317084](assess/image-20211207160317084.png) 



Selective Synaptic Plasticity
---

![image-20211208094655226](assess/image-20211208094655226.png)



![image-20211208095322832](assess/image-20211208095322832.png)

![image-20211208095631316](assess/image-20211208095631316.png)

$b_i$是人为设置的，不可以通过学习的方法得出，通过学习的方法使$L'(θ)$最小，机器会学出$b_i$=0.

![image-20211208100113605](assess/image-20211208100113605.png)

![image-20211208100148350](assess/image-20211208100148350.png)

![image-20211208100540809](assess/image-20211208100540809.png)

$b_i$如何设计的方法如下:

![image-20211208100731918](assess/image-20211208100731918.png)

学习任务的顺序也会影响到LLL的效果，因此评估方式一般为穷举所有任务顺序，做每一个顺序的实验，最后对实验结果取平均值。

### GEM(早年的方法)

![image-20211208101805736](assess/image-20211208101805736.png)

$g^b·g<0$则表示两个任务中$θ^b$参数梯度下降的方向不同,则需要修改$g到g'$

Additional Neural Resource Allocation
---

> 改变一下使用在每一个任务里面的Neural Resource

早期做法Progressive Neural Networks，每增加一个新任务就增加新的Network，不去改变前一个任务训练出来的模型，训练的时候只训练新增加的参数。缺点是每增加一个任务就需要新的空间，模型会变大，任务多的时候模型会太大导致无法保存。任务量不多是可以使用的。

![image-20211208102933391](assess/image-20211208102933391.png)

PackNet，CPG

> PackNet一开始就开一个大的Network，每增加一个任务就多使用一部分参数。CPG结合Progressive Neural Networks和PackNet

![image-20211208103121822](assess/image-20211208103121822.png)



![image-20211208103451419](assess/image-20211208103451419.png)



当任务的Class数量不同时，可以用以下论文中的方法。

![image-20211208103748986](assess/image-20211208103748986.png)

![image-20211208103946408](assess/image-20211208103946408.png)

Network Compression
===

> 模型压缩

Network Pruning 
---

> 网络修剪：对Network中的一些无关紧要的参数修剪掉

![image-20211208104943574](assess/image-20211208104943574.png)

**Weight pruning**

这种方法不好

![image-20211208105258004](assess/image-20211208105258004.png)

![image-20211208105343652](assess/image-20211208105343652.png)

![image-20211208105512911](assess/image-20211208105512911.png)

<img src="assess/image-20211208110315654.png" alt="image-20211208110315654" style="zoom: 80%;" />

<img src="assess/image-20211208110331956.png" alt="image-20211208110331956" style="zoom: 80%;" />

![image-20211208110355023](assess/image-20211208110355023.png)

<img src="assess/image-20211208111056601.png" alt="image-20211208111056601" style="zoom:80%;" />

![image-20211208111115669](assess/image-20211208111115669.png)



Knowledge Distillation
---

> 知识蒸馏：根据大的Network来制造小的Network

![image-20211208111910935](assess/image-20211208111910935.png)

![image-20211208111934772](assess/image-20211208111934772.png)

![image-20211208134430398](assess/image-20211208134430398.png)



Parameter Quantization
---

> 用比较少的存储空间存储参数

<img src="assess/image-20211208135247563.png" alt="image-20211208135247563" style="zoom:67%;" /><img src="assess/image-20211208135318319.png" alt="image-20211208135318319" style="zoom:67%;" />

![image-20211208135414595](assess/image-20211208135414595.png)



![image-20211208135533643](assess/image-20211208135533643.png)



Architecture Design
---

> 网络结构设计

![image-20211208140115861](assess/image-20211208140115861.png)



![image-20211208140211058](assess/image-20211208140211058.png)

![image-20211208141624923](assess/image-20211208141624923.png)

![image-20211208141902117](assess/image-20211208141902117.png)



![image-20211208142132773](assess/image-20211208142132773.png)

![image-20211208142522377](assess/image-20211208142522377.png)

![image-20211208142540193](assess/image-20211208142540193.png)







Dynamic Computation
---

> 动态调整运算量

 ![image-20211208143933422](assess/image-20211208143933422.png)

![image-20211208144048291](assess/image-20211208144048291.png)

![image-20211208144329485](assess/image-20211208144329485.png)

![image-20211208144407075](assess/image-20211208144407075.png)

Meta Learing
===

> Meta Learning: Learn to learn (学习如何学习)

![image-20211208144906610](assess/image-20211208144906610.png)

![image-20211208145047786](assess/image-20211208145047786.png)

![image-20211208145106185](assess/image-20211208145106185.png)

![image-20211208145419731](assess/image-20211208145419731.png)

![image-20211208145638796](assess/image-20211208145638796.png)

![image-20211208145836997](assess/image-20211208145836997.png)

![image-20211208145848556](assess/image-20211208145848556.png)

<img src="assess/image-20211208150331544.png" alt="image-20211208150331544" style="zoom:67%;" /><img src="assess/image-20211208150342174.png" alt="image-20211208150342174" style="zoom:67%;" />





![image-20211208150233715](assess/image-20211208150233715.png)

![image-20211208150441869](assess/image-20211208150441869.png)

![image-20211208150453236](assess/image-20211208150453236.png)

![image-20211208150550084](assess/image-20211208150550084.png)

![image-20211208150639351](assess/image-20211208150639351.png)

![image-20211208150914421](assess/image-20211208150914421.png)

![image-20211208151152988](assess/image-20211208151152988.png)

如果$L(φ)$不能进行梯度下降，可以采用强化学习的方法来实现。

![image-20211208151546685](assess/image-20211208151546685.png)

**ML v.s. Meta**
---

![image-20211208151949484](assess/image-20211208151949484.png)

![image-20211208152214947](assess/image-20211208152214947.png)

![image-20211208152336268](assess/image-20211208152336268.png)

![image-20211208152626844](assess/image-20211208152626844.png)

![image-20211208152749699](assess/image-20211208152749699.png)



![image-20211208153020272](assess/image-20211208153020272.png)

![image-20211208153525290](assess/image-20211208153525290.png)

Development task验证任务，类似于机器学习的验证集

在learning algorithm中哪些东西是可以被学习的
---

![image-20211208154227123](assess/image-20211208154227123.png)

![image-20211208154120330](assess/image-20211208154120330.png)

### 学习初始化的参数

![image-20211208154243998](assess/image-20211208154243998.png)

![image-20211208154356382](assess/image-20211208154356382.png)

![image-20211208154554095](assess/image-20211208154554095.png)

MAML用到了标注的资料，而Pre-train没用用标注的资料

![image-20211208154828599](assess/image-20211208154828599.png)

![image-20211208155329253](assess/image-20211208155329253.png)

MAML的每个任务网络架构都是一样的

![image-20211208155636684](assess/image-20211208155636684.png)



![image-20211208155808375](assess/image-20211208155808375.png)

![image-20211208155954461](assess/image-20211208155954461.png)

![image-20211208160013472](assess/image-20211208160013472.png)

![image-20211208160522423](assess/image-20211208160522423.png)

![image-20211208160819082](assess/image-20211208160819082.png)

![image-20211208160846971](assess/image-20211208160846971.png)

### Optimizer

![image-20211208160928066](assess/image-20211208160928066.png)



![image-20211208161134307](assess/image-20211208161134307.png)

### Network Structure

> 网络结构

![image-20211208161244690](assess/image-20211208161244690.png)

![image-20211208161442297](assess/image-20211208161442297.png)

![image-20211208161620955](assess/image-20211208161620955.png)

![image-20211208161715768](assess/image-20211208161715768.png)

![image-20211208161736632](assess/image-20211208161736632.png)

### Data Processing

![image-20211208161825539](assess/image-20211208161825539.png)



![image-20211208161838069](assess/image-20211208161838069.png)

### Sample Reweighting

> 样本权重

![image-20211208161954978](assess/image-20211208161954978.png)

![image-20211208162034982](assess/image-20211208162034982.png)

![image-20211208162132876](assess/image-20211208162132876.png)









Applications
---

> 应用

![image-20211208162309332](assess/image-20211208162309332.png)

![image-20211208163003442](assess/image-20211208163003442.png)

















序列模型
===

在时间$t$观察到$x_t$,那么得到$T$个不独立的随机変量$(x_1,x_2,...x_T)$~$p(X)$,使用条件概率展开$p(a, b)=p(a)p(b|a)=p(b)p(a|b)$

![image-20211129154531344](assess/image-20211129154531344.png)













RNN模型
===

RNN是一个序列模型

- 循环神经网络的输出取決于当下输入和前时间的隐变量
- 应用到语言模型中时，循环神经网络根据当前词预测下一次时刻词
- 通常使用困惑度来衡量语言模型的好坏

Elman Network

<img src="assess/image-20211129103937817.png" alt="image-20211129103937817" style="zoom:80%;" />

<img src="assess/image-20211129104031984.png" alt="image-20211129104031984" style="zoom:80%;" />

![image-20211206190734729](assess/image-20211206190734729.png)

![image-20211129104458812](assess/image-20211129104458812.png)

Bidirectional RNN(双向RNN)

![image-20211129104613863](assess/image-20211129104613863.png)

LSTM
---

> Long Short-term Memory(LSTM)

LSTM可以解决gradient vanishing(梯度消失)

![image-20211129105559012](assess/image-20211129105559012.png)

Input Gate:控制外接能不能把值存入Memory Cell

Output Gate:控制外接能不能把值从Memory Cell读出来

Forget Gate：什么时候将Memory Cell中的东西保留或忘记(清空)

`什么时候打开和关闭gate是神经忘络自己学到的`

四个输入：

存入Memory Cell中的值，操控三个门的信号

<img src="assess/image-20211129131046516.png" alt="image-20211129131046516" style="zoom:50%;" /><img src="assess/image-20211129131137541.png" alt="image-20211129131137541" style="zoom:50%;" />

激活函数通常使用sigmoid函数

![image-20211129131448948](assess/image-20211129131448948.png)



![image-20211129132357072](assess/image-20211129132357072.png)

`参数一般需要四倍的参数`

<img src="assess/image-20211129132958497.png" alt="image-20211129132958497" style="zoom:50%;" /><img src="assess/image-20211129132923132.png" alt="image-20211129132923132" style="zoom:50%;" />

Z是一个张量，维度为n(LSTM的个数)

![image-20211129133221259](assess/image-20211129133221259.png)

![image-20211129133404417](assess/image-20211129133404417.png)

RNN不好训练的原因

![image-20211129135327129](assess/image-20211129135327129.png)

![image-20211129135742122](assess/image-20211129135742122.png)



![image-20211129222630333](assess/image-20211129222630333.png)

![image-20211129222649778](assess/image-20211129222649778.png)





GRU
---

> Gate Recurrent Unit（GRU）门控循环单元
>
> 需要的参数量比LSTM少

![image-20211129212116372](assess/image-20211129212116372.png)



![image-20211129212725820](assess/image-20211129212725820.png)

![image-20211129212808015](assess/image-20211129212808015.png)



当$Z_t$接近0时则不考虑过去的状态，等价于RNN

![image-20211129212944645](assess/image-20211129212944645.png)





双向RNN
---

> 双向循环神经网络通过反向更新的隐藏层来利用方向时间信息，`通常用来对序列抽取特征、填空，而不是预测未来，不能用于预测模型`

![image-20211129223353112](assess/image-20211129223353112.png)

![image-20211129223406276](assess/image-20211129223406276.png)



两层是一组单元，每两组为一个单位



编码器-解码器
===

> 处理输入和输出都是长度可变的序列

*编码器*（encoder）：它接受一个`长度可变`的序列作为输入，并将其转换为`具有固定形状`的编码状态。

*解码器*（decoder）：它将`固定形状`的编码状态映射到`长度可变`的序列。

![../_images/encoder-decoder.svg](assess/encoder-decoder.svg)

编码器
---

```python
from torch import nn
#@save
class Encoder(nn.Module):
    """编码器-解码器结构的基本编码器接口。"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

解码器
---

```python
#@save
class Decoder(nn.Module):
    """编码器-解码器结构的基本解码器接口。"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
       """ init_state 函数用于将编码器的输出（enc_outputs）转换为编码后的状态。"""
      """enc_outputs：编码器所有的输出"""
     
        raise NotImplementedError

    def forward(self, X, state):
      """"""
        raise NotImplementedError
```

合并编码器和解码器
---

```python
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器结构的基类。"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

seq2seq
===

> 在机器翻译中使用两个循环神经网络进行序列到序列学习

![image-20211130102326964](assess/image-20211130102326964.png)

特定的“<eos>”表示序列结束词元。 一旦输出序列生成此词元，模型就可以停止执行预测。 在循环神经网络解码器的初始化时间步，有两个特定的设计决定。 首先，特定的**“<bos>”表示序列开始词元**，**它是解码器的输入序列的第一个词元**。 其次，使用`循环神经网络编码器最终的隐藏状态来初始化解码器的隐藏状态`。 例如，在 [[Sutskever et al., 2014\]](https://zh-v2.d2l.ai/chapter_references/zreferences.html#sutskever-vinyals-le-2014) 的设计中，正是基于这种设计将输入序列的编码信息送入到解码器中来生成输出序列的。 在其他一些设计中 [[Cho et al., 2014b\]](https://zh-v2.d2l.ai/chapter_references/zreferences.html#cho-van-merrienboer-gulcehre-ea-2014)，如 [图9.7.1](https://zh-v2.d2l.ai/chapter_recurrent-modern/seq2seq.html#fig-seq2seq) 所示，`编码器最终的隐藏状态在每一个时间步都作为解码器的输入序列的一部分`。

> 循环网络编码器

![image-20211130103218340](assess/image-20211130103218340.png)

```python
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器。"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        """嵌入层（embedding layer）来获得输入序列中每个词元的特征向量。
        嵌入层的权重是一个矩阵，其行数等于输入词表的大小（vocab_size），其列数等于特征向量的维度（embed_size）。
        对于任意输入词元的索引  i ，嵌入层获取权重矩阵的第  i  行（从  0  开始）以返回其特征向量。"""
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # `output`的形状: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state[0]`的形状: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

实例化一个两层门控循环单元编码器，其隐藏单元数为 16。给定一小批量的输入序列 `X`（批量大小为 4，时间步为 7）。在完成所有时间步后，最后一层的隐藏状态的输出是一个张量（`output` 由编码器的循环层返回），其形状为（时间步数, 批量大小, 隐藏单元数）

```python
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
#4是bachsize,7是句子长度
output, state = encoder(X)
output.shape
>>>torch.Size([7, 4, 16])
state.shape
>>>torch.Size([2, 4, 16])

```









Attention-based Model(自注意力模型)











Teacher forcing
---

RNN 存在两种训练模式（mode）：

- free-running mode: 上一个state的输出作为下一个state的输入

- teacher-forcing mode: 使用来自先验时间步长的输出作为输入

  ![img](assess/1630237-20210422182310408-901727114.png)

常见的训练RNN网络的方式是free-running mode，即`将上一个时间步的输出作为下一个时间步的输入`

teacher-forcing 在训练网络过程中，每次不使用上一个state的输出作为下一个state的输入，而是直接使用训练数据的标准答案(ground truth)的对应上一项作为下一个state的输入。

`Teacher Forcing工作原理`: 在训练过程的$t$时刻，使用训练数据集的期望输出或实际输出:$ y(t)$， 作为下一时间步骤的输入:$ x(t+1)$，而不是使用模型生成的输出$h(t)$

**一个例子：**训练这样一个模型，在给定序列中前一个单词的情况下生成序列中的下一个单词。

给定如下输入序列:

```html
Mary had a little lamb whose fleece was white as snow
```

首先，我们得给这个序列的首尾加上起止符号:

```html
[START] Mary had a little lamb whose fleece was white as snow [END]
```

对比两个训练过程：

| No.  | Free-running: X | Free-running: y^y^ | teacher-forcing: X             | teacher-forcing: y^y^ | teacher-forcing: Ground truth |
| ---- | --------------- | ------------------ | ------------------------------ | --------------------- | ----------------------------- |
| 1    | "[START]"       | "a"                | "[START]"                      | "a"                   | "Marry"                       |
| 2    | "[START]", "a"  | ?                  | "[START]", "Marry"             | ?                     | "had"                         |
| 3    | ...             | ...                | "[START]", "Marry", "had"      | ?                     | "a"                           |
| 4    |                 |                    | "[START]", "Marry", "had", "a" | ?                     | "little"                      |
| 5    |                 |                    | ...                            | ...                   | ...                           |
|      |                 |                    |                                |                       |                               |

free-running 下如果一开始生成"a"，之后作为输入来生成下一个单词，模型就偏离正轨。因为生成的错误结果，会导致后续的学习都受到不好的影响，导致学习速度变慢，模型也变得不稳定。

而使用teacher-forcing，模型生成一个"a"，可以在计算了error之后，丢弃这个输出，把"Marry"作为后续的输入。该模型将更正模型训练过程中的统计属性，更快地学会生成正确的序列。

**缺点：**teacher-forcing过于依赖ground truth数据，在训练过程中，模型会有较好的效果，但是在测试的时候因为不能得到ground truth的支持，所以如果目前生成的序列在训练过程中有很大不同，模型就会变得脆弱。`模型的cross-domain能力会更差，即如果测试数据集与训练数据集来自不同的领域，模型的performance就会变差。`

**teacher-forcing缺点的解决方法**

在预测单词这种离散值的输出时，一种常用方法是：对词表中每一个单词的预测概率执行搜索，生成多个候选的输出序列。

这个方法常用于机器翻译(MT)等问题，以优化翻译的输出序列。

beam search是完成此任务应用最广的方法，通过这种启发式搜索(heuristic search)，可减小模型学习阶段performance与测试阶段performance的差异。

![img](assess/1630237-20210422182333853-460589386.png)



curriculum learning
---

`Curriculum Learning`是Teacher Forcing的一个变种：一开始老师带着学，后面慢慢放手让学生自主学。

Curriculum Learning即有计划地学习：

- 使用一个概率$p$去选择使用ground truth的输出$y(t)$还是前一个时间步骤模型生成的输出$h(t)$作为当前时间步骤的输入$x(t+1)$。
- 这个概率$p$会随着时间的推移而改变，称为**计划抽样(scheduled sampling)**。
- 训练过程会从force learning开始，慢慢地降低在训练阶段输入ground truth的频率