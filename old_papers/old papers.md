# Transformer: Attention Is All You Need

## 1\. introduction

RNN problem: no parallelization



using CNN: Extended Neural GPU [16], ByteNet [18] and ConvS2S [9] 

deficiency: difficult to learn dependencies between distanc positions



**Attention**: modeling of dependencies **without regard to their distance** in the input or output sequences  

Parallel training

**Transformer**: 

only using self-attention, no RNN or CNN

## 2.implementation

encoder-decoder structure  

**encoder**: multi-head self-attention + fully connection, residual, layer norm

**decoder**: three sub-layers, one performs multi-head attention over the output of the encoder stack.

add mask, predictions for position i can depend only on the known outputs at positions less than i.  

**Scaled Dot-Product Attention  **:

previous works use additive attention(using a simple feed-forward layer), or simply dot product

this: scaling by $1/\sqrt{d_k}$ (suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. dk太大的时候点乘积也会很大，然后softmax的梯度就很小，所以要做scaling)

![image-20231017142312539](D:\coursefile\LLM\typorapic\image-20231017142312539.png)

**Multi-head attention**: 

linear project q, k and v

concatenate each head

Multi-head attention allows the model to jointly **attend to information from different representation subspaces at different positions**. With a single attention head, averaging inhibits this.  

![image-20231017143107300](D:\coursefile\LLM\typorapic\image-20231017143107300.png)

application: encoder-decoder: Q, V from encoder output and K from decoder output

encoder self-attention

decoder: self-attention + mask

feed forward: 2 linear with relu, or 1*1 convolution

"we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation "**why?**

**Positional encoding**: to make use of the order of the sequence  

for any fixed offset k, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$

![image-20231017220617433](D:\coursefile\LLM\typorapic\image-20231017220617433.png)

![image-20231017220630843](D:\coursefile\LLM\typorapic\image-20231017220630843.png)

###  3.

why self-attention: 

**complexity, sequential operations, maximum path length**:

![image-20231017175400645](D:\coursefile\LLM\typorapic\image-20231017175400645.png)

self-attention could yield more interpretable models.   

## 4\. training and results

standard WMT 2014 English-German dataset   

encoded using byte-pair encoding  

**model variations**:

number of heads: single-head and too many heads both get worse

中文解读：https://zhouyifan.net/2022/11/12/20220925-Transformer/

note：在Transformer中，解码器的嵌入层和输出线性层是共享权重的——输出线性层表示的线性变换是嵌入层的逆变换，其目的是把网络输出的嵌入再转换回one-hot向量。如果某任务的输入和输出是同一种语言，那么编码器的嵌入层和解码器的嵌入层也可以共享权重。

论文中写道：“输入输出的嵌入层和softmax前的线性层共享权重”。这个描述不够清楚。如果输入和输出的不是同一种语言，比如输入中文输出英文，那么共享一个词嵌入是没有意义的。



问题：为什么是embedding变换成KQV然后再做attention？可以直接拿embedding做点乘吗

# GPT1.0

https://zhuanlan.zhihu.com/p/412351920

(GPT1.0 is earlier than BERT!!!)

use transformer decoder

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  

https://shangzhi-huang.gitbook.io/workspace/nlp-zhi-yu-yan-mo-xing/lun-wen-bert-pretraining-of-deep-bidirectional-transformers-for-language-understanding

## 1.background

ELMo(RNN based), GPT1.0(one direction)



## 2.implementation

bidirectional

unified architecture across different tasks. minimal difference between the pre-trained architecture and the final downstream architecture.  

pre-training: unlabeled data



**MLM: masked language model**

一个关于BERT和GPT的解读

https://blog.csdn.net/JamesX666/article/details/124596388

![image-20231018191228174](D:\coursefile\LLM\typorapic\image-20231018191228174.png)

![image-20231018191310236](D:\coursefile\LLM\typorapic\image-20231018191310236.png)



