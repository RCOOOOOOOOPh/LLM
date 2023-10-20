this is for papers that are not so important (which means other than Transformer, BERT, OpenAI papers and other important papers). For some (or most of the) papers, I just read others translation and thought in zhihu, csdn, etc.

略读paper记录，或者说就是看的别人的解读，为了尽快了解这个领域。

# T5 (Google) 2019.10

T5 means Text-to-Text Transfer Transformer 

实在是有点太长了，先搜了下BERT做对比，然后剩下内容直接看解析

这篇里搜索“GPT", "OpenAI"啥都搜不到。问题：为什么不对比一下？

https://zhuanlan.zhihu.com/p/88377084

all text-based language problems into a **text to text format**一个模型干所有NLP任务

提出了C4 corpus

模型：encoder-decoder，跟之前的transformer差别不大，改了下position encoding；encoder和decoder的规模分别都跟BERT差不多，所以参数量大约是BERT两倍

训练：greedy decoding

fine tune任务：4 tasks, machine translation, question answering, abstractive summarization, text classification

无监督目标函数：有点像BERT的

（问题：BERT/GPT不也可以吗？创新之处在哪）

文章里搜BERT的结果是：BERT produce a single prediction per input token or a single prediction for an entire input sequence. This makes them applicable for classification or span prediction tasks but not for generative tasks  

说是BERT只适合做分类这种，不适合做生成

**模型结构：**

![image-20231019020243511](D:\coursefile\LLM\typorapic\image-20231019020243511.png)

这个之后可以仔细看一下

比较**Unsupervised objectives**

![image-20231019020944629](D:\coursefile\LLM\typorapic\image-20231019020944629.png)

**training strategy:**

![image-20231019021021621](D:\coursefile\LLM\typorapic\image-20231019021021621.png)

看来这个的模型大小和数据还不够？

总之看这个知乎上的解析，没什么创新性的东西。尤其是现在GPT-3，GPT-4的时代了，甚至GPT1都提了zero-shot，这个模型本身应该没啥用了。就像那篇笔记里的人说的一样，OpenAI是一开始就想搞通用人工智能的，Google只是完成一点小任务。不过这篇里面的一些trick可以了解一下。

代码解读：

https://zhuanlan.zhihu.com/p/455216504

# Switch Transformers

https://zhuanlan.zhihu.com/p/362525526

https://zhuanlan.zhihu.com/p/344702054

https://zhuanlan.zhihu.com/p/351115630

**1\. Mixture of Experts（MoE）**仅选择模型中的一部分进行计算

由一个「门控网络」来选择咨询哪些专家。

1.6万亿参数，但是用了MoE所以**算起来还比较快**

![img](https://pic2.zhimg.com/v2-f359de43f252179e1bb88838820c4b9d_r.jpg)

将Transformer中的前馈全连接子层（Feed-Forward Network，FFN）视为Expert，使用多个FFN代替原来单一的FFN，并且使用了最简单的路由选择策略，将K设置为1，即不同的输入只会选择一个FFN进行计算（**sparse routing**)

（1）路由计算量减少，**只有一个expert激活**；

（2）expert中的batch_size（专家容量）至少减半；

（3）简化路由的实现，减少传统MOE方法中通信的代价。

**2\. 数据和权重划分：**

![image-20231019023911359](D:\coursefile\LLM\typorapic\image-20231019023911359.png)

**3\. 随机精度**

![image-20231019024634419](D:\coursefile\LLM\typorapic\image-20231019024634419.png)

问题：没看懂.......

4\. **No-Token-Left-Behind机制**

反复地把第一次路由的所有溢出的token重新进行路由

结论：

（1）Switch Transformer比MoE和Dense模型都要好；

（2）Switch Transformer在capacity比较小的时候效果更好；

（3）Switch Transformer比MoE要少一点计算量，如果让它们计算量相等的话，那么Switch Transformer还可以有更多提升(Switch-base-Expand)。

# On the Opportunities and Risks of Foundation Models

foundation models就是“大模型”

https://zhuanlan.zhihu.com/p/519403998

很长，没什么必要看原文吧。