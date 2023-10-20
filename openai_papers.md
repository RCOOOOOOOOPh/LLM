# Scaling Laws for Neural Language Models

https://zhuanlan.zhihu.com/p/620479884

https://blog.csdn.net/qq_52852138/article/details/131697352

一个对LLM的理论研究，结论就是大力出奇迹........大模型和大量数据才能得到更好的效果。这篇开门见山把所有结论先列出来了，后面才讲的实验，这种风格不错

## 核心结论

- 模型表现和规模强相关，和模型的shape弱相关：规模包括模型参数量N（不包括embedding）、数据集大小D和计算量C，模型shape指模型depth、width、number of self-attention heads
- 幂方法则：对于模型参数量N、数据集大小D和计算量C三个因素，如果其他两个充足的前提下，模型表现和第三个因素成**幂方关系**（指数下降，类似$e^{-x}$）。实验曲线如下，可以看出D的影响最大

![image-20231019030303641](D:\coursefile\LLM\typorapic\image-20231019030303641.png)

- 过拟合：当同时增加数据量和模型参数量时，模型表现会一直变好。当其中一个因素受限时，模型表现随另外一个因素增加变好，但是会逐渐衰减。下图表示数据量不足时，模型很快出现过拟合导致在测试集上效果很快衰减。数据和模型参数量的比例关系大致为$N^{0.74}/D$ ，也就是**模型参数增大8倍，数据也需要增大5倍**才能发挥模型参数的全部潜力。

Universality of training：在模型参数量不变的情况下，模型的表现是可以预测的。通过对早期的训练曲线进行推断，就能粗略估计训练更长时间后模型的表现

Transfer improves with text performance：当在分布不同的文本上评估模型时，结果与在验证集上的结果密切相关，损失的偏移量大致恒定。这说明用验证集的结果来作为评估指标是合理的

Sample efficiency：大模型能在更少的step内，更少的数据（图4）上达到相同的性能

Convergence is inefficient：当计算量固定时，但是模型大小和数据量没有限制时，**大模型在得到最佳性能时，还远远没有收敛**。最大训练效率训练比训练小模型到收敛是更 sample efficient的，数据需求随着计算量增长比较慢 D ∼ C 0.27 D \sim C^{0.27}D∼C 
0.27


Optimal batch size: 最好的batch size与loss有 power-law 关系，也受到梯度噪声规模的影响

# Codex: Evaluating Large Language Models Trained on Code  

https://blog.csdn.net/qq_32275289/article/details/124438494

https://blog.csdn.net/qq_36936443/article/details/125452458?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-125452458-blog-124438494.235%5Ev38%5Epc_relevant_anti_t3_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-125452458-blog-124438494.235%5Ev38%5Epc_relevant_anti_t3_base&utm_relevant_index=2

基于GPT的模型架构，在GItHub上微调，可以用来编写Python代码

感觉相比GPT的主线工作，像是个做着玩的？毕竟现在的GPT3和GPT4也可以写代码。

细节：

目标函数没有使用BLEU（困惑度），因为代码不同于自然语言，即使特别相似，但仍然可能不是一个合法的语句，作者使用**pass@k**来评估模型，即生成n个输出（n>k），从中随机抽取k个输出，输出通过单元测试的概率

![image-20231019032958867](D:\coursefile\LLM\typorapic\image-20231019032958867.png)

对输出做softmax得到概率之前，会除以一个超参数Temperature，来调节不同输出之间的概率差距，当pass@k中的采样数k越大时，T越大效果越好

模型局限性

1\. 样本有效性不够，需要训练很多的代码，模型才能输出比较简单的实验

2\. Prompt应该怎么写才能获得比较理想的代码，作者找了13 basic building block（对字符串做一些简单的操作：如改变大小写、变换位置等），将文档块任意串起来，发现文档越长，生成代码的质量越差，说明docstring不宜过长（这段可以仔细读一下）

