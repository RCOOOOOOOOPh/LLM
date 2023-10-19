# Language Models are Few-Shot Learners  

Abstract

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art finetuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

摘要
最近的研究已经表明，通过在大规模文本语料库上进行预训练，然后在特定任务上进行微调，可以在许多自然语言处理（NLP）任务和基准测试中获得显著的性能提升。虽然通常在体系结构上是任务无关的，但这种方法仍然需要成千上万个示例的特定任务微调数据集。相比之下，人类通常可以仅凭几个示例或简单的说明执行新的语言任务 - 目前，当前的NLP系统仍然在这方面存在一定的困难。在这里，**我们展示了扩大语言模型规模极大地改善了任务无关的少样本性能，有时甚至可以达到之前的最先进的微调方法的竞争力**。具体来说，我们训练了一个具有1750亿参数的自回归语言模型GPT-3，比以前的非稀疏语言模型多了10倍，并测试了它在少样本设置中的性能。对于所有任务，GPT-3都没有进行任何梯度更新或微调，任务和少样本演示纯粹通过与模型的文本交互来指定。GPT-3在许多NLP数据集上都取得了强大的性能，包括翻译、问答和填空任务，以及一些需要即时推理或领域适应的任务，如词语解谜、在句子中使用新词，或进行三位数的算术运算。与此同时，我们还发现了一些数据集，GPT-3的少样本学习仍然存在困难，以及一些数据集，GPT-3面临与在大型网络语料库上训练相关的方法论问题。最后，我们发现GPT-3能够生成新闻文章样本，使人类评估员难以区分是否为人类撰写的文章。我们讨论了这一发现以及GPT-3的更广泛社会影响。

# 1 Introduction

Recent years have featured a trend towards pre-trained language representations in NLP systems, applied in increasingly flexible and task-agnostic ways for downstream transfer. First, single-layer representations were learned using word vectors [MCCD13, PSM14] and fed to task-specific architectures, then RNNs with multiple layers of representations and contextual state were used to form stronger representations [DL15, MBXS17, PNZtY18] (though still applied to task-specific architectures), and more recently pre-trained recurrent or transformer language models [VSP+17] have been directly fine-tuned, entirely removing the need for task-specific architectures [RNSS18, DCLT18, HR18].

近年来，自然语言处理系统中出现了一种趋势，即采用越来越**灵活和任务无关的方式应用预训练的语言表示进行下游传输**。首先，使用词向量[MCCD13，PSM14]学习了单层表示，并将其提供给特定任务的体系结构，然后使用具有多层表示和上下文状态的RNN来形成更强的表示[D15，MBXS17，PNZtY18]（尽管仍然应用于特定任务的体系结构），最近更多的是使用预训练的循环或变换语言模型[VSP+17]进行直接微调，完全消除了对特定任务体系结构的需求[RNSS18，DCLT18，HR18]。

This last paradigm has led to substantial progress on many challenging NLP tasks such as reading comprehension, question answering, textual entailment, and many others, and has continued to advance based on new architectures and algorithms [RSR+19, LOG+19, YDY+19, LCG+19]. However, a major limitation to this approach is that while the architecture is task-agnostic, there is still a need for task-specific datasets and task-specific fine-tuning: to achieve strong performance on a desired task typically requires fine-tuning on a dataset of thousands to hundreds of thousands of examples specific to that task. Removing this limitation would be desirable, for several reasons.

这最后的范式已经在许多具有挑战性的自然语言处理任务上取得了显著进展，例如阅读理解、问题回答、文本蕴涵等，而且还在基于新体系结构和算法的基础上不断发展[RSP+19，LOG+19，YDY+19，LCG+19]。然而，这种方法的一个主要限制是，虽然体系结构是任务无关的，但**仍然需要特定任务的数据集和任务特定的微调**：要在所需任务上获得强大的性能通常需要在针对该任务的**数千到数十万个示例的数据集上进行微调**。消除这一限制将是可取之处，出于几个原因。

First, from a practical perspective, the need for a large dataset of labeled examples for every new task limits the applicability of language models. There exists a very wide range of possible useful language tasks, encompassing anything from correcting grammar, to generating examples of an abstract concept, to critiquing a short story. For many of these tasks it is difficult to collect a large supervised training dataset, especially when the process must be repeated for every new task.

首先，从实际角度来看，对于每个新任务都需要大量带标签的示例数据集，这限制了语言模型的适用性。存在着广泛的可能有用的语言任务，涵盖了从纠正语法错误到生成抽象概念示例再到评论短篇故事等各种任务。对于许多这些任务，**很难收集到大规模的监督训练数据集**，特别是当这个过程必须**针对每个新任务重复进行**时。

Second, the potential to exploit spurious correlations in training data fundamentally grows with the expressiveness of the model and the narrowness of the training distribution. This can create problems for the pre-training plus fine-tuning paradigm, where models are designed to be large to absorb information during pre-training, but are then fine-tuned on very narrow task distributions. For instance [HLW+20] observe that larger models do not necessarily generalize better out-of-distribution. There is evidence that suggests that the generalization achieved under this paradigm can be poor because the model is overly specific to the training distribution and does not generalize well outside it [YdC+19, MPL19]. Thus, the performance of fine-tuned models on specific benchmarks, even when it is nominally at human-level, may exaggerate actual performance on the underlying task [GSL+18, NK19].

第二，利用训练数据中的偶然关联的潜力基本上与模型的表达能力和训练分布的狭窄性增长。这可能会为预训练加微调范式带来问题，因为这些模型在预训练期间被设计得很大，以吸收信息，但随后在非常狭窄的任务分布上进行微调。例如，[HLW+20]观察到，更大的模型未必在分布之外更好地进行泛化。有证据表明，在这种范式下实现的泛化能力可能较差，因为模型对训练分布过于具体，无法很好地在其之外进行泛化[YdC+19，MPL19]。因此，即使微调后的模型在特定基准测试上的性能在名义上达到了人类水平，也可能夸大了在基础任务上的实际性能[GSL+18，NK19]。

Third, humans do not require large supervised datasets to learn most language tasks – a brief directive in natural language (e.g. “please tell me if this sentence describes something happy or something sad”) or at most a tiny number of demonstrations (e.g. “here are two examples of people acting brave; please give a third example of bravery”) is often sufficient to enable a human to perform a new task to at least a reasonable degree of competence. Aside from pointing to a conceptual limitation in our current NLP techniques, this adaptability has practical advantages – it allows humans to seamlessly mix together or switch between many tasks and skills, for example performing addition during a lengthy dialogue. To be broadly useful, we would someday like our NLP systems to have this same fluidity and generality.

第三，**人类在学习大多数语言任务时不需要大规模的监督数据集** - 通常**只需要自然语言中的简短指示**（例如：“请告诉我这个句子是否描述了开心的事情还是伤心的事情”），或者**最多只需要一小部分示范**（例如：“这里有两个人表现出勇敢的例子；请给出勇敢的第三个例子”），**通常足以使人类能够以至少合理的程度完成新任务**。除了指出了我们当前NLP技术的概念限制之外，这种适应能力还具有实际优势 - 它使人类能够在许多任务和技能之间无缝混合或切换，例如在漫长的对话中进行加法运算。为了广泛应用，我们希望有一天我们的NLP系统具备这种流动性和通用性。

One potential route towards addressing these issues is meta-learning1 – which in the context of language models means the model develops a broad set of skills and pattern recognition abilities at training time, and then uses those abilities at inference time to rapidly adapt to or recognize the desired task (illustrated in Figure 1.1). Recent work [RWC+19] attempts to do this via what we call “in-context learning”, using the text input of a pretrained language model as a form of task specification: the model is conditioned on a natural language instruction and/or a few demonstrations of the task and is then expected to complete further instances of the task simply by predicting what comes next  

解决这些问题的一个潜在途径是**元学习**（meta-learning）[注1] - 在语言模型的背景下，这意味着模型在训练时发展了**广泛的技能和模式识别能力**，然后在推理时使用这些能力来快速适应或识别所需的任务（如图1.1所示）。最近的研究[RWC+19]尝试通过我们称之为“**上下文学习”的方式来实现这一点**，利用预训练语言模型的文本输入作为任务规范的一种形式：**模型受到自然语言指令和/或任务的少数示范的约束，然后被期望通过预测接下来会发生什么来完成任务的更多实例**。

While it has shown some initial promise, this approach still achieves results far inferior to fine-tuning – for example [RWC+19] achieves only 4% on Natural Questions, and even its 55 F1 CoQa result is now more than 35 points behind the state of the art. Meta-learning clearly requires substantial improvement in order to be viable as a practical method of solving language tasks.  

尽管这种方法已经显示出了一些初步的潜力，但它的结果**仍远不及微调** - 例如，[RWC+19] 在自然问题任务中仅达到了4%的性能，甚至它的CoQa任务中的55 F1分数现在已经落后于最先进方法超过35个百分点。显然，元学习需要在可行的实际语言任务解决方法方面取得实质性的改进。

Another recent trend in language modeling may offer a way forward. In recent years the capacity of transformer language models has increased substantially, from 100 million parameters [RNSS18], to 300 million parameters [DCLT18], to 1.5 billion parameters [RWC+19], to 8 billion parameters [SPP+19], 11 billion parameters [RSR+19], and finally 17 billion parameters [Tur20]. Each increase has brought improvements in text synthesis and/or downstream NLP tasks, and there is evidence suggesting that log loss, which correlates well with many downstream tasks, follows a smooth trend of improvement with scale [KMH+20]. Since **in-context learning involves absorbing many skills and tasks within the parameters of the model,** it is plausible that in-context learning abilities might show similarly strong gains with scale.  

语言建模的另一个最近的趋势可能提供了前进的途径。近年来，transformer语言模型的容量大幅增加，从1亿参数[RNSS18]，到3亿参数[DCLT18]，再到15亿参数[RWC+19]，8亿参数[SPP+19]，110亿参数[RSR+19]，最终到了170亿参数[Tur20]。每次增加都带来了文本合成和/或下游自然语言处理任务的改进，有证据表明，与许多下游任务相关的对数损失（log loss）随着规模的增加呈现出平滑的改善趋势[KMH+20]。由于**上下文学习涉及吸收模型参数内的许多技能和任务**，因此可以合理推测，在规模扩大的情况下，上下文学习的能力可能会表现出类似的强大增益。

In this paper, we test this hypothesis by training a 175 billion parameter autoregressive language model, which we call GPT-3, and measuring its in-context learning abilities. Specifically, we evaluate GPT-3 on over two dozen NLP datasets, as well as several novel tasks designed to test rapid adaptation to tasks unlikely to be directly contained in the training set. For each task, we evaluate GPT-3 under 3 conditions: (a) “**few-shot learning**”, or in-context learning where we allow as many demonstrations as will fit into the model’s context window (typically 10 to 100), (b) “**one-shot learning**”, where we allow only one demonstration, and (c) “**zero-shot**” learning, where no demonstrations are allowed and only an instruction in natural language is given to the model. GPT-3 could also in principle be evaluated in the traditional fine-tuning setting, but we leave this to future work.  

在本文中，我们通过训练一个具有1750亿参数的自回归语言模型，我们称之为GPT-3，来测试这一假设，并测量它的上下文学习能力。具体来说，我们在超过两打的NLP数据集上评估了GPT-3，以及一些旨在测试对不太可能直接包含在训练集中的任务进行快速适应的新任务。对于每个任务，我们在三种条件下评估GPT-3：(a) "**少样本学习**"，或者在上下文学习中，我们允许尽可能多的示范，以适应模型的上下文窗口（通常为10到100个示范）；(b) "**一次样本学习**"，我们只允许一个示范；(c) "**零样本学习**"，不允许示范，只给模型提供自然语言的指令。原则上，GPT-3也可以在传统的微调设置中进行评估，但我们留待未来研究。

Figure 1.2 illustrates the conditions we study, and shows few-shot learning of a simple task requiring the model to remove extraneous symbols from a word. **Model performance improves with the addition of a natural language task description**, and with the number of examples in the model’s context, K. Few-shot learning also improves dramatically with model size. Though the results in this case are particularly striking, the general trends with both model size and number of examples in-context hold for most tasks we study. We emphasize that these “learning” curves involve no gradient updates or fine-tuning, just increasing numbers of demonstrations given as conditioning.

图1.2展示了我们研究的条件，显示了一个简单任务的少样本学习，该任务要求模型从一个单词中去除多余的符号。模型的性能随着自然语言任务描述的添加以及模型上下文中示例的数量K而改善。**随着模型规模的增加，少样本学习也有了显著的改善**。尽管这种情况下的结果尤其引人注目，但我们研究的大多数任务都表现出了与模型规模和上下文中示例数量的一般趋势。我们强调，这些**"学习"曲线不涉及任何梯度更新或微调，只是作为条件给出的示范数量的增加。**

![image-20231013184306242](D:\coursefile\LLM\typorapic\image-20231013184306242.png)

Broadly, on NLP tasks GPT-3 achieves promising results in the zero-shot and one-shot settings, and in the the few-shot setting is sometimes competitive with or even occasionally surpasses state-of-the-art (despite state-of-the-art being held by fine-tuned models). For example, GPT-3 achieves 81.5 F1 on CoQA in the zero-shot setting, 84.0 F1 on CoQA in the one-shot setting, 85.0 F1 in the few-shot setting. Similarly, GPT-3 achieves 64.3% accuracy on TriviaQA in the zero-shot setting, 68.0% in the one-shot setting, and 71.2% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.  

总的来说，在自然语言处理任务中，GPT-3在零样本和一次样本的设置下取得了令人鼓舞的结果，在少样本的设置下，**有时甚至能与甚至偶尔超越最先进的模型（尽管最先进的模型是经过微调的**）。例如，GPT-3在CoQA任务的零样本设置下取得了81.5的F1分数，在一次样本设置下取得了84.0的F1分数，在少样本设置下取得了85.0的F1分数。类似地，GPT-3在TriviaQA任务的零样本设置下取得了64.3%的准确率，在一次样本设置下取得了68.0%的准确率，在少样本设置下取得了71.2%的准确率，相对于在相同封闭式设置下操作的经过微调的模型来说，这是最先进的。

GPT-3 also displays one-shot and few-shot proficiency at tasks designed to test rapid adaption or on-the-fly reasoning, which include unscrambling words, performing arithmetic, and using novel words in a sentence after seeing them defined only once. We also show that in the few-shot setting, GPT-3 can generate synthetic news articles which human evaluators have difficulty distinguishing from human-generated articles.

此外，GPT-3还在旨在测试快速适应或即时推理的任务中表现出了一次样本和少样本的熟练水平，这些任务包括对单词进行解密、进行算术运算以及在只看到一次定义后在句子中使用新词。我们还展示，在少样本设置下，GPT-3能够生成合成的新闻文章，使人类评估员难以区分是否为人类生成的文章。

At the same time, we also find some tasks on which few-shot performance struggles, even at the scale of GPT-3. This includes natural language inference tasks like the ANLI dataset, and some reading comprehension datasets like RACE or QuAC. By presenting a broad characterization of GPT-3’s strengths and weaknesses, including these limitations, we hope to stimulate study of few-shot learning in language models and draw attention to where progress is most needed.

A heuristic sense of the overall results can be seen in Figure 1.3, which aggregates the various tasks (though it should not be seen as a rigorous or meaningful benchmark in itself).  

与此同时，我们还发现一些任务在少样本性能方面表现不佳，即使在GPT-3的规模下也是如此。这包括像ANLI数据集这样的自然语言推理任务，以及一些阅读理解数据集，如RACE或QuAC。通过对GPT-3的优势和劣势进行广泛的刻画，包括这些限制，我们希望激发对语言模型中少样本学习的研究，并引起对进展最需要的领域的关注。

总体结果的启发式感觉可以在图1.3中看到，该图汇总了各种任务（尽管它本身不应被视为严格或有意义的基准）。

![image-20231013184828319](D:\coursefile\LLM\typorapic\image-20231013184828319.png)

图1.3：所有42个以准确度为标准的基准测试的综合表现。零样本性能随着模型规模的增加稳步提高，而少样本性能增长更迅猛，表明更大的模型在上下文学习方面更为熟练。有关标准NLP基准套件SuperGLUE的更详细分析，请参见图3.8

We also undertake a systematic study of "data contamination -- a growing problem when training high capacity model.on datasets such as Common Crawl, which can potentially include content from test datasets simply because succontent often exists on the web. In this paper we develop systematic tools to measure data contamination and quantifits distorting effects. Although we find that data contamination has a minimal effect on GPT-3's performance on mostdatasets, we do identify a few datasets where it could be inflating results, and we either do not report results on thesedatasets or we note them with an asterisk, depending on the severity.

我们还进行了一项系统性的研究，即“**数据污染**” - 这是在训练高容量模型时变得越来越常见的问题，尤其是在像Common Crawl这样的数据集上，这些数据集可能包含了测试数据集中的内容，因为这些内容通常存在于网络上。在本文中，我们开发了系统性工具来测量数据污染并量化其扭曲效果。尽管我们发现数据污染对GPT-3在大多数数据集上的性能影响微乎其微，但我们确实识别出一些数据集可能会夸大结果，对于这些数据集，我们要么不报告结果，要么在其上加上一个星号，具体取决于严重程度。

In addition to all the above, we also train a series of smaller models (ranging from 125 million parameters to 13 billionparameters) in order to compare their performance to GPT-3 in the zero, one and few-shot settings. Broadly, for mosttasks we find relatively smooth scaling with model capacity in all three settings; one notable pattern is that the gabetween zero-, one-, and few-shot performance often grows with model capacity, perhaps suggesting that larger modelare more proficient meta learners

除了上述内容，我们还训练了一系列较小的模型（从1.25亿参数到130亿参数不等），以便将它们在零样本、一次样本和少样本设置下的性能与GPT-3进行比较。总的来说，对于大多数任务，我们发现在这三种设置下，模型容量的增加会带来相对平稳的性能提升；一个显著的模式是零样本、一次样本和少样本性能之间的差距通常会随着模型容量的增加而扩大，这也许表明更大的模型更擅长元学习。

Finally, given the broad spectrum of capabilities displayed by GPT-3, we discuss concerns about bias, fairness, andbroader societal impacts, and attempt a preliminary analysis of GPT-3's characteristics in this regard

最后，鉴于GPT-3展示出的广泛能力，我们讨论了与偏见、公平性和更广泛的社会影响有关的问题，并尝试初步分析了GPT-3在这方面的特点。The remainder of this paper is organized as follows. In Section 2, we describe our approach and methods for trainingGPT-3 and evaluating it. Section 3 presents results on the full range of tasks in the zero-, one- and few-shot setingsSection 4 addresses questions of data contamination (train-test overlap ). Section 5 discusses limitations of GPT-3Section 6 discusses broader impacts. Section 7 reviews related work and Section 8 concludes.

本文的其余部分组织如下。在第2节中，我们描述了训练GPT-3和评估它的方法和方法。第3节介绍了零样本、一次样本和少样本设置下的所有任务的结果。第4节涉及数据污染（训练-测试重叠）的问题。第5节讨论了GPT-3的局限性。第6节讨论了更广泛的影响。第7节回顾了相关工作，第8节进行总结。

# 2 Approach

Our basic pre-training approach, including model, data, and training, is similar to the process described in (RWC+19with relatively straightforward scaling up of the model size, dataset size and diversity, and length of training. Our ustof in-context learning is also similar to (RWC+191, but in this work we systematically explore different settings foilearning within the context. Therefore, we start this section by explicitly defining and contrasting the different settingsthat we will be evaluating GPT-3 on or could in principle evaluate GPT-3 on. These settings can be seen as lying on aspectrum of how much task-specific data they tend to rely on. Specifically, we can identify at least four points on thisspectrum (see Figure 2.1 for an illustration):

我们的基本预训练方法，包括模型、数据和训练，类似于(RWC+19)中描述的过程，只是我们对模型规模、数据集规模和多样性以及训练时长进行了相对简单的扩展。我们在上下文学习的定制方面也类似于(RWC+19)，但在这项工作中，我们系统性地探讨了不同的上下文学习设置。因此，我们通过明确定义和对比我们将在GPT-3上评估或原则上可以评估GPT-3的不同设置来开始本节。这些设置可以看作是在它们倾向于依赖多少特定任务数据方面位于一个光谱上。具体来说，我们可以确定这个光谱上至少有四个点（参见图2.1进行说明）：

**• Fine-Tuning (FT)** has been the most common approach in recent years, and involves updating the weights ofa pre-trained model by training on a supervised dataset specific to the desired task. Typically thousands tohundreds of thousands of labeled examples are used. The main advantage of fine-tuning is strong performanceon many benchmarks. The main disadvantages are the need for a new large dataset for every task, the potentialfor poor generalization out-of-distribution (MPL19 , and the potential to exploit spurious features of thetraining data (GSL+18, NK19], potentially resulting in an unfair comparison with human performance. Inthis work we do not fine-tune GPT-3 because our focus is on task-agnostic performance, but GPT-3 can befine-tuned in principle and this is a promising direction for future work.

近年来，微调（Fine-Tuning，FT）已成为最常见的方法，它涉及通过在特定于所需任务的监督数据集上进行训练来更新预训练模型的权重。通常使用数千到数十万个标记的示例。微调的主要优点是在许多基准测试上表现出色。主要缺点包括需要为**每个任务准备一个新的大型数据集、在分布之外泛化可能较差**（MPL19），以及有可能利用训练数据的虚假特征（GSL+18, NK19），这可能导致**与人类表现的不公平比较**。在这项工作中，我们不对GPT-3进行微调，因为我们的重点是任务无关的性能，但原则上可以对GPT-3进行微调，这是未来研究的一个有前途的方向。

**• Few-Shot (FS)** is the term we will use in this work to refer to the setting where the model is given a fewdemonstrations of the task at inference time as conditioning (RWC+191, but no weight updates are allowed.As shown in Figure 2.1, for a typical dataset an example has a context and a desired completion (for examplean English sentence and the French translation), and few-shot works by giving K examples of context anccompletion, and then one final example of context, with the model expected to provide the completion. Wetypically set K in the range of 10 to 100 as this is how many examples can fit in the model's context window(nctx = 2048). The main advantages of few-shot are a major reduction in the need for task-specific data andreduced potential to learn an overly narrow distribution from a large but narrow fine-tuning dataset. The maindisadvantage is that results from this method have so far been much worse than state-of-the-art fine-tunecmodels. Also, a small amount of task specific data is still required, As indicated by the name, few-shotlearning as described here for language models is related to few-shot learning as used in other contexts inML (HYCO1, VBL+16) - both involve learning based on a broad distribution of tasks (in this case implicit inthe pre-training data) and then rapidly adapting to a new task.

在这项工作中，我们将使用Few-Shot（FS）这个术语来指代模型在推理时以一些任务示例作为条件（RWC+19）的设置，但不允许权重更新。如图2.1所示，对于典型的数据集，一个示例具有上下文和所需的完成部分（例如，英语句子和法语翻译），Few-Shot的工作方式是提供上下文和完成部分的K个示例，然后提供上下文的最后一个示例，模型需要提供完成部分。通常，我们将K设置在10到100的范围内，因为这是模型上下文窗口（nctx = 2048）中能容纳的示例数量。Few-Shot的主要优点是**大幅减少了对任务特定数据的需求，并减少了从大型但狭窄的微调数据集中学习过于狭窄分布的潜力**。主要缺点是到目前为止，此方法的**结果远远不如最先进的微调模型**。此外，仍然需要一小部分任务特定数据。正如其名称所示，这里描述的语言模型中的Few-Shot学习与ML中其他上下文中使用的Few-Shot学习（HYCO1、VBL+16）相关联，两者都涉及基于广泛的任务分布（在这种情况下隐含在预训练数据中）进行学习，然后迅速适应新任务。

**• One-Shot (1S)** is the same as few-shot except that only one demonstration is allowed, in addition to a natural language description of the task, as shown in Figure 1. The reason to distinguish one-shot from few-shot and zero-shot (below) is that it most closely matches the way in which some tasks are communicated to humans. For example, when asking humans to generate a dataset on a human worker service (for example Mechanical Turk), it is common to give one demonstration of the task. By contrast it is sometimes difficult to communicate the content or format of a task if no examples are given  

**• 一次样本 (1S)** 与少样本相同，只是允许一个示范，同时提供了一个任务的自然语言描述，如图1所示。之所以区分一次样本与少样本和零样本（下文将介绍）是因为它最接近某些任务与人类沟通的方式。例如，当要求人类在人力工作服务（例如Mechanical Turk）上生成数据集时，通常会提供一个任务示范。相比之下，如果不提供示例，有时很难传达任务的内容或格式。

Zero-Shot (0S) is the same as one-shot except that no demonstrations are allowed, and the model is only givena natural language instruction describing the task. This method provides maximum convenience. potential forrobustness, and avoidance of spurious correlations (unless they occur very broadly across the large corpus oipre-training data), but is also the most challenging setting. In some cases it may even be difhcult for humansto understand the format of the task without prior examples, so this setting is in some cases "unfairly hard'For example. if someone is asked to "make a table of world records for the 200m dash”, this request can beambiguous. as it may not be clear exactly what format the table should have or what should be included (andeven with careful clarifcation. understanding precisely what is desired can be diffcult). Nevertheless, for ateast some settings zero-shot is closest to how humans perform tasks - for example, in the translation examplein Figure 2.1. a human would likely know what to do from ust the text instruction.

**• 零样本 (0S)** 与一次样本相同，只是不允许示范，模型只能得到一个描述任务的自然语言指令。这种方法提供了最大的便利性、鲁棒性的潜力以及避免虚假相关性（除非它们在大规模的预训练数据中广泛发生），但也是最具挑战性的设置。在某些情况下，**甚至可能难以让人类理解任务的格式，如果没有事先的示例，这种设置在某些情况下可能是“不公平的难”**。例如，如果有人要求“制作一张世界纪录的200米短跑表”，这个请求可能会含糊不清，因为可能不清楚表格应该具有什么格式或应该包含什么内容（即使进行了仔细澄清，理解需要什么可能也很困难）。尽管如此，至少在某些情况下，**零样本设置是最接近人类执行任务的方式**，例如，在图2.1的翻译示例中，人类很可能能够仅从文本指令中知道如何执行任务。

Figure 2.1 shows the four methods using the example of translating English to French. In this paper we focus onzero-shot, one-shot and few-shot, with the aim of comparing them not as competing alternatives, but as differentproblem settings which offer a varying trade-off between performance on specifc benchmarks and sample effciencyWe especially highlight the few-shot results as many of them are only slightly behind state-of-the-art fine-tuned modelsUltimately, however, one-shot, or even sometimes zero-shot, seem like the fairest comparisons to human performanceand are important targets for future work
Sections 2.1-2.3 below give details on our models, training data, and training process respectively. Section 2.4 discussesthe details of how we do few-shot. one-shot, and zero-shot evaluations

图2.1以将英语翻译成法语的示例展示了这四种方法。在本文中，我们专注于零样本、一次样本和少样本，旨在将它们视为不同的问题设置，它们在特定基准测试性能和样本效率之间提供了不同的权衡，而不是竞争性的替代方案。我们特别强调了少样本的结果，因为其中许多结果只稍微落后于最先进的微调模型。然而，最终，一次样本，甚至有时零样本，似乎是与人类表现最公平的比较，也是未来工作的重要目标。

接下来的2.1-2.3节将详细介绍我们的模型、训练数据和训练过程。第2.4节讨论了我们如何进行少样本、一次样本和零样本评估的细节。

## 2.1 Model and Architectures

We use the same model and architecture as GPT-2 RWC+ 191. including the modifed initialization. pre-normalizationand reversible tokenization described therein. with the exception that we use alternating dense and locally banded sparseattention patterns in the layers of the transformer, similar to the Sparse Transformer (CGRS19. To study the dependenceof ML performance on model size, we train 8 different sizes of model, ranging over three orders of magnitude from 125million parameters to 175 billion parameters, with the last being the model we call GPT-3. Previous work (KMH+20suggests that with enough training data, scaling of validation loss should be approximately a smooth power law as aunction of size: training models of many different sizes allows us to test this hypothesis both for validation loss and fordownstream language tasks
Table 2.1 shows the sizes and architectures of our 8 models. Here mparams is the total number of trainable parametersnlayers is the total number of layers, dmodel is the number of units in each bottleneck layer (we always have thefeedforward layer four times the size of the bottleneck layer, df = 4 * dmodel), and dhead is the dimension of eachattention head. All models use a context window of nctx = 2048 tokens. We partition the model across GPUs alongooth the depth and width dimension in order to minimize data-transfer between nodes. The precise architecturaparameters for each model are chosen based on computational efficiency and load-balancing in the layout of modelsacross GPU's. Previous work (KMH+201 suggests that validation loss is not strongly sensitive to these parameterswithin a reasonably broad range.

我们使用与GPT-2（RWC+19）相同的模型和架构，包括其中描述的修改的初始化、预标准化和可逆记号化，但与之不同的是，我们在Transformer的层中使用了交替的密集和局部带状稀疏注意模式，类似于Sparse Transformer（CGRS19）。为了研究机器学习性能与模型规模的依赖关系，我们训练了8种不同规模的模型，范围从125百万参数到**1750亿参数，其中最后一个模型被称为GPT-3**。以前的研究（KMH+20）表明，使用足够的训练数据，验证损失的缩放应该大致是尺寸的平滑幂律函数：训练许多不同尺寸的模型使我们能够测试这一假设，无论是对于验证损失还是对于下游语言任务。

表2.1显示了我们8个模型的大小和架构。这里mparams是可训练参数的总数，nlayers是总层数，dmodel是瓶颈层中的单元数（我们总是将前馈层的大小设为瓶颈层的四倍，df = 4 * dmodel），dhead是每个注意头的维度。所有模型都使用nctx = 2048个标记的上下文窗口。我们在GPU上沿着深度和宽度维度对模型进行分区，以最小化节点之间的数据传输。每个模型的精确架构参数是基于计算效率和负载均衡在GPU布局中选择的。以前的研究（KMH+20）表明，在合理范围内，验证损失对这些参数不太敏感。

![image-20231013190407912](D:\coursefile\Hao\Language Models are Few-Shot Learners\image-20231013190407912.png)

## 2\.2 Training Dataset

Datasets for language models have rapidly expanded, culminating in the Common Crawl dataset(RSR+ 191 constitutingnearly a trillion words. This size of dataset is sufficient to train our largest models without ever updating on the samesequence twice. However. we have found that unfiltered or lightly filtered versions of Common Crawl tend to haveower quality than more curated datasets. Therefore, we took 3 steps to improve the average quality of our datasets(1) we downloaded and filtered a version of CommonCrawl based on similarity to a range of high-quality referencecorpora, (2) we performed fuzzy deduplication at the document level, within and across datasets, to prevent redundancyand preserve the integrity of our held-out validation set as an accurate measure of overfitting, and (3) we also added<nown high-quality reference corpora to the training mix to augment CommonCrawl and increase its diversity.

语言模型的数据集迅速扩展，最终形成了Common Crawl数据集（RSR+19），包含了近1万亿个单词。这个规模的数据集足以训练我们最大的模型，而且不会在同一序列上多次更新。然而，我们发现未经过滤或轻度过滤的Common Crawl版本往往质量较低，不如更精心策划的数据集。因此，我们采取了3个步骤来提高我们数据集的平均质量：(1) 基于与一系列高质量参考语料库的相似性，下载并**过滤了一个版本的Common Crawl，**(2) 在文档级别内和跨数据集进行**模糊去重**，以防止冗余并保持我们的保留验证集的完整性，作为过拟合的准确度测量，(3) 我们还在训练集中添加了**已知的高质量参考语料库**，以增加Common Crawl的多样性。

Details of the first two points (processing of Common Crawl) are described in Appendix A. For the third, we addedseveral curated high-quality datasets, including an expanded version of the WebText dataset RWC+191. collectedby scraping links over a longer period of time, and first described in (KMH+201, two internet-based books corporaBooks1 and Books2) and English-language Wikipedia.

关于前两点（Common Crawl的处理）的详细信息在附录A中有描述。对于第三点，我们添加了几个经过策划的高质量数据集，包括WebText数据集的扩展版本（RWC+19），通过在较长一段时间内抓取链接收集而来，并在（KMH+20）中首次描述，以及两个基于互联网的书籍语料库Books1和Books2以及英语维基百科。

Table 2.2 shows the final mixture of datasets that we used in training. The CommonCrawl data was downloaded from41 shards of monthly CommonCrawl covering 2016 to 2019, constituting 45TB of compressed plaintext before filteringand 570GB after filtering, roughly equivalent to 400 billion byte-pair-encoded tokens. Note that during training, datasetsare not sampled in proportion to their size, but rather datasets we view as higher-quality are sampled more frequentlysuch that CommonCrawl and Books2 datasets are sampled less than once during training, but the other datasets aresampled 2-3 times. This essentially accepts a small amount of overfitting in exchange for higher quality training data.

表2.2显示了我们在训练中使用的数据集的最终混合。CommonCrawl数据来自覆盖2016年至2019年的41个月CommonCrawl分片，经过过滤前压缩的纯文本大小为45TB，过滤后为570GB，大致相当于4000亿个字节对编码的标记。请注意，在训练过程中，**数据集的采样不是按其大小比例采样的，而是我们认为质量更高的数据集会更频繁地被采样**，因此在训练期间，CommonCrawl和Books2数据集的采样次数较少，而其他数据集的采样次数为2-3次。这实际上**接受了一定程度的过拟合，以换取更高质量的训练数据**。

A major methodological concern with language models pretrained on a broad swath of internet data, particularly large models with the capacity to memorize vast amounts of content, is potential contamination of downstream tasks by having their test or development sets inadvertently seen during pre-training. To reduce such contamination, we searched for and attempted to remove any overlaps with the development and test sets of all benchmarks studied in this paper. Unfortunately, a bug in the filtering caused us to ignore some overlaps, and due to the cost of training it was not feasible to retrain the model. In Section 4 we characterize the impact of the remaining overlaps, and in future work we will more aggressively remove data contamination.  

语言模型在广泛的互联网数据上进行预训练的一个重要方法论问题，尤其是容量足以记住大量内容的大型模型，可能会导致下游任务受到其开发或测试集在预训练期间被无意中看到的潜在**污染**。为了减少这种污染，我们搜索并**尝试删除与本文研究的所有基准测试的开发和测试集的重叠部分**。不幸的是，在筛选过程中出现了一些重叠的问题，由于重新训练模型的成本问题，我们无法重新训练模型。在第4节中，我们将对剩余重叠的影响进行描述，在未来的工作中，我们将更积极地去除数据污染。

## 2.3 Training Process

As found in [KMH+20, MKAT18], larger models can typically use a larger batch size, but require a smaller learning rate. We measure the gradient noise scale during training and use it to guide our choice of batch size [MKAT18]. Table 2.1 shows the parameter settings we used. To train the larger models without running out of memory, we use a mixture of model parallelism within each matrix multiply and model parallelism across the layers of the network. All models were trained on V100 GPU’s on part of a high-bandwidth cluster provided by Microsoft. Details of the training process and hyperparameter settings are described in Appendix B.

正如[KMH+20, MKAT18]中所发现的，更大的模型通常可以使用**更大的批量大小，但需要更小的学习速率**。我们在训练期间测量梯度噪声规模，并用它来指导我们选择批量大小[MKAT18]。表2.1显示了我们使用的参数设置。为了在训练更大的模型时不耗尽内存，我们在每个矩阵相乘中使用模型并行性以及网络层之间的模型并行性的混合。所有模型都是在由Microsoft提供的高带宽集群的一部分上使用V100 GPU进行训练的。有关训练过程和超参数设置的详细信息请参阅附录B。

## 2.4 Evaluation

For few-shot learning, we evaluate each example in the evaluation set by randomly drawing K examples from thattask's training set as conditioning, delimited by 1 or 2 newlines depending on the task. For LAMBADA and Storyclozehere is no supervised training set available so we draw conditioning examples from the development set and evaluateon the test set. For Winograd (the original, not SuperGLUE version) there is only one dataset, so we draw conditioningexamples directly from it.

对于few-shot学习，我们通过从**该任务的训练集中随机选择K个示例作为条件来评估评估集中的每个示例**，示例之间使用1个或2个换行符分隔，具体取决于任务。对于LAMBADA和Storycloze，没有可用的监督训练集，所以我们从开发集中获取条件示例并在测试集上进行评估。对于Winograd（原始版本，而不是SuperGLUE版本），只有一个数据集，所以我们直接从中获取条件示例。

K can be any value from 0 to the maximum amount allowed by the model's context window, which is nctx = 2048for all models and typically fits 10 to 100 examples. Larger values of K are usually but not always better, so when aseparate development and test set are available, we experiment with a few values of K on the development set and thenrun the best value on the test set. For some tasks (see Appendix G) we also use a natural language prompt in addition to(or for K - 0. instead of) demonstrations.

K可以是0到模型的上下文窗口允许的最大值之间的任何值，对于所有模型来说，上下文窗口的大小为nctx = 2048，通常可以容纳10到100个示例。较大的K值通常但并不总是更好，因此在有独立的开发和测试集时，我们会在开发集上尝试一些K值，然后在测试集上运行最佳值。对于一些任务（请参阅附录G），除了（或替代K-0）示例外，我们还使用自然语言提示。

On tasks that involve choosing one correct completion from several options (multiple choice), we provide K examplesof context plus correct completion, followed by one example of context only, and compare the L M likelihood oieach completion. For most tasks we compare the per-token likelihood (to normalize for length), however on a smallnumber of datasets (ARC, OpenBookA, and RACE) we gain additional benefit as measured on the development setP(completion context)by normalizing by the unconditional probability of each completion, by computing P(completion|context)/P(completion|answer_context), where" or "A: " and is used to prompt that the completion should be an answeranswer_context is the string "Answer :but is otherwise generic.
On tasks that involve binary classification, we give the options more semantically meaningful names (e.g. “True orFalse rather than 0 or 1) and then treat the task like multiple choice: we also sometimes frame the task similar to whatis done by [RSR+19] (see Appendix G) for details.

在涉及从多个选项中选择一个正确答案的任务（多项选择题）上，我们提供K个示例的上下文和正确完成，然后提供一个示例的上下文，然后比较每个完成的语言模型似然性。对于大多数任务，我们比较每个标记的似然性（以进行长度标准化），然而在一小部分数据集（ARC、OpenBookA和RACE）上，我们通过计算P（完成|上下文）/P（完成|答案上下文）来将结果标准化，从开发集上进行测量。其中“或“A：“和”用于提示完成应该是一个答案，而“answer_context”是“Answer：”但除此之外是通用的字符串。

对于涉及二元分类的任务，我们会给选项更具语义的名称（例如“真或假”，而不是0或1），然后将任务视为多项选择题：我们有时也会将任务设置成与[RSR+19]（请参阅附录G）中所做的方式类似。有关详细信息。

On tasks with free-form completion, we use beam search with the same parameters as (RSR+19): a beam width of 4and a length penalty of a = 0.6. We score the model using F1 similarity score, BLEU. or exact match, depending on what is standard for the dataset at hand.
what is standard for the dataset at hand.Final results are reported on the test set when publicly available, for each model size and learning setting (zero-, one-and few-shot). When the test set is private. our model is often too large to fit on the test server. so we report results onthe development set, We do submit to the test server on a small number of datasets (SuperGLUE, TriviaOA, PiOa)where we were able to make submission work. and we submit only the 200B few-shot results. and report developmentset results for everything else.

在自由形式完成的任务中，我们使用与 (RSR+19) 相同的参数进行束搜索，即束宽度为 4，长度惩罚为 a = 0.6。我们根据数据集的标准使用 F1 相似性分数、BLEU 或精确匹配来评估模型性能。

最终结果通常在公开可用的测试集上报告，针对不同的模型大小和学习设置（零次、一次和少次学习）。当测试集是私有的时候，我们的模型通常太大，无法适应测试服务器，因此我们会在开发集上报告结果。在一小部分数据集上，我们会尝试将结果提交到测试服务器（如 SuperGLUE、TriviaOA、PiOa），并且仅提交 200B 少次学习的结果，对于其他数据集，我们会报告在开发集上的结果。

# 3\. Results

In Figure 3.1 we display training curves for the 8 models described in Section 2. For this graph we also include 6additional extra-small models with as few as 100.000 parameters. As observed in (KMH+201, language modelingperformance follows a power-law when making efficient use of training compute. After extending this trend by twomore orders of magnitude, we observe only a slight (if any) departure from the power-law. One might worry that theseimprovements in cross-entropy loss come only from modeling spurious details of our training corpus. However, we wilsee in the following sections that improvements in cross-entropy loss lead to consistent performance gains across abroad spectrum of natural language tasks.

在图3.1中，我们展示了第2节中描述的8个模型的训练曲线。为此图，我们还包括了6个额外的超小模型，其中最少有100,000个参数。正如在（KMH+201）中观察到的，当高效利用训练计算时，语言建模性能遵循幂律。在将这一趋势延伸了两个数量级后，我们观察到离幂律只有轻微（如果有的话）偏离。有人可能担心，这些交叉熵损失的改进仅来自对训练语料库的非关键细节进行建模。然而，我们将在以下各节中看到，交叉熵损失的改进导致了在广泛的自然语言任务中一致的性能提升。

Below, we evaluate the 8 models described in Section 2 (the 175 billion parameter parameter GPT-3 and 7 smalleimodels) on a wide range of datasets. We group the datasets into 9 categories representing roughly similar tasks.

In Section 3.1 we evaluate on traditional language modeling tasks and tasks that are similar to language modelingsuch as Cloze tasks and sentence/paragraph completion tasks. In Section 3.2 we evaluate on "closed book" questioranswering tasks: tasks which require using the information stored in the model's parameters to answer generalknowledge questions. In Section 3.3 we evaluate the model's ability to translate between languages (especially one-shotand few-shot). In Section 3.4 we evaluate the model's performance on Winograd Schema-like tasks. In Section 3.5 weevaluate on datasets that involve commonsense reasoning or guestion answering, In Section 3.6 we evaluate on readingcomprehension tasks, in Section 3.7 we evaluate on the SuperGLUE benchmark suite, and in 3.8 we briefly exploreNL1. Finally, in Section 3.9. we invent some additional tasks designed especially to probe in-context learning abilities -these tasks focus on on-the-fly reasoning, adaptation skills, or open-ended text synthesis. We evaluate all tasks in thefew-shot.one-shot,and zero-shot settings.

在下文中，我们评估第2节中描述的8个模型（包括1750亿参数的GPT-3和7个较小的模型）在各种数据集上的性能。我们将这些数据集分为9个类别，代表大致相似的任务。在第3.1节中，**我们对传统的语言建模任务和类似语言建模的任务进行评估**，如Cloze任务和句子/段落完成任务。在第3.2节中，**我们评估“闭卷”问答任务，这些任务要求使用模型参数中存储的信息来回答常识问题**。在第3.3节中，我们评估模型在语言翻译（尤其是一次学习和少次学习）方面的能力。在第3.4节中，我们评估模型在类似Winograd Schema的任务上的性能。在第3.5节中，我们评估涉及常识推理或问题回答的数据集。在第3.6节中，我们评估阅读理解任务，第3.7节中，我们评估SuperGLUE基准套件，在3.8节中，我们简要探讨NL1。最后，在第3.9节中，我们设计了一些特别用于探究上下文学习能力的附加任务，这些任务侧重于即兴推理、适应能力或开放式文本合成。我们在少次学习、一次学习和零次学习的设置下评估所有任务。

## 3.1Language Modeling, Cloze, and Completion Tasks

In this section we test GPT-3's performance on the traditional task of language modeling, as well as related tasksthat involve predicting a single word of interest, completing a sentence or paragraph, or choosing between possiblecompletions of a piece of text.

在这一部分，我们测试GPT-3在传统的语言建模任务上的性能，以及涉及预测感兴趣的单词、完成句子或段落，或在可能的文本完成之间进行选择的相关任务。

### 3.1.1 Language Modeling

We calculate zero-shot perplexity on the Penn Tree Bank (PTB) (MKM+94] dataset measured in RWC+19). We omitthe 4 Wikipedia-related tasks in that work because they are entirely contained in our training data, and we also omit theone-billion word benchmark due to a high fraction of the dataset being contained in our training set, PTB escapes thestissues due to predating the modern internet. Our largest model sets a new SOTA on PTB by a substantial margin of 15points. achieving a perplexity of 20.50. Note that since PTB is a traditional language modeling dataset it does not havea clear separation of examples to define one-shot or few-shot evaluation around, so we measure only zero-shot

我们在Penn Tree Bank (PTB) 数据集上计算zero-shot 困惑度，该数据集由RWC+19测得。我们在那项研究中省略了与维基百科相关的4项任务，因为它们完全包含在我们的训练数据中，而且由于我们的训练集中包含了数据集的大部分内容，我们也省略了十亿字的基准测试。PTB由于它的年代较早，不受现代互联网的影响。我们的最大模型在PTB上取得了令人瞩目的15个点的新记录，将困惑度降低到20.50。请注意，由于PTB是传统的语言建模数据集，它没有明确的示例分离，无法定义一个围绕零-shot的一次或少次评估，因此我们只测量零-shot。

### 3.1.2 LAMBADA

The LAMBADA dataset (PKL+16] tests the modeling of long-range dependencies in text -- the model is asked topredict the last word of sentences which require reading a paragraph of context. It has recently been suggested that thecontinued scaling of language models is yielding diminishing returns on this difficult benchmark, (BHT+201 reflect onthe small 1.5% improvement achieved by a doubling of model size between two recent state of the art results (ISPP+19and [Tur20]) and argue that “continuing to expand hardware and data sizes by orders of magnitude is not the path forward”. We find that path is still promising and in a zero-shot setting GPT-3 achieves 76% on LAMBADA, a gain of 8% over the previous state of the art.  

LAMBADA数据集（PKL+16）测试文本中的长距离依赖性建模——模型被要求预测需要阅读段落上下文的句子的最后一个单词。最近有人提出，继续扩大语言模型的规模在这一困难的基准测试上的回报逐渐减小，(BHT+201 反映了在两个最近的最新结果（ISPP+19和[Tur20]）之间模型规模翻倍仅带来1.5%的改进，并认为“继续扩大硬件和数据规模数个数量级不是前进的道路”。然而，我们发现这条道路仍然很有前途，GPT-3在零-shot设置下在LAMBADA上取得了76%的成绩，超过了以往最先进的技术水平，提升了8%。

LAMBADA is also a demonstration of the flexibility of few-shot learning as it provides a way to address a problem thatclassically occurs with this dataset, Although the completion in LAMBADA is always the last word in a sentence, astandard language model has no way of knowing this detail. It thus assigns probability not only to the correct ending butalso to other valid continuations of the paragraph. This problem has been partially addressed in the past with stop-wordfilters (RWC+191 (which ban continuation words). The few-shot setting instead allows us to “frame the task as acloze-test and allows the language model to infer from examples that a completion of exactly one word is desired. Weuse the following fill-in-the-blank format.

LAMBADA还展示了few-shot学习的灵活性，因为它提供了解决这一数据集中经典问题的方法。尽管LAMBADA中的完成总是句子的最后一个单词，标准语言模型却无法知道这个细节。因此，它不仅为正确的结尾分配概率，还为段落的其他有效延续分配概率。过去，这个问题已经部分得到了解决，通过使用停用词过滤器（RWC+191）（它禁止了延续词）。而few-shot设置允许我们“将任务构建成一个填空测试，并让语言模型从示例中推断出确切需要一个单词的完成。我们使用以下的填空格式。

Alice was friends with Bob. Alice went to visit her friend -> Bob
George bought some baseball equipment, a ball, a glove, and a. ->

When presented with examples formatted this way, GPT-3 achieves 86.4% accuracy in the few-shot setting, an increaseof over 18% from the previous state-of-the-art, We observe that few-shot performance improves strongly with modelsize. While this setting decreases the performance of the smallest model by almost 20%, for GPT-3 it improves accuracyby 10%, Finally, the fil-in-blank method is not effective one-shot, where it always performs worse than the zero-shotsetting. Perhaps this is because all models still require several examples to recognize the pattern.

当以这种方式呈现示例时，GPT-3在few-shot设置下的准确率达到了86.4％，较之前的最先进水平提高了18％以上。我们观察到few-shot性能随模型规模的增加而显著提高。虽然这种设置几乎将最小的模型的性能降低了20％，但对于GPT-3，它提高了10％的准确性。最后，填空法在一次尝试中并不有效，它在零尝试设置下表现始终不如。可能这是因为所有模型仍然需要多个示例来识别模式。

One note of caution is that an analysis of test set contamination identified that a significant minority of the LAMBADAlataset appears to be present in our training data - however analysis performed in Section 4 suggests negligible impacon performance.

值得注意的是，对测试集污染的分析发现，LAMBADA数据集的相当一部分似乎存在于我们的训练数据中，但是第4节中进行的分析表明对性能的影响可以忽略不计。

### 3.1.3 HellaSwag

The HellaSwag dataset (ZHB+19 involves picking the best ending to a story or set of instructions. The examples wereadversarially mined to be difficult for language models while remaining easy for humans (who achieve 95.6% accuracyGPT-3 achieves 78.1% accuracy in the one-shot setting and 79.3% accuracy in the few-shot seting, outperforming the75.4% accuracy of a fine-tuned 1.5 parameter language model (ZHR+191 but still a fair amount lower than the overalSOTA of 85.6% achieved by the fine-tuned multi-task model ALUM

HellaSwag数据集（ZHB+19涉及选择最佳的故事或一组说明的结局。这些示例经过对抗性挖掘，对语言模型而言难度很大，但对人类来说很容易（人类的准确率达到了95.6%）。GPT-3在单次尝试设置下的准确率为78.1%，在少次尝试设置下的准确率为79.3%，超过了细调的1.5亿参数语言模型的75.4%准确率（ZHR+19），但仍然比多任务细调模型ALUM的整体SOTA准确率低，后者达到了85.6%。

### 3.1.4 StoryCloze

We next evaluate GPT-3 on the StoryCloze 2016 dataset (MCH+161. which involves selecting the correct endingsentence for five-sentence long stories. Here GPT-3 achieves 83.2% in the zero-shot setting and 87.7% in the few-shotsetting (with K = 70). This is still 4.1% lower than the fine-tuned SOTA using a BERT based model (LDL19) but improves over previous zero-shot results by roughly 10%.

接下来，我们对GPT-3在StoryCloze 2016数据集（MCH+16）上进行评估，该数据集涉及选择五句长故事的正确结局句。在零次尝试设置下，GPT-3的准确率为83.2%，在少次尝试设置下（K = 70）为87.7%。这仍然比使用基于BERT的细调模型（LDL19）的SOTA低4.1%，但相对于以前的零次尝试结果有了约10%的提高。

## 3\.2  Closed Book Ouestion Answering

in this section we measure GPT-3's ability to answer ouestions about broad factual knowledge. Due to the immenseamount of possible queries, this task has normally been approached by using an information retrieval system to findrelevant text in combination with a model which learns to generate an answer given the question and the retrievecext. Since this setting allows a system to search for and condition on text which potentially contains the answer itis denoted “open-book".(RRS20 recently demonstrated that a large language model can perform surprisingly weldirectly answering the ouestions without conditioning on auxlliary information They denote this more restrictiveevaluation setting as "closed-book”. Their work suggests that even higher-capacity models could perform even betterand we test this hypothesis with GPT-3. We evaluate GPT-3 on the 3 datasets in (RRS201: Natural Ouestions (KPR+19WebOuestions (BCFL131 and Trivia0A (JCWZ17l using the same splits. Note that in addition to all results being inthe closed-book setting, our use of few-shot, one-shot, and zero-shot evaluations represent an even stricter setting thanprevious closed-book OA work: in addition to external content not being allowed, fine-tuning on the 0&A dataset itselis also not permitted.

在这一部分，我们测量GPT-3回答广泛的事实知识问题的能力。由于可能的查询数量巨大，通常会使用信息检索系统来查找相关文本，结合一个模型，该模型学会在**给定问题和检索文本的情况下生成答案**。由于这种设置允许系统搜索并在可能包含答案的文本上进行调节，因此它被称为“**开卷**”。（RRS20最近证明，一个大型语言模型可以出奇地**直接回答问题而不依赖辅助信息的调节**。他们将这种更严格的评估设置称为“**闭卷**”。他们的工作表明，即使容量更大的模型也可以表现得更好，我们用GPT-3来测试这个假设。我们使用相同的拆分来评估GPT-3在（RRS20中的3个数据集上：自然问题（KPR+19），Web问题（BCFL13）和Trivia0A（JCWZ17）。请注意，除了所有结果都在闭卷设置中，我们使用零次尝试、一次尝试和零次尝试的评估代表了比以前的闭书OA工作更严格的设置：除了不允许使用外部内容外，也不允许在0&A数据集本身上进行细调。

The results for GPT-3 are shown in Table 3.3. On TriviaA, we achieve 64.3% in the zero-shot setting, 68.0% in theone-shot setting, and 71.2% in the few-shot setting. The zero-shot result already outperforms the fine-tuned T5-11B b14.2%, and also outperforms a version with 0&A tailored span prediction during pre-training by 3.8%, The one-shotresult improves by 3.7% and matches the SOTA for an open-domain OA svstem which not only fine-tunes but alscmakes use of a learned retrieval mechanism over a 15.3B parameter dense vector index of 21M documents (LPP+20GPT-3's few-shot result further improves performance another 3.2% beyond this.

GPT-3的结果如表3.3所示。在TriviaA上，我们在零次尝试设置中获得64.3%，在一次尝试设置中获得68.0%，在少数尝试设置中获得71.2%。零次尝试的结果已经超过了经过微调的T5-11B模型14.2%，也超过了在预训练期间使用0&A定制的跨度预测的版本3.8%。一次尝试的结果提高了3.7%，与开放领域OA系统的SOTA相匹配，该系统不仅进行微调，还利用了学到的检索机制，涵盖了2100万文档的15.3B参数密集矢量索引（LPP+20）。GPT-3的少数尝试结果进一步提高了3.2%。

On WebOuestions (WebOs), GPT-3 achieves 14.4% in the zero-shot setting, 25.3% in the one-shot setting, and 41.5%in the few-shot setting. This compares to 37.4% for fie-tuned T5-11B, and 44.7% for fine-tuned T5-11B+SSMwhich uses a 0&A-specific pre-training procedure. GPT-3 in the few-shot setting approaches the performance ofstate-of-the-art fine-tuned models Notably, compared to Trivia0A, WebOS shows a much larger gain from zero-shot tofew-shot (and indeed its zero-shot and one-shot performance are poor), perhaps suggesting that the WebOs questionsand/or the style of their answers are out-of-distribution for GPT-3. Nevertheless, GPT-3 appears able to adapt to thisdistribution, recovering strong performance in the few-shot setting.

在WebOuestions（WebOs）上，GPT-3在零次尝试设置中获得14.4%，在一次尝试设置中获得25.3%，在少数尝试设置中获得41.5%。与经过微调的T5-11B的37.4%相比，以及使用0&A特定的预训练过程的经过微调的T5-11B+SSM的44.7%。GPT-3在少数尝试设置中接近了最先进的经过微调模型的性能。值得注意的是，与Trivia0A相比，WebOS在零次尝试到少数尝试之间有更大的提高（事实上，它的零次尝试和一次尝试表现不佳），这也许表明WebOs的问题和/或答案风格对于GPT-3而言属于分布之外。然而，GPT-3似乎能够适应这种分布，从而在少数尝试设置中表现出色。

On Natural Ouestions (NOs) GPT-3 achieves 14.6% in the zero-shot setting, 23.0% in the one-shot setting, and 29.9% inthe few-shot setting, compared to 36.6% for fine-tuned T5 11B+SSM. Similar to WebOS, the large gain from zero-shotto few-shot may suggest a distribution shift, and may also explain the less competitive performance compared toTriviaOA and WebOs. In particular, the questions in NOs tend towards very fine-grained knowledge on Wikipediaspecifically which could be testing the limits of GPT-3's capacity and broad pretraining distribution.Overall, on one of the three datasets GPT-3's one-shot matches the open-domain fine-tuning SOTA. On the other twodatasets it approaches the performance of the closed-book SOTA despite not using fine-tuning. On all 3 datasets, weind that performance scales very moothly with model size (Figure 3.3 and Appendix H Figure H.7), possibly reflectingthe idea that model capacity translates directly to more "knowledge' absorbed in the parameters of the model

在自然问题（NOs）上，GPT-3在零次尝试设置中获得14.6%，在一次尝试设置中获得23.0%，在少数尝试设置中获得29.9%，而经过微调的T5 11B+SSM为36.6%。与WebOS类似，从零次尝试到少数尝试的大幅提高可能表明分布的变化，这也可能解释了相对于TriviaOA和WebOs而言表现较不竞争。特别是，在NOs中的问题往往涉及维基百科上非常细粒度的知识，这可能正在测试GPT-3的容量和广泛的预训练分布的极限。总的来说，在这三个数据集中，GPT-3的一次尝试与开放领域的微调SOTA相匹配。在其他两个数据集中，尽管没有使用微调，它也接近了封闭书SOTA的性能。在所有三个数据集上，我们发现性能与模型大小呈非常平稳的关系，这可能反映了模型容量直接转化为模型参数中更多“知识”的想法。

## 3.3 Translation

For GPT-2 a filter was used on a multilingual collection of documents to produce an English only dataset due to capacityconcerns. Even with this filtering GPT-2 showed some evidence of multilingual capability and performed non-triviallwhen translating between French and English despite only training on 10 megabytes of remaining French text. Since weincrease the capacity by over two orders of magnitude from GPT-2 to GPT-3, we also expand the scope of the trainingdataset to include more representation of other languages, though this remains an area for further improvement. Asliscussed in 2.2 the maiority of our data is derived from raw Common Crawl with only ouality-based fltering. AlthoughGPT-3's training data is still primarily English (93% by word count), it also includes 7% of text in other languagesThese languages are documented in the supplemental material. In order to better understand translation capability, wealso expand our analysis to include two additional commonly studied languages, German and Romanian.

对于GPT-2，由于容量问题，对多语言文档集合进行了过滤，以生成仅包含英语的数据集。尽管进行了此过滤，GPT-2在法语和英语之间的翻译表现出一些多语言能力的迹象，尽管只在剩下的10兆字节的法语文本上进行了训练。由于我们将容量从GPT-2增加了两个数量级以上，因此我们还扩大了培训数据集的范围，以包括更多的其他语言的代表性，尽管这仍然需要进一步改进。如2.2所讨论，我们的大部分数据源自原始Common Crawl，仅经过基于质量的过滤。尽管GPT-3的训练数据仍然主要是英语（按字数计算占93%），但还包括其他语言的7%的文本。这些语言在补充材料中有文档记录。为了更好地理解翻译能力，我们还扩大了分析范围，包括其他两种常用研究的语言，德语和罗马尼亚语。

Existing unsupervised machine translation approaches often combine pretraining on a pair of monolingual datasetswith back-translation (SHB151 to bridge the two languages in a controlled way. By contrast, GPT-3 learns from ablend of training data that mixes many languages together in a natural way. combining them on a word, sentenceand document level. GPT-3 also uses a single training obiective which is not customized or designed for any task inparticular. However, our one / few-shot settings aren't strictly comparable to prior unsupervised work since they makeuse of a small amount of paired examples (1 or 64). This corresponds to up to a page or two of in-context training dataResults are shown in Table 34. Zero-shot GPT-3. which only receives on a natural language description of the taskstill underperforms recent unsupervised NMT results. However, providing only a single example demonstration for each translation task improves performance by over 7 BLEU and nears competitive performance with prior workGPT-3 in the full few-shot setting further improves another 4 BLEU resulting in similar average performance to priorunsupervised NMT work. GPT-3 has a noticeable skew in its performance depending on language direction. For thethree input languages studied, GPT-3 significantly outperforms prior unsupervised NMT work when translating intoEnglish but underperforms when translating in the other direction. Performance on En-Ro is a noticeable outlier atover 10 BLEU worse than prior unsupervised NMT work, This could be a weakness due to reusing the byte-level BPEtokenizer of GPT-2 which was developed for an almost entirely English training dataset, For both Fr-En and De-Enfew shot GPT-3 outperforms the best supervised result we could find but due to our unfamiliarity with the literature anothe appearance that these are un-competitive benchmarks we do not suspect those results represent true state of the artFor Ro-En, few shot GPT-3 performs within 0.5 BLEU of the overall SOTA which is achieved by a combination ofunsupervised pretraining, supervised finetuning on 608K labeled examples, and backtranslation (LHCG19b1.

现有的无监督机器翻译方法通常将在一对单语数据集上进行的预训练与反向翻译相结合，以控制方式桥接两种语言。相比之下，GPT-3从自然方式混合多种语言的训练数据中学习，将它们在词、句子和文档级别进行组合。GPT-3还使用了一个单一的训练目标，没有为特定任务定制或设计。然而，我们的一/ few-shot设置与以前的无监督工作不是严格可比的，因为它们使用了少量成对示例（1或64）。这对应于最多一页或两页的上下文训练数据。结果如表3.4所示。仅接收任务的自然语言描述的零-shot GPT-3在BLEU方面仍然不如最近的无监督神经机器翻译结果。然而，为每个翻译任务提供单个示例演示可以提高性能超过7 BLEU，并接近以前的工作的竞争性性能。完全的few-shot设置进一步提高了4 BLEU，使平均性能与以前的无监督神经机器翻译工作相似。GPT-3在不同语言方向上的性能差异明显。对于所研究的三种输入语言，GPT-3在翻译成英语时明显优于以前的无监督神经机器翻译工作，但在反向翻译时则表现不佳。在En-Ro上的性能是明显的离群值，比以前的无监督神经机器翻译工作差10多个BLEU。这可能是因为重用了GPT-2的字节级BPE分词器，该分词器是针对几乎完全是英语的训练数据开发的。对于Fr-En和De-En，few-shot GPT-3优于我们找到的最佳监督结果，但由于我们不熟悉文献，并且这些可能不是竞争性基准，我们不认为这些结果代表真正的最新技术。对于Ro-En，few-shot GPT-3的性能与总体SOTA相差不到0.5 BLEU，这一SOTA是通过无监督预训练、对608K标记示例进行监督微调和反向翻译的组合（LHCG19b）获得的。

Finally, across all language pairs and across all three settings (zero-, one-, and few-shot), there is a smooth trend ofimprovement with model capacity. This is shown in Figure 3.4 in the case of few-shot results, and scaling for all threesettings is shown in Appendix H

最后，在所有语言对和三种设置（零、一和few-shot）中，都出现了与模型容量相关的平稳改善趋势。这在few-shot结果的情况下如图3.4所示，所有三种设置的扩展都在附录H中显示。

## 3.4 Winograd-Style Tasks

The Winograd Schemas Challenge (LDM12] is a classical task in NLP that involves determining which word a pronounrefers to, when the pronoun is grammatically ambiguous but semantically unambiguous to a human. Recently fine-tunedlanguage models have achieved near-human performance on the original Winograd dataset, but more difficult versions such as the adversarially-mined Winogrande dataset (SBBC19] still significantly lag human performance. We testGPT-3's performance on both Winograd and Winogrande, as usual in the zero-, one-, and few-shot seting.

Winograd Schemas Challenge（LDM12）是自然语言处理领域的经典任务，涉及确定代词引用的具体词语，当代词在语法上具有歧义，但在语义上对人类来说是明确的。最近，经过精细调整的语言模型在原始的Winograd数据集上已经实现了接近于人类表现的性能，但对于更难的版本，如经过对抗性策略生成的Winogrande数据集（SBBC19），仍然明显滞后于人类表现。我们测试了GPT-3在Winograd和Winogrande数据集上的性能，通常使用零次、一次和少数次的评估设置。

On Winograd we test GPT-3 on the original set of 273 Winograd schemas, using the same "partial evaluation" methoddescribed in (RWC+19). Note that this setting differs slightly from the WSC task in the SuperGLUE benchmark, whichis presented as binary classification and requires entity extraction to convert to the form described in this section. 0nWinograd GPT-3 achieves 88.3%, 89.7%, and 88.6% in the zero-shot, one-shot, and few-shot settings, showing no clearin-context learning but in all cases achieving strong results just a few points below state-of-the-art and estimated humanperformance. We note that contamination analysis found some Winograd schemas in the training data but this appearsto have only a small effect on results (see Section 4)

Winograd Schemas Challenge（LDM12）是自然语言处理领域的经典任务，涉及确定代词引用的具体词语，当代词在语法上具有歧义，但在语义上对人类来说是明确的。最近，经过精细调整的语言模型在原始的Winograd数据集上已经实现了接近于人类表现的性能，但对于更难的版本，如经过对抗性策略生成的Winogrande数据集（SBBC19），仍然明显滞后于人类表现。我们测试了GPT-3在Winograd和Winogrande数据集上的性能，通常使用零次、一次和少数次的评估设置。

On the more difficult Winogrande dataset, we do find gains to in-context learning: GPT-3 achieves 70.2% in thezero-shot setting, 73.2% in the one-shot setting, and 77.7% in the few-shot setting. For comparison a fine-tunedRoBERTA model achieves 79%, state-of-the-art is 84.6% achieved with a fine-tuned high capacity model (T5), andhuman performance on the task as reported by [SBBC191 is 94.0%.

在更困难的Winogrande数据集上，我们发现在上下文学习方面有所提高：GPT-3在零次设置中达到了70.2%，在一次设置中达到了73.2%，在少数次设置中达到了77.7%。为了比较，一个经过微调的RoBERTA模型达到了79%的成绩，最新的艺术水平是使用经过微调的高容量模型（T5）达到的84.6%，而根据[SBBC191]报告的任务人类表现为94.0%。

## 3\.5 Common Sense Reasoning

Next we consider three datasets which attempt to capture physical or scientifc reasoning, as distinct from sentencecompletion, reading comprehension, or broad knowledge question answering, The first, PhysicalOA (PIOA) (BZB+19asks common sense questions about how the physical world works and is intended as a probe of grounded understandingof the world. GPT-3 achieves 81.0% accuracy zero-shot, 80.5% accuracy one-shot, and 82.8% accuracy few-shot(the last measured on PIOA's test server). This compares favorably to the 79.4% accuracy prior state-of-the-art of a fne-tuned RoBERTa. PIOA shows relatively shallow scaling with model size and is still over 10% worse than humanerformance, but GPT-3's few-shot and even zero-shot result outperform the current state-of-the-art. Our analysislagged PIOA for a potential data contamination issue (despite hidden test labels), and we therefore conservatively markthe result with an asterisk. See Section 4 for details

接下来，我们考虑了三个试图捕捉物理或科学推理的数据集，与句子完成、阅读理解或广泛的知识问题回答不同。首先，PhysicalOA (PIOA) (BZB+19)提出了有关物理世界运作方式的常识问题，旨在探讨对世界的基础理解。GPT-3在零次设置下达到81.0%的准确率，在一次设置下达到80.5%的准确率，最后在PIOA的测试服务器上实现了82.8%的准确率。这与经过微调的RoBERTa的79.4%准确率相比表现更好。PIOA在模型大小上的扩展相对较浅，仍然比人类表现差了超过10%，但GPT-3的少数次甚至零次结果超过了当前的艺术水平。我们对PIOA进行了潜在的数据污染问题的分析（尽管隐藏了测试标签），因此我们以星号标记了结果。详细信息请参见第4节。

ARC (CCE+18] is a dataset of multiple-choice questions collected from 3rd to 9th grade science exams. On the'Challenge" version of the dataset which has been filtered to questions which simple statistical or information retrievamethods are unable to correctly answer, GPT-3 achieves 51.4% accuracy in the zero-shot setting, 53.2% in the one-shotsetting, and 51.5% in the few-shot setting. This is approaching the performance of a fine-tuned RoBERTa baseline(55.9%) from UnifedOA (KKS+201. On the "Easy ersion of the dataset (questions which either of the mentionedbaseline approaches answered correctly), GPT-3 achieves 68.8%, 71.2%, and 70.1% which slightly exceeds a fine-tunedRoBERTa baseline from (KKS+201. However, both of these results are still much worse than the overall SOTAsachieved by the UnifiedOA which exceeds GPT-3's few-shot results by 27% on the challenge set and 22% on the easyset.

ARC (CCE+18) 是一个收集自第三到第九年级科学考试的多项选择问题数据集。在该数据集的“挑战”版本中，该数据集已经经过筛选，传统的统计或信息检索方法无法正确回答其中的问题。在零次设置下，GPT-3的准确率达到了51.4%，在一次设置下为53.2%，在少数次设置下为51.5%。这接近了来自UnifiedOA (KKS+20) 的微调RoBERTa基线（55.9%）的性能。在数据集的“简单”版本中（其中提到的基线方法中的任何一个都能正确回答问题），GPT-3的准确率分别为68.8%、71.2%和70.1%，略高于来自(KKS+20)的微调RoBERTa基线。然而，这两个结果仍然远远不如UnifiedOA的整体最佳结果，UnifiedOA在挑战集上超过了GPT-3的少数次结果27%，在简单集上超过了GPT-3的22%。

On OpenBookOA (MCKS18), GPT-3 improves significantly from zero to few shot settings but is still over 20 pointsshort of the overall SOTA. GPT-3's few-shot performance is similar to a fine-tuned BERT Large baseline on theleaderboard.

Overal. in-context learning with GP-3 shows mixed results on commonsense reasoning tasks. with only small aninconsistent gains observed in the one and few-shot learning settings for both PIOA and ARC, but a signifcantimprovement is observed on OpenBookOA. GPT-3 sets SOTA on the new PIOA dataset in all evaluation settings.

在OpenBookOA（MCKS18）上，GPT-3从零到少次设置有了显著的改进，但仍然比整体最佳结果差20多个百分点。GPT-3的少次设置性能类似于排行榜上的微调BERT Large基线。

总的来说，在通感推理任务中，GPT-3的在上下文学习方面的表现各异，PIOA和ARC的一次和少次学习设置中只观察到了小幅且不一致的增益，但在OpenBookOA上观察到了显著的改进。GPT-3在新的PIOA数据集上在所有评估设置中都取得了最佳结果。

