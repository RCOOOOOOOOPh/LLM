# Large Language Models are Zero-Shot Reasoners  

Note.

1.

Abstract

Pretrained large language models (LLMs) are widely used in many sub-fields of natural language processing (NLP) and generally known as excellent few-shot learners with task-specific exemplars. Notably, chain of thought (CoT) prompting, a recent technique for eliciting complex multi-step reasoning through step-bystep answer examples, achieved the state-of-the-art performances in arithmetics and symbolic reasoning, difficult system-2 tasks that do not follow the standard scaling laws for LLMs. While these successes are often attributed to LLMs’ ability for few-shot learning, we show that LLMs are decent zero-shot reasoners **by simply adding “Let’s think step by step”** before each answer. Experimental results demonstrate that our Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP), symbolic reasoning (Last Letter, Coin Flip), and other logical reasoning tasks (Date Understanding, Tracking Shuffled Objects), without any hand-crafted few-shot examples, e.g. increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with large-scale InstructGPT model (text-davinci- 002), as well as similar magnitudes of improvements with another off-the-shelf large model, 540B parameter PaLM. The versatility of this single prompt across very diverse reasoning tasks hints at untapped and understudied fundamental zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive capabilities may be extracted by simple prompting. We hope our work not only serves as the minimal strongest zero-shot baseline for the challenging reasoning benchmarks, but also highlights the importance of carefully exploring and analyzing the enormous zero-shot knowledge hidden inside LLMs before crafting finetuning datasets or few-shot exemplars.

摘要
预训练的大型语言模型（LLM）广泛用于自然语言处理（NLP）的许多子领域，通常被认为是出色的少样本学习者，具有特定任务的示例。值得注意的是，最近一种通过逐步答案示例引发复杂的多步推理的技术，即思维链（CoT）提示，在算术和符号推理方面取得了最先进的性能，这些是不遵循LLM标准缩放定律的难度较大的系统-2任务。尽管这些成功通常被归因于LLM的少样本学习能力，但我们展示**LLM在零样本推理中也表现出色**，只需在**每个答案前添加“让我们逐步思考”**。实验证明，我们的零样本-CoT，在使用相同的单个提示模板的情况下，显着优于多样的基准推理任务上的零样本LLM性能，包括算术（MultiArith，GSM8K，AQUA-RAT，SVAMP）、符号推理（Last Letter，Coin Flip）和其他逻辑推理任务（Date Understanding，Tracking Shuffled Objects），都没有手工制作的少样本示例，例如使用大规模InstructGPT模型（text-davinci-002）将MultiArith的准确性从17.7%提高到78.7%，将GSM8K从10.4%提高到40.7%，另一个现成的大型模型，540B参数PaLM也有类似幅度的改进。这个单一提示在非常多样化的推理任务上的多功能性提示着LLM未开发和少研究的基本零样本能力，暗示高级、多任务的广泛认知能力可以通过简单提示来提取。我们希望我们的工作不仅作为具有挑战性的推理基准的最强零样本基线，而且还强调在制定微调数据集或少样本示例之前，仔细探索和分析LLM内部隐藏的大量零样本知识的重要性。

# 1 Introduction

Scaling up the size of language models has been key ingredients of recent revolutions in natural language processing (NLP) [Vaswani et al., 2017, Devlin et al., 2019, Raffel et al., 2020, Brown et al., 2020, Thoppilan et al., 2022, Rae et al., 2021, Chowdhery et al., 2022]. The success of large language models (LLMs) is often attributed to (in-context) few-shot or zero-shot learning. It can solve various tasks by simply **conditioning the models on a few examples (few-shot) or instructions describing the task (zero-shot)**. The method of conditioning the language model is called “**prompting**” [Liu et al., 2021b], and designing prompts either manually [Schick and Schütze, 2021, Reynolds and McDonell, 2021] or automatically [Gao et al., 2021, Shin et al., 2020] has become a hot topic in NLP.

增大语言模型规模一直是自然语言处理（NLP）领域近年革命的关键要素[Vaswani等，2017年，Devlin等，2019年，Raffel等，2020年，Brown等，2020年，Thoppilan等，2022年，Rae等，2021年，Chowdhery等，2022年]。大型语言模型（LLM）的成功通常归因于（上下文中的）少样本或零样本学习。它可以通过**简单地将模型置于少量示例（少样本）或描述任务的指令（零样本）上**来解决各种任务。调节语言模型的方法被称为“**提示**”[Liu等，2021b]，设计提示，无论是手动设计[Schick和Schütze，2021年，Reynolds和McDonell，2021年]还是自动设计[Gao等，2021年，Shin等，2020年]，已成为NLP领域的热门话题。

In contrast to the excellent performance of LLMs in intuitive and single-step system-1 [Stanovich and West, 2000] tasks with task-specific few-shot or zero-shot prompting [Liu et al., 2021b], even language models at the scale of 100B or more parameters had struggled on system-2 tasks requiring slow and multi-step reasoning [Rae et al., 2021]. To address this shortcoming, Wei et al. [2022], Wang et al. [2022] have proposed **chain of thought prompting (CoT)**, which feed LLMs with the step-by-step reasoning examples rather than standard question and answer examples (see Fig. 1-a). Such chain of thought demonstrations facilitate models to generate a reasoning path that decomposes the complex reasoning into multiple easier steps. Notably with CoT, the reasoning performance then satisfies the scaling laws better and jumps up with the size of the language models. For example, when combined with the 540B parameter PaLM model [Chowdhery et al., 2022], chain of thought prompting significantly increases the performance over standard few-shot prompting across several benchmark reasoning tasks, e.g., GSM8K (17.9% → 58.1%). 

与LLMs在直观和单步系统1 [Stanovich和West，2000]任务中表现出色，使用任务特定的少样本或零样本提示[Liu等，2021b]形成鲜明对比，即使在具有100B或更多参数的规模的语言模型中，它们在**需要缓慢和多步推理的系统2任务上仍然面临困难**[Rae等，2021年]。为了解决这一不足，Wei等人[2022年]，Wang等人[2022年]提出了“**链式思维提示”（CoT）**，它向LLMs提供了逐步推理示例，而不是标准的问答示例（见图1-a）。这种链式思维演示有助于模型生成将复杂的推理分解为多个较简单步骤的推理路径。值得注意的是，**使用CoT后，推理性能更好地满足了缩放规律**，并随着语言模型的规模增加而提高。例如，当与540B参数的PaLM模型[Chowdhery等，2022年]结合使用时，链式思维提示显著提高了在多个基准推理任务上的性能，例如GSM8K（从17.9%提高到58.1%）。

While the successes of CoT prompting [Wei et al., 2022], along those of many other task-specific prompting work [Gao et al., 2021, Schick and Schütze, 2021, Liu et al., 2021b], are often attributed to LLMs’ ability for few-shot learning [Brown et al., 2020], we show that LLMs are decent zero-shot reasoners by adding a simple prompt, Let’s think step by step, to facilitate step-by-step thinking before answering each question (see Figure 1). **Despite the simplicity, our Zero-shot-CoT successfully generates a plausible reasoning path in a zero-shot manner and reaches the correct answer in a problem where the standard zero-shot approach fails.** Importantly, our Zero-shot-CoT is **versatile and task-agnostic,** unlike most prior task-specific prompt engineering in the forms of examples (few-shot) or templates (zero-shot) [Liu et al., 2021b]: it can facilitate step-by-step answers across various reasoning tasks, including arithmetic (MultiArith [Roy and Roth, 2015], GSM8K [Cobbe et al., 2021], AQUA-RAT [Ling et al., 2017], and SVAMP [Patel et al., 2021]), symbolic reasoning (Last letter and Coin flip), commonsense reasoning (CommonSenseQA [Talmor et al., 2019] and Strategy QA [Geva et al., 2021]), and other logical reasoning tasks (Date understanding and Tracking Shuffled Objects from BIG-bench [Srivastava et al., 2022]) without modifying the prompt per task.  

尽管链式思维提示（CoT）的成功[Wei等，2022]以及许多其他任务特定提示工作的成功[Gao等，2021，Schick和Schütze，2021，Liu等，2021b]通常归因于LLMs的少样本学习能力[Brown等，2020]，我们通过添加一个简单的提示“让我们逐步思考”，以**在回答每个问题之前促进逐步思考，展示了LLMs具备良好的零样本推理能力**（见图1）。尽管简单，我们的“零样本-CoT”成功地以零样本方式生成了一个合理的推理路径，并在标准零样本方法失败的问题中得出了正确的答案。重要的是，我们的“零样本-CoT”是**多才多艺的**，而且**不依赖于任务**，不像以前的大多数任务特定的提示工程形式是示例（少样本）或模板（零样本）[Liu等，2021b]：它可以在各种推理任务中促进逐步回答，包括算术（MultiArith [Roy and Roth，2015]，GSM8K [Cobbe et al.，2021]，AQUA-RAT [Ling等，2017]和SVAMP [Patel等，2021]），符号推理（最后一个字母和抛硬币），常识推理（CommonSenseQA [Talmor等，2019]和Strategy QA [Geva等，2021]）以及其他逻辑推理任务（来自BIG-bench的Date understanding和Tracking Shuffled Objects[Srivastava等，2022]），而不需要针对每个任务修改提示。

We empirically evaluate Zero-shot-CoT against other prompting baselines in Table 2. While our Zero-shot-CoT underperforms Few-shot-CoT with carefully-crafted and task-specific step-by-step examples, Zero-shot-CoT achieves enormous score gains compared to the zero-shot baseline, e.g. from 17.7% to 78.7% on MultiArith and from 10.4% to 40.7% on GSM8K with large-scale InstructGPT  model (text-davinci-002). We also evaluate Zero-shot-CoT with another off-the-shelf large model, 540B parameter PaLM, showing similar magnitudes of improvements on MultiArith and GSM8K. Importantly, with our single fixed prompt, zero-shot LLMs have a significantly better scaling curve comparable to that of the few-shot CoT baseline. We also show that besides Few-shot-CoT requiring human engineering of multi-step reasoning prompts, their performance deteriorates if prompt example question types and task question type are unmatched, suggesting high sensitivity to per-task prompt designs. In contrast, the versatility of this single prompt across diverse reasoning tasks hints at untapped and understudied zero-shot fundamental capabilities of LLMs, such as higher-level broad cognitive capabilities like generic logical reasoning [Chollet, 2019]. While the vibrant field of LLMs started out from the premise of excellent few-shot learners [Brown et al., 2020], we hope our work encourages more research into uncovering high-level and multi-task zero-shot capabilities hidden inside those models.  

我们在表2中对Zero-shot-CoT与其他提示基线进行了实证评估。尽管我们的Zero-shot-CoT在与精心制作和特定任务的逐步示例相比性能不佳，但与零样本基线相比，Zero-shot-CoT取得了巨大的得分增益，例如，使用大规模InstructGPT模型（text-davinci-002）在MultiArith上从17.7%提高到78.7%，在GSM8K上从10.4%提高到40.7%。我们还使用另一款现成的大型模型，拥有540B参数的PaLM，对MultiArith和GSM8K进行了类似幅度的改进。重要的是，使用我们的单一固定提示，零样本LLMs具有与少样本CoT基线相媲美的更好的扩展曲线。我们还表明，尽管Few-shot-CoT需要人工设计多步推理提示，但如果提示示例问题类型与任务问题类型不匹配，其性能会下降，表明对每个任务的提示设计非常敏感。相反，这个单一提示在各种推理任务中的多才多艺提示了LLMs未经开发和研究的零样本基本能力，例如更高级别的广泛认知能力，如通用逻辑推理[Chollet，2019]。虽然LLMs的活跃领域始于出色的少样本学习者[Brown等，2020]的前提，但我们希望我们的工作鼓励更多的研究来揭示这些模型内部隐藏的高级和多任务零样本能力。