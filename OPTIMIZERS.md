# LARGE LANGUAGE MODELS AS OPTIMIZERS  

摘要
优化是无处不在的。虽然基于导数的算法已经成为各种问题的强大工具，但梯度的缺失给许多现实世界的应用带来了挑战。在这项工作中，我们提出了一种名为**PROmpting优化（OPRO）**的简单有效的方法，以利用大型语言模型（LLMs）作为优化器，其中优化任务以自然语言描述。在每个优化步骤中，LLM从包含先前生成的解和它们的值的提示中生成新的解决方案，然后评估新的解决方案并将其添加到下一个优化步骤的提示中。我们首先展示了OPRO在线性回归和旅行推销员问题上的应用，然后转向提示优化，目标是找到最大化任务准确性的指令。通过多种LLMs，我们证明了由OPRO优化的最佳提示在GSM8K上比人工设计的提示表现出高达8％的优势，在Big-Bench Hard任务上高达50％。

# 1 INTRODUCTION

Optimization is critical for all areas. Many optimization techniques are iterative: the optimization starts from an initial solution, then iteratively updates the solution to optimize the objective function (Amari, 1993; Qian, 1999; Kingma & Ba, 2015; Bäck & Schwefel, 1993; Rios & Sahinidis, 2013; Reeves, 1993). The optimization algorithm typically needs to be customized for an individual task to deal with the specific challenges posed by the decision space and the performance landscape, especially for derivative-free optimization.

优化对于所有领域都至关重要。 许多优化技术都是迭代的：优化从初始解开始，然后迭代更新解以优化目标函数（Amari，1993；Qian，1999；Kingma & Ba，2015；Bäck & Schwefel，1993；Rios & Sahinidis，2013） ；里夫斯，1993）。 优化算法通常需要针对单个任务进行定制，以应对决策空间和性能景观带来的特定挑战，特别是对于无导数优化。

In this work, we propose **Optimization by PROmpting (OPRO)**, a simple and effective approach to utilize large language models (LLMs) as optimizers. With the advancement of prompting techniques, LLMs have achieved impressive performance on a variety of domains (Wei et al., 2022; Kojima et al., 2022; Wang et al., 2022; Zhou et al., 2022a; Madaan et al., 2023; Bai et al., 2022; Chen et al., 2023e). Their ability to understand natural language lays out a new possibility for optimization: instead of formally defining the optimization problem and deriving the update step with a programmed solver, we describe the optimization problem in natural language, then instruct the LLM to iteratively generate new solutions based on the problem description and the previously found solutions. Optimization with LLMs enables quick adaptation to different tasks by changing the problem description in the prompt, and the optimization process can be customized by adding instructions to specify the desired properties of the solutions.

在这项工作中，我们提出了一种名为"优化引导"（OPRO）的方法，这是一种简单而有效的方法，用于利用大型语言模型（LLMs）作为优化器。随着提示技术的进步，LLMs在各种领域取得了令人印象深刻的性能（Wei等，2022；Kojima等，2022；Wang等，2022；Zhou等，2022a；Madaan等，2023；Bai等，2022；Chen等，2023e）。它们理解自然语言的能力为优化提供了新的可能性：我们不再需要正式定义优化问题并使用编程求解器推导更新步骤，而是以**自然语言描述优化问题，然后指示LLM基于问题描述和先前找到的解决方案迭代生成新的解决方案**。使用LLMs进行优化可以通过更改提示中的问题描述来快速适应不同的任务，并且可以通过添加指令来自定义优化过程，以指定所需解决方案的属性。

To demonstrate the potential of LLMs for optimization, we first present case studies on linear regression and the traveling salesman problem, which are two classic optimization problems that underpin many others in mathematical optimization, computer science, and operations research. On small-scale optimization problems, we show that LLMs are able to find good-quality solutions simply through prompting, and sometimes match or surpass hand-designed heuristic algorithms.

为了展示LLMs在优化中的潜力，我们首先针对线性回归和旅行推销员问题进行了案例研究。这两个问题是数学优化、计算机科学和运筹学中许多其他问题的基础，都是经典的优化问题。在小规模的优化问题上，我们展示了LLMs仅通过提示就能够找到高质量的解决方案，有时甚至能够匹敌或超越手工设计的启发式算法。

Next, we demonstrate the **ability of LLMs to optimize prompts**: the **optimization goal is to find a prompt** that maximizes the task accuracy. Specifically, we focus on natural language processing tasks where both the task input and output are in text formats. LLMs are shown to be sensitive to the prompt format (Zhao et al., 2021; Lu et al., 2021; Wei et al., 2023; Madaan & Yazdanbakhsh, 2022); in particular, semantically similar prompts may have drastically different performance (Kojima et al., 2022; Zhou et al., 2022b; Zhang et al., 2022), and the optimal prompt formats can be model-specific and task-specific (Ma et al., 2023; Chen et al., 2023c). Therefore, prompt engineering is often important for LLMs to achieve good performance (Reynolds & McDonell, 2021). However, the large and discrete prompt space makes it challenging for optimization, especially when only API access to the LLM is available. Following prior work on continuous and discrete prompt optimization (Lester et al., 2021; Li & Liang, 2021; Zhou et al., 2022b; Pryzant et al., 2023), we assume a training set is available to compute the training accuracy as the objective value for optimization, and we show in experiments that optimizing the prompt for accuracy on a small training set is sufficient to reach high performance on the test set.

接下来，我们展示了LLMs优化提示的能力：优化目标是找到一个能够最大化任务准确性的提示。具体来说，我们关注自然语言处理任务，其中任务的输入和输出都以文本格式呈现。LLMs对提示的格式非常敏感（Zhao等，2021；Lu等，2021；Wei等，2023；Madaan＆Yazdanbakhsh，2022）；特别是，语义上相似的提示可能在性能上有显著差异（Kojima等，2022；Zhou等，2022b；Zhang等，2022），而最佳的提示格式可能是模型特定和任务特定的（Ma等，2023；Chen等，2023c）。因此，对于LLMs来实现良好性能，提示工程通常非常重要（Reynolds＆McDonell，2021）。然而，庞大而离散的提示空间使得优化变得具有挑战性，尤其是当只能通过API访问LLM时。在之前关于连续和离散提示优化的工作（Lester等，2021；Li＆Liang，2021；Zhou等，2022b；Pryzant等，2023）的基础上，我们假设有一个训练集可用于计算训练准确性作为优化的目标值，并在实验中表明，在小规模训练集上优化提示的准确性足以在测试集上实现高性能。

**The prompt to the LLM serves as a call to the optimizer, and we name it the meta-prompt.** Figure 3 shows an example. The meta-prompt contains two core pieces of information. The first piece is **previously generated prompts with their corresponding training accuracies**. The second piece is the **optimization problem description**, which includes several exemplars randomly selected from the training set to exemplify the task of interest. We also provide instructions for the LLM to understand the relationships among different parts and the desired output format. Different from recent work on using LLMs for automatic prompt generation (Zhou et al., 2022b; Pryzant et al., 2023), each optimization step in our work generates new prompts that aim to increase the test accuracy based on a trajectory of previously generated prompts, instead of editing one input prompt according to natural language feedback (Pryzant et al., 2023) or requiring the new prompt to follow the same semantic meaning (Zhou et al., 2022b). Making use of the full optimization trajectory, **OPRO enables the LLM to gradually generate new prompts** that improve the task accuracy throughout the optimization process, where the initial prompts have low task accuracies.

对LLM的提示充当了一个优化器的调用，我们称之为元提示。图3显示了一个示例。元提示包含两个核心信息。第一部分是以前生成的提示及其相应的训练准确性。第二部分是优化问题描述，其中包括从训练集中随机选择的几个示例，以示范感兴趣的任务。我们还提供了LLM理解不同部分之间关系和所需输出格式的指示。与最近关于使用LLMs进行自动提示生成的工作不同（Zhou等，2022b；Pryzant等，2023），我们的工作中的每个优化步骤生成的新提示旨在根据先前生成的提示的轨迹增加测试准确性，而不是根据自然语言反馈编辑一个输入提示（Pryzant等，2023）或要求新提示遵循相同的语义意义（Zhou等，2022b）。利用完整的优化轨迹，**OPRO使LLM逐渐生成新提示**，以提高整个优化过程中的任务准确性，而初始提示具有较低的任务准确性。

![image-20231009024403064](D:\coursefile\LLM\typorapic\image-20231009024403064.png)

图3：在GSM8K上使用经过指令调整的PaLM 2-L（PaLM 2-L-IT）进行提示优化的元提示示例，生成的指令将被添加到评分LLM输出（第4.1节的A_begin）的开头的“A：”之前。 <INS>表示将添加生成的指令的位置。蓝色文本包含解决方案-分数对；紫色文本描述了优化任务和输出格式；橙色文本是元指令。

We conduct comprehensive evaluation on several LLMs, including text-bison 1 and Palm 2-L in the PaLM-2 model family (Anil et al., 2023), as well as gpt-3.5-turbo and gpt-4 in the GPT model family 2 . We optimize prompts on GSM8K (Cobbe et al., 2021) and Big-Bench Hard (Suzgun et al., 2022), which are reasoning benchmarks where prompting techniques have achieved remarkable performance breakthrough (Wei et al., 2022; Kojima et al., 2022; Suzgun et al., 2022). Starting from initial prompts with low task accuracies, we show that all LLMs in our evaluation are able to serve as optimizers, which consistently improve the performance of the generated prompts through iterative optimization until convergence (see Figure 1). In particular, while these LLMs generally produce instructions of different styles (see Table 1), with zero-shot prompting, their best generated instructions match the few-shot chain-of-thought prompting performance when applied to PaLM 2-L (Anil et al., 2023), outperforming the zero-shot performance with human-designed prompts by up to 8% on GSM8K. Additionally, we observe that the OPRO-optimized prompts transfer to other benchmarks of the same domain and also deliver notable performance gain.

我们对几种LLMs进行了全面评估，包括PaLM-2模型家族中的text-bison 1和Palm 2-L（Anil等，2023），以及GPT模型家族中的gpt-3.5-turbo和gpt-4。我们在GSM8K（Cobbe等，2021）和Big-Bench Hard（Suzgun等，2022）上优化提示，这些是推理基准，其中提示技术取得了显著的性能突破（Wei等，2022；Kojima等，2022；Suzgun等，2022）。从具有低任务准确性的初始提示开始，我们展示了我们评估中的所有LLMs都能够充当优化器，通过迭代优化一直提高生成的提示的性能，直到收敛（参见图1）。特别是，尽管这些LLMs通常生成不同风格的指令（见表1），但在零-shot提示的情况下，它们最佳生成的指令与PaLM 2-L（Anil等，2023）的few-shot chain-of-thought提示性能相匹配，在GSM8K上，它们在零-shot性能上超过了人工设计的提示高达8%。此外，我们观察到，OPRO优化的提示可以转移到同一领域的其他基准，并且也能够带来显著的性能提升。

# 2 OPRO: LLM AS THE OPTIMIZER 

Figure 2 illustrates the overall framework of OPRO. In each optimization step, the LLM generates candidate solutions to the optimization task based on the optimization problem description and previously evaluated solutions in the meta-prompt. Then the new solutions are evaluated and added to the meta-prompt for the subsequent optimization process. The optimization process terminates when the LLM is unable to propose new solutions with better optimization scores, or a maximum number of optimization steps has reached. We first outline the desired features of LLMs for optimization, then describe the key design choices based on these desirables.

图2展示了OPRO的总体框架。在每个优化步骤中，LLM根据优化问题描述和元提示中先前评估的解决方案生成优化任务的候选解决方案。然后，新的解决方案将被评估并添加到元提示中，以用于后续的优化过程。当LLM无法提出具有更好的优化分数的新解决方案，或者达到了最大优化步骤数时，优化过程终止。首先，我们概述了用于优化的LLMs的期望特性，然后根据这些期望特性描述了关键的设计选择。

## 2.1 DESIRABLES OF OPTIMIZATION BY LLMS 

Making use of natural language descriptions. The main advantage of LLMs for optimization is their ability of understanding natural language, which allows people to describe their optimization tasks without formal specifications. For instance, in prompt optimization where the goal is to find a prompt that optimizes the task accuracy, the task can be described with a high-level text summary along with input-output examples. 

充分利用自然语言描述。LLMs在优化中的主要优势在于它们能够理解自然语言，这使人们能够在没有正式规范的情况下描述他们的优化任务。例如，在提示优化中，目标是找到一个可以优化任务准确性的提示，任务可以用高级文本摘要以及输入输出示例来描述。

![image-20231009025051286](C:\Users\think\Desktop\LARGE LANGUAGE MODELS AS OPTIMIZERS\image-20231009025051286.png)

图2：OPRO框架概述。给定元提示作为输入，LLM生成新的解决方案以满足目标函数，然后将新的解决方案及其分数添加到元提示中，以进行下一次优化步骤。**元提示包含整个优化过程中获得的解决方案-分数对，以及任务的自然语言描述和（在提示优化中）来自任务的一些示例**。有关提示优化的示例元提示，请参见图3。

Trading off exploration and exploitation. The exploration-exploitation trade-off is a fundamental challenge in optimization, and it is important for LLMs serving as optimizers to balance these two competing goals. This means that the LLM should be able to exploit promising areas of the search space where good solutions are already found, while also exploring new regions of the search space so as to not miss potentially better solutions.

权衡探索和利用。探索与利用的权衡是优化中的一个基本挑战，对于充当优化器的LLMs来说，平衡这两个竞争性目标非常重要。这意味着LLM应该能够充分利用搜索空间中已找到的良好解的有前景区域，同时也要探索搜索空间的新区域，以避免错过潜在的更好解决方案。

## 2.2 META-PROMPT DESIGN

As the input to the LLM that acts as the optimizer, the meta-prompt contains the following two essential parts.

作为充当优化器的LLM的输入，元提示包含以下两个基本部分。

Optimization problem description. The first part is the text description of the optimization problem, including the objective function and solution constraints. For example, for prompt optimization, the LLM can be instructed to “generate a new instruction that achieves a higher accuracy”, and we denote such instructions in the meta-prompt as **meta-instructions**. We can also provide customized meta-instructions as an informal regularization of the generated solutions, such as “the instruction should be concise and generally applicable”.

优化问题描述。第一部分是**优化问题的文本描述**，包括目标函数和解决方案约束。例如，在提示优化中，可以**指示LLM“生成一个新的指令，以获得更高的准确性”**，我们将这种指令在元提示中称为**元指令**。我们还可以提供定制的元指令，作为生成的解决方案的非正式正则化，例如**“指令应该简洁且通用适用”**。

Optimization trajectory. Besides understanding natural language instructions, LLMs are also shown to be able to recognize patterns from in-context demonstrations (Wei et al., 2023; Madaan & Yazdanbakhsh, 2022; Mirchandani et al., 2023). Our meta-prompt makes use of this property and instructs the LLM to leverage the optimization trajectory for generating new solutions. Specifically, the optimization trajectory includes past solutions paired with their optimization scores, sorted in the ascending order. Including optimization trajectory in the meta-prompt allows the LLM to identify similarities of solutions with high scores, encouraging the LLM to build upon existing good solutions to construct potentially better ones without the need of explicitly defining how the solution should be updated.

优化轨迹。除了理解自然语言指令外，LLMs还被证明能够从上下文演示中识别模式（Wei等，2023；Madaan＆Yazdanbakhsh，2022；Mirchandani等，2023）。我们的元提示利用了这一特性，并指示LLM利用优化轨迹生成新的解决方案。具体来说，**优化轨迹包括过去的解决方案以及它们的优化分数，按升序排列**。在元提示中包括优化轨迹使LLM能够识别具有高分数的解决方案的相似性，鼓励LLM在构建潜在更好的解决方案时建立在现有的好解决方案之上，而无需明确定义解决方案应如何更新。

## 2.3 SOLUTION GENERATION

At the solution generation step, the LLM generates new solutions with the meta-prompt as input. The following are the key optimization challenges we address in this stage.

在解决方案生成步骤中，LLM以元提示作为输入生成新的解决方案。以下是我们在这一阶段解决的关键优化挑战。

Optimization stability. In the optimization process, not all solutions achieve high scores and monotonically improve over prior ones. Due to the sensitivity of in-context learning to the prompt, LLM output can be drastically affected by low-quality solutions in the input optimization trajectory, especially at the beginning when the solution space has not been adequately explored. This sometimes results in optimization instability and large variance. To improve stability, we prompt the LLM to generate multiple solutions at each optimization step, allowing the LLM to simultaneously explore multiple possibilities and quickly discover promising directions to move forward.

优化稳定性。在优化过程中，并非所有解决方案都能获得高分数并且逐步改进。由于上下文学习对提示的敏感性，LLM输出可能会受到输入优化轨迹中**低质量解决方案的严重影响**，特别是在解决方案空间尚未充分探索的初期。这有时会导致优化不稳定性和大的方差。为了提高稳定性，我们提示LLM在每个优化步骤中**生成多个解决方案**，允许LLM同时探索多种可能性，并迅速发现有希望的前进方向。

Exploration-exploitation trade-off. We tune the LLM sampling temperature to balance between exploration and exploitation. A lower temperature encourages the LLM to exploit the solution space around the previously found solutions and make small adaptations, while a high temperature allows the LLM to more aggressively explore solutions that can be notably different.

探索与利用的权衡。我们调整LLM的抽样温度以平衡探索与利用。较低的温度鼓励LLM在先前找到的解决方案周围的解决方案空间中进行利用并进行小的调整，而较高的温度允许LLM更积极地探索可能明显不同的解决方案。

# 3 MOTIVATING EXAMPLE: MATHEMATICAL OPTIMIZATION

We first demonstrate the potential of LLMs in serving as optimizers for mathematical optimization. In particular, we present a case study on linear regression as an example of continuous optimization, and on the Traveling Salesman Problem (TSP) as an example of discrete optimization. On both tasks, we see LLMs properly capture the optimization directions on small-scale problems merely based on the past optimization trajectory provided in the meta-prompt.

我们首先展示了LLMs在数学优化中作为优化器的潜力。具体而言，我们以线性回归为例进行了连续优化的案例研究，并以旅行推销员问题（TSP）为例进行了离散优化的案例研究。在这两个任务中，我们看到LLMs仅基于元提示中提供的过去优化轨迹，能够正确捕捉小规模问题的优化方向。

## 3.1 LINEAR REGRESSION

In linear regression problems, the goal is to find the linear coefficients that probabilistically best explain the response from the input variables. We study the setting in which the independent and dependent variables X and y are both one-dimensional and an intercept b is present, so that there are two one-dimensional variables w, b to optimize over. In a synthetic setting, we sample ground truth values for one-dimensional variables wtrue and btrue, and generate 50 data points by y = wtruex + btrue + ϵ, in which x ranges from 1 to 50 and ϵ is the standard Gaussian noise. Our optimization starts from 5 randomly sampled (w, b) pairs. In each step, we prompt an instructiontuned LLM with a meta-prompt that includes the best 20 (w, b) pairs in history and their sorted objective values. The meta-prompt then asks for a new (w, b) pair that further decreases the objective value. A sample meta-prompt is shown in Figure 17 of Appendix C.1. We prompt the meta-prompt 8 times to generate at most 8 new (w, b) pairs in each step to improve optimization stability. Then we evaluate the objective value of the proposed pair and add it to history. We do black-box optimization: the analytic form does not appear in the meta-prompt text. This is because the LLM can often calculate the solution directly from the analytic form.

在线性回归问题中，目标是找到概率上最能解释输入变量响应的线性系数。我们研究了独立变量X和因变量y都是一维且存在截距b的情况，因此存在两个一维变量$w$和$b$需要进行优化。在一个合成环境中，我们对一维变量$w_{true}$和$b_{true}$进行了基本真值采样，并通过$y = w_{true}x + b_{true} + ϵ$生成了50个数据点，其中x的范围从1到50，ϵ是标准的高斯噪声。我们的优化从随机采样的5个(w, b)对开始。在每个步骤中，我们使用一个经过指令调整的LLM**提示一个元提示，其中包括历史上最佳的20个(w, b)对及其排序的目标值**。然后元提示要求**生成一个新的(w, b)对**，以进一步减少目标值。附录C.1的图17中显示了一个示例元提示。我们提示元提示8次，在每个步骤中生成至多8个新的(w, b)对以提高优化稳定性。然后，我们评估所提出的对的目标值，并将其添加到历史记录中。我们进行黑盒优化：元提示文本中不包含分析形式。这是因为LLM通常可以直接从分析形式中计算出解决方案。

(注：把C.1放在这了）

![image-20231009030457176](D:\coursefile\LLM\typorapic\image-20231009030457176.png)

Table 2 summarizes the results with one of the following optimizer LLMs: text-bison, gpt-3.5-turbo, and gpt-4. We study three settings of wtrue and btrue: within the starting region [10, 20] × [10, 20], “near outside” (each of wtrue and btrue is outside the starting region but the distance is less than 10), and “far outside” (each of wtrue and btrue is outside the starting region and the distance is greater than 10). We see:

表2总结了使用以下优化器LLMs之一的结果：text-bison、gpt-3.5-turbo和gpt-4。我们研究了三种wtrue和btrue的设置：在初始区域[10, 20] × [10, 20]内，“近外”（wtrue和btrue中的每一个都在初始区域之外，但距离小于10），以及“远外”（wtrue和btrue中的每一个都在初始区域之外，距离大于10）。我们可以看到：

• The number of unique (w, b) pairs explored by each model is fewer than exhaustive search, indicating these models are able to to do black-box optimization: compare the numbers and propose a descent direction.

每个模型探索的唯一(w, b)对的数量都比穷举搜索少，表明这些模型能够进行黑盒优化：比较这些数字并提出下降方向。

• The text-bison and gpt-4 models outperform gpt-3.5-turbo in convergence speed: they arrive at the optima with fewer steps. The gpt-4 model also outperforms in finding the optima with fewer explored unique points. Taking a closer look at the optimization trajectory, we see gpt-4 is the best at proposing a reasonable next step from the history: for example, when the history shows the objective values of (w, b) = (8, 7), (w, b) = (8, 6), and (w, b) = (8, 5) are decreasing, it has a highest chance to propose (w, b) = (8, 4) for evaluation.

• text-bison和gpt-4模型在收敛速度上优于gpt-3.5-turbo：它们在更少的步骤内到达最优解。gpt-4模型在找到最优解时也比探索的唯一点数少。仔细观察优化轨迹，我们可以看到**gpt-4最擅长从历史中提出合理的下一步：例如，当历史显示(w, b) = (8, 7)、(w, b) = (8, 6)和(w, b) = (8, 5)的目标值在下降时，它最有可能提出(w, b) = (8, 4)进行评估。**

• The problem becomes harder for all models when the ground truth moves farther from the starting region: all models need more explorations and more steps.

• 当基本真值远离起始区域时，所有模型都会变得更加困难：所有模型都需要更多的探索和更多的步骤。

## 3.2 TRAVELING SALESMAN PROBLEM (TSP)

Next, we consider the Traveling Salesman Problem (TSP) (Jünger et al., 1995; Gutin & Punnen, 2006), a classical combinatorial optimization problem with numerous algorithms proposed in literature, including heuristic algorithms and solvers (Rosenkrantz et al., 1977; Golden et al., 1980; Optimization et al., 2020; Applegate et al., 2006; Helsgaun, 2017), and approaches based on training deep neural networks (Kool et al., 2019; Deudon et al., 2018; Chen & Tian, 2019; Nazari et al., 2018). Specifically, given a set of n nodes with their coordinates, the TSP task is to find the shortest route that traverses all nodes from the starting node and finally returns to the starting node.

接下来，我们考虑旅行推销员问题（TSP）（Jünger et al., 1995; Gutin & Punnen, 2006），这是一个经典的组合优化问题，在文献中提出了许多算法，包括启发式算法和求解器（Rosenkrantz et al., 1977; Golden et al., 1980; Optimization et al., 2020; Applegate et al., 2006; Helsgaun, 2017），以及基于训练深度神经网络的方法（Kool et al., 2019; Deudon et al., 2018; Chen & Tian, 2019; Nazari et al., 2018）。具体来说，给定一组带有坐标的n个节点，TSP任务是找到从起始节点开始穿越所有节点并最终返回起始节点的最短路径。

Our optimization process with LLMs starts from 5 randomly generated solutions, and each optimization step produces at most 8 new solutions. We present the meta-prompt in Figure 18 of Appendix C.1. We generate the problem instances by sampling n nodes with both x and y coordinates in [−100, 100]. We use the Gurobi solver (Optimization et al., 2020) to construct the oracle solutions and compute the optimality gap for all approaches, where the optimality gap is defined as the difference between the distance in the solution constructed by the evaluated approach and the distance achieved by the oracle solution, divided by the distance of the oracle solution. Besides evaluating OPRO with different LLMs including text-bison, gpt-3.5-turbo and gpt-4, we also compare OPRO to the following heuristics:

我们使用LLMs的优化过程从随机生成的5个解开始，每个优化步骤最多生成8个新的解。我们在附录C.1的图18中展示了元提示。我们通过在[-100, 100]范围内采样带有x和y坐标的n个节点来生成问题实例。我们使用Gurobi求解器（Optimization et al., 2020）来构建Oracle解决方案，并计算所有方法的最优性差距，其中最优性差距定义为评估方法构建的解决方案中的距离与Oracle解决方案达到的距离之间的差距，除以Oracle解决方案的距离。除了使用不同的LLMs，包括text-bison、gpt-3.5-turbo和gpt-4来评估OPRO外，我们还将OPRO与以下启发式方法进行比较：

• Nearest Neighbor (NN). Starting from an initial node, the solution is constructed with the nearest neighbor heuristic: At each step, among the remaining nodes that are not included in the current partial solution, NN selects the node with the shortest distance to the end node of the partial solution, and adds it as the new end node. The process finishes when all nodes have been added to the solution.

• Farthest Insertion (FI). One caveat of the nearest neighbor heuristic is that it does not take the distance between the start and end node into consideration when constructing partial solutions. To address this issue, FI aims to optimize the cost of inserting new nodes into the partial solution at each step. Define the minimal insertion cost of adding a new node k as c(k) = min(i,j) d(i, k) + d(k, j) − d(i, j), where i and j are adjacent nodes in the current tour, and d(·, ·) represents the distance between two nodes. At each step, FI adds a new node that maximizes the minimal insertion cost.

• 最近邻算法（NN）。从初始节点开始，解决方案是通过最近邻启发式方法构建的：在每一步中，在当前部分解决方案中没有包含的剩余节点中，NN选择到部分解决方案的末尾节点距离最短的节点，并将其添加为新的末尾节点。当所有节点都已添加到解决方案中时，该过程结束。

• 最远插入算法（FI）。最近邻启发式算法的一个问题是，在构建部分解决方案时，它不考虑起始节点和结束节点之间的距离。为了解决这个问题，FI旨在优化在每一步中将新节点插入到部分解决方案中的成本。将添加新节点k的最小插入成本定义为c(k) = min(i,j) d(i, k) + d(k, j) − d(i, j)，其中i和j是当前巡回中相邻的节点，d(·, ·)表示两个节点之间的距离。在每一步中，FI添加一个最大化最小插入成本的新节点。

We present the results in Table 3. We randomly generate 5 problem instances for each number of nodes n. In addition to measuring the optimality gap, on problems where the LLM finds the optimal solutions, we also show the number of optimization steps taken to reach the global optimum. First, we observe that gpt-4 significantly outperforms gpt-3.5-turbo and text-bison across all problem sizes. Specifically, on smaller-scale problems, gpt-4 reaches the global optimum about 4× faster than other LLMs. On larger-scale problems, especially with n = 50, gpt-4 still finds solutions with a comparable quality to heuristic algorithms, while both text-bison and gpt-3.5-turbo get stuck at local optima with up to 20× worse optimality gaps.

我们在表3中呈现了结果。我们随机生成了每个节点数n的5个问题实例。除了测量最优性差距外，在LLM找到最优解的问题上，我们还显示了达到全局最优解所需的优化步数。首先，我们观察到gpt-4在所有问题规模上都明显优于gpt-3.5-turbo和text-bison。具体而言，在较小规模的问题上，gpt-4比其他LLM快约4倍达到全局最优解。在较大规模的问题上，特别是在n = 50的情况下，gpt-4仍然找到了与启发式算法相当质量的解决方案，而text-bison和gpt-3.5-turbo在局部最优解处停滞，最优性差距高达20倍。

On the other hand, the performance of OPRO degrades dramatically on problems with larger sizes. When n = 10, all LLMs find the optimal solutions for every evaluated problem; as the problem size gets larger, the OPRO optimality gaps increase quickly, and the farthest insertion heuristic starts to outperform all LLMs in the optimality gap.

另一方面，OPRO在规模较大的问题上性能急剧下降。当n = 10时，所有的LLM都找到了每个评估问题的最优解；随着问题规模的增加，OPRO的最优性差距迅速增加，最远插入启发式算法开始在最优性差距方面胜过所有LLM。

Limitations. We would like to note that OPRO is designed for neither outperforming the stateof-the-art gradient-based optimization algorithms for continuous mathematical optimization, nor surpassing the performance of specialized solvers for classical combinatorial optimization problems such as TSP. Instead, the goal is to demonstrate that LLMs are able to optimize different kinds of objective functions simply through prompting, and reach the global optimum for some smallscale problems. Our evaluation reveals several limitations of OPRO for mathematical optimization. Specifically, the length limit of the LLM context window makes it hard to fit large-scale optimization problem descriptions in the prompt, e.g., linear regression with high-dimensional data, and traveling salesman problems with a large set of nodes to visit. In addition, the optimization landscape of some objective functions are too bumpy for the LLM to propose a correct descending direction, causing the optimization to get stuck halfway. We further elaborate our observed failure cases in Appendix A.

限制。我们需要指出的是，**OPRO的设计既不是为了在连续数学优化方面超越最先进的基于梯度的优化算法，也不是为了超越专门的解决方案，如TSP等经典组合优化问题的性能**。相反，目标是**通过提示来证明LLM能够仅仅通过提示来优化不同类型的目标函数，并且对于一些小规模的问题达到全局最优解**。我们的评估揭示了OPRO在数学优化方面的一些限制。具体而言，**LLM上下文窗口的长度限制使得很难将大规模优化问题的描述适应到提示中**，例如具有高维数据的线性回归，以及要访问大量节点的旅行推销员问题。此外，**一些目标函数的优化景观对于LLM来说太崎岖，以至于无法提出正确的下降方向**，导致优化在中途卡住。我们在附录A中进一步详细阐述了我们观察到的失败案例。

# 4 APPLICATION: PROMPT OPTIMIZATION

Next, we demonstrate the effectiveness of OPRO on prompt optimization, where the objective is to find the prompt that maximizes task accuracy. We first introduce the problem setup, then illustrate the meta-prompt design.

接下来，我们演示 OPRO 在提示优化方面的有效性，其目标是找到最大化任务准确性的提示。 我们首先介绍问题设置，然后说明元提示设计。

## 4.1 PROBLEM SETUP

We focus on prompt optimization for natural language tasks, where both the input and output are in the text format. The task is represented as a dataset with training and test splits, where the training set is used to calculate the training accuracy as the objective value during the optimization process, and we compute the test accuracy on the test set after the optimization finishes. While traditional optimization often requires a decently large training set, our experiment shows that a small number or fraction of training samples (e.g., 3.5% of the training set for GSM8K (Cobbe et al., 2021), 20% for Big-Bench Hard (Suzgun et al., 2022)) is sufficient. The objective function evaluator is an LLM to which the optimized prompt will be applied, and it can be the same or different from the LLM for optimization. We denote the LLM for objective function evaluation as the scorer LLM, and the LLM for optimization as the optimizer LLM.

我们专注于自然语言任务的提示优化，其中输入和输出都以文本格式表示。任务被表示为一个带有训练和测试拆分的数据集，训练集用于在优化过程中计算训练准确性作为目标值，我们在优化完成后在测试集上计算测试准确性。虽然传统的优化通常需要相当大的训练集，但我们的实验表明，少量或部分训练样本（例如GSM8K的训练集的3.5％（Cobbe等人，2021），Big-Bench Hard的20％（Suzgun等人，2022））就足够了。目标函数评估器是一个LLM，优化后的提示将应用于它，它可以与用于优化的LLM相同或不同。我们将用于目标函数评估的LLM表示为得分器LLM，将用于优化的LLM表示为优化器LLM。

The output of the optimizer LLM is an instruction, which is concatenated to the question part of every exemplar and prompts the scorer LLM. We consider the following positions to insert the instruction:

 • Q_begin: the instruction is added before the original question. 

• Q_end: the instruction is added after the original question. 

• A_begin: the instruction is added to the beginning of the scorer LLM output. This is applicable to pretrained LLMs without instruction tuning, where the prompt is formatted as a sequence of QA pairs. 

We exemplify these prompting formats in Appendix B.

优化器LLM的输出是一条指令，它被连接到每个示例的问题部分，并提示得分器LLM。我们考虑以下位置来插入指令：

 • Q_begin：指令添加在原始问题之前。
 • Q_end：指令添加在原始问题之后。
 • A_begin：指令添加到得分器LLM输出的开头。这适用于没有指令调整的预训练LLM，其中提示被格式化为一系列QA对。

我们在附录B中示例了这些提示格式。

## 4.2 META-PROMPT DESIGN

Figure 3 shows an example of the meta-prompt for prompt optimization on GSM8K (Cobbe et al., 2021). More details are as follows.

Figure 3显示了在GSM8K（Cobbe等人，2021）上进行的提示优化的元提示的示例。更多细节如下。

Optimization problem examples. The problem description includes a few examples taken from the training set to demonstrate the task for the generated instructions. For example, from the input-output pair in Figure 3, we can infer this is a math word problem. The input-output pair also demonstrates the position where the generated instruction will be added to, and this is essential for the optimizer LLM to generate instructions of the same style. In each optimization step, we add several (three for example) training examples to the meta-prompt by random sampling the training set or choose the ones the previous instructions fall short of.

优化问题示例。问题描述包括从训练集中选取的几个示例，以演示生成的指令的任务。例如，从图3中的输入-输出对中，我们可以推断这是一个数学文字问题。输入-输出对还展示了生成的指令将被添加到的位置，这对于优化器LLM生成相同风格的指令至关重要。在每个优化步骤中，我们通过随机抽样训练集或选择前一指令不足的示例来将几个（例如三个）训练示例添加到元提示中。

Optimization trajectory. The optimization trajectory includes instructions generated from the past optimization steps, along with their scores. The old instructions and scores are sorted by the score in ascending order. The score is the training accuracy in prompt optimization. We only keep instructions with the highest scores in the meta-prompt in consideration of the LLM context length limit.

优化轨迹。优化轨迹包括从过去的优化步骤生成的指令，以及它们的分数。旧的指令和分数按分数升序排序。在提示优化中，分数是训练准确性。考虑到LLM上下文长度的限制，我们只保留分数最高的指令在元提示中。

Meta-instructions. We also add meta-instructions: the instructions to the optimizer LLM that explain the optimization goal and instruct the model how to use the above information. The meta-instructions may also specify the desired generated instruction format for easier parsing.

元指令。我们还添加元指令：这些是用来解释优化目标并指导模型如何使用上述信息的指令，提供给优化器LLM。元指令也可以指定所需的生成指令格式，以便更容易解析。

# 5 PROMPT OPTIMIZATION EXPERIMENTS

We present the evaluation results for prompt optimization in this section. Our experiments demonstrate that OPRO brings a significant performance gain across the board, with different combinations of LLMs as the optimizer and the scorer.

我们在本节中介绍即时优化的评估结果。 我们的实验表明，通过将 LLM 的不同组合作为优化器和评分器，OPRO 带来了全面的显着性能提升。

## 5.1 EVALUATION SETUP

Models. The LLMs we use as the optimizer and the scorer are:

• Optimizer LLM: Pre-trained PaLM 2-L (Anil et al., 2023), instruction-tuned PaLM 2-L (denoted PaLM 2-L-IT), text-bison, gpt-3.5-turbo, and gpt-4. 

• Scorer LLM: Pre-trained PaLM 2-L and text-bison.

With pre-trained PaLM 2-L as the scorer, the optimizer LLM generates A_begin instructions. Since text-bison has been instruction-tuned, the optimizer LLM generates Q_begin and Q_end instructions when text-bison is used as the scorer.

使用预训练的 PaLM 2-L 作为评分器，优化器 LLM 生成 A_begin 指令。 由于 text-bison 已进行指令调整，因此当使用 text-bison 作为评分器时，优化器 LLM 会生成 Q_begin 和 Q_end 指令。

Benchmarks. Our primary evaluation benchmarks are GSM8K (Cobbe et al., 2021) and Big-Bench Hard (BBH) (Suzgun et al., 2022). GSM8K is a benchmark of grade school math word problems with 7,473 training samples and 1,319 test samples, where chain-of-thought prompting (Wei et al., 2022) and the zero-shot instruction “Let’s think step by step.” (Kojima et al., 2022) have drastically improved the performance over the standard prompting. BBH is a suite of 23 challenging BIG-Bench tasks (Srivastava et al., 2022) that covers a wide range of topics beyond arithmetic reasoning, including symbolic manipulation and commonsense reasoning. Each task contains up to 250 examples in total.

基准。 我们的主要评估基准是 GSM8K（Cobbe 等人，2021）和 Big-Bench Hard (BBH)（Suzgun 等人，2022）。 GSM8K 是小学数学应用题的基准，有 7,473 个训练样本和 1,319 个测试样本，其中有思维链提示（Wei et al., 2022）和零样本指令“让我们一步一步思考”。 （Kojima et al., 2022）与标准提示相比，显着提高了性能。 BBH 是一套由 23 项具有挑战性的 BIG-Bench 任务组成的套件（Srivastava 等人，2022），涵盖算术推理之外的广泛主题，包括符号操作和常识推理。 每个任务总共包含最多 250 个示例。

To examine the transferability of the optimized instructions, we also evaluate the instructions optimized for GSM8K on two other mathematical reasoning datasets, i.e., MultiArith (Roy & Roth, 2016) and AQuA (Ling et al., 2017).

为了检查优化指令的可转移性，我们还在另外两个数学推理数据集上评估了针对 GSM8K 优化的指令，即 MultiArith (Roy & Roth, 2016) 和 AQuA (Ling et al., 2017)。

Implementation details. We set the temperature to be 0 when evaluating the performance of generated instructions, in which case the scorer LLM greedily decodes. Unless otherwise specified, we set the default temperature to be 1.0 for optimizer LLMs to generate diverse and creative instructions. At each optimization step, we prompt the optimizer LLM with the meta-prompt 8 times to generate 8 instructions, then we add these instructions with their training scores to the optimization trajectory in the meta-prompt. Our meta-prompt at each step contains the best 20 instructions so far and 3 randomly picked exemplars from the training set. We study the effect of different hyperparameters in ablation studies (Section 5.3). Appendix C.2 presents the full meta-prompts for different optimizer LLMs.

实施细节。 在评估生成指令的性能时，我们将温度设置为 0，在这种情况下，记分器 LLM 会贪婪地解码。 除非另有说明，我们将默认温度设置为 1.0，以便优化器 LLM 生成多样化且富有创意的指令。 在每个优化步骤中，我们使用元提示提示优化器 LLM 8 次以生成 8 条指令，然后将这些指令及其训练分数添加到元提示中的优化轨迹中。 我们每一步的元提示都包含迄今为止最好的 20 条指令以及从训练集中随机挑选的 3 个示例。 我们研究了消融研究中不同超参数的影响（第 5.3 节）。 附录 C.2 提供了不同优化器 LLM 的完整元提示。

## 5.2 MAIN RESULTS

We show prompt optimization curves on GSM8K and two BBH tasks in this section. The curves on other BBH tasks are deferred to Appendix D, and the tables containing all accuracy numbers are in Appendix E.

我们在本节中展示了 GSM8K 和两个 BBH 任务的即时优化曲线。 其他 BBH 任务的曲线请参见附录 D，包含所有准确度数字的表格位于附录 E 中。

### 5.2.1 GSM8K

For prompt optimization, we randomly sample 3.5% examples from the GSM8K training set. The same subset is used throughout optimization, so that the task accuracies computed at intermediate optimization steps are approximations of the training accuracy on all 7,473 training examples. This balances the evaluation cost with the generalization performance. After the optimization procedure finishes, we evaluate the found instructions on the entire GSM8K test set.

对于提示优化，我们从GSM8K培训集中随机抽取3.5%的示例。在整个优化过程中，使用相同的子集，因此在中间优化步骤计算的任务准确性是对所有7,473个培训示例的近似。这样可以平衡评估成本和泛化性能。优化程序完成后，我们在整个GSM8K测试集上评估找到的指令

Figure 1(a) in Section 1 shows prompt optimization curves with pre-trained PaLM 2-L as scorer and PaLM 2-L-IT as optimizer, and the initial instruction is “Let’s solve the problem” with a (approximated, and same below) training accuracy of 60.5. We observe that the optimization curve shows an overall upward trend with several leaps throughout the optimization process, for example:

第1节中的图1(a)显示了使用预训练的**PaLM 2-L作为评分器**和PaLM 2-L-IT作为优化器进行提示优化的曲线，初始指令是“让我们一起解决这个问题”，训练准确度约为60.5（近似值，以下同样）。我们观察到优化曲线在整个优化过程中总体呈上升趋势，中间有几次跃升，例如：

• “Let’s think carefully about the problem and solve it together.” at Step 2 with the training accuracy 63.2;

• “Let’s break it down!” at Step 4 with training accuracy 71.3; 

• “Let’s calculate our way to the solution!” at Step 5 with training accuracy 73.9; 

• “Let’s do the math!” at Step 6 with training accuracy 78.2.

- 在第2步时使用“让我们仔细考虑这个问题，一起解决它。”，训练准确度为63.2；
- 在第4步时使用“让我们把它分解！”，训练准确度为71.3；
- 在第5步时使用“让我们通过计算找到解决方案！”，训练准确度为73.9；
- 在第6步时使用“让我们做数学！”，训练准确度为78.2。

The optimization curves also generally show a decrease of the variance among the accuracies of instructions generated at each step, indicating that the optimizer LLM generates distributionally better instructions throughout the optimization.

优化曲线还通常显示出在每个步骤生成的指令准确性之间的方差减小，表明优化器LLM在整个优化过程中生成了在分布上更好的指令。

Next, we present the results of generating Q_begin instructions with the text-bison scorer and the PaLM 2-L-IT optimizer, starting from an empty instruction with a 57.1 training accuracy. The optimization curve in Figure 4(a) shows a similar upward trend, during which a few leaps in the training accuracy include:

• “Solve the following problems using the given information.” at Step 2 with training accuracy 59.8;

• “Solve the following problems by applying the given information and using the appropriate mathematical operations.” at Step 3 with training accuracy 64.0;

• “Let’s read the problem carefully and identify the given information. Then, we can create an equation and solve for the unknown variable.” at Step 4 with training accuracy 67.0;

• “I’m always down for solving a math word problem together. Just give me a moment to read and understand the problem. Then, I’ll create an equation that models the problem, which I’ll solve for the unknown variable. I also may or may not use some helpful diagrams or visuals to understand the problem. Lastly, be sure to allow me some time to carefully check my work before submitting any responses!” at Step 29 with training accuracy 70.1.

接下来，我们展示了使用text-bison评分器和PaLM 2-L-IT优化器从一个空白指令开始生成Q_begin指令的结果，训练准确度为57.1。 图4(a)中的优化曲线显示了一个类似的上升趋势，其中训练准确度的几次飞跃包括：

- 在第2步训练准确度为59.8的情况下，“使用给定信息解决以下问题。”
- 在第3步，训练准确度为64.0的情况下，“通过应用给定信息并使用适当的数学运算来解决以下问题。”
- 在第4步，训练准确度为67.0的情况下，“让我们仔细阅读问题并确定给定的信息。然后，我们可以创建一个方程，解出未知变量。”
- 在第29步，训练准确度为70.1的情况下，“我总是愿意一起解决数学题。只需给我一点时间来阅读和理解问题。然后，我会创建一个模拟问题的方程，然后解出未知变量。我也可能会使用一些有用的图表或可视化工具来理解问题。最后，请务必给我一些时间仔细检查我的工作，然后再提交任何答案！”。

Note a leap in an optimization curve doesn’t always correspond to a much better instruction being discovered; instead, it can be due to a large qualitative improvement of all 8 generated instructions in this step. The latter usually happens several steps after the former: after a much better instruction is discovered in one step, the meta-prompt gradually gets rid of worse instructions in the latter steps by generating instructions similar to the much-better one. The top instructions kept in the meta-prompt gradually improves in this procedure. At a point when the meta-prompt only triggers higher quality instructions, the leap happens.

请注意，优化曲线中的飞跃并不总是对应于发现了更好的指令；相反，它可能是由于在该步骤中所有8个生成的指令都有很大的定性改进。后者通常发生在前者之后的几个步骤：在一个步骤中发现了一个更好的指令后，通过生成与更好的指令相似的指令，元提示逐渐摆脱了在后续步骤中生成的较差指令。在此过程中，保留在元提示中的顶部指令逐渐改善。当元提示只触发更高质量的指令时，飞跃会发生。

Finally, Figure 4(b) shows that the pre-trained PaLM 2-L can also serve as the optimizer LLM and improve its own prediction performance. Different from other optimizer LLMs that are instructiontuned, the pre-trained PaLM 2-L performs better when the prompt is formatted in a few-shot manner. Therefore, we include two initial instructions to start the optimization: the empty instruction (with a training accuracy 32.2) and “The answer is” (with a training accuracy 33.3). See Figure 19 in Appendix C for the meta-prompt format. The generated instructions follow the same style as “The answer is”: most instructions are also phrases suitable as the prefix of a sentence, like “Here you go:” (generated at Step 11 with training accuracy 61.3) and “Let’s do it:” (generated at Step 13 with training accuracy 75.1).

最后，图4(b)显示，预先训练的 PaLM 2-L 也可以充当优化器 LLM，并提高其自身的预测性能。与其他经过指令调整的优化器 LLM 不同，当提示以 few-shot 方式格式化时，预训练的 PaLM 2-L 表现更好。因此，我们包含了两个初始指令来启动优化过程：空指令（训练精度为32.2）和“The answer is”（训练精度为33.3）。有关元提示格式，请参阅附录C中的图19。生成的指令遵循与“The answer is”相同的样式：大多数指令也是适合作为句子前缀的短语，例如“Here you go:”（在第11步生成，训练精度为61.3）和“Let’s do it:”（在第13步生成，训练精度为75.1）。

Table 4 summarizes top instructions found on GSM8K with different scorer and optimizer LLMs. We observe that:

• The styles of instructions found by different optimizer LLMs vary a lot: PaLM 2-L-IT and text-bison ones are concise, while GPT ones are long and detailed. 

• Although some top instructions contain the “step-by-step” phrase, most others achieve a comparable or better accuracy with different semantic meanings.

表4总结了在 GSM8K 上使用不同的评分器和优化器 LLM 找到的顶级指令。我们观察到：

• 不同优化器 LLM 找到的指令样式差异很大：PaLM 2-L-IT 和 text-bison 的指令都很简洁，而 GPT 的指令则很长，详细。

• 尽管一些顶级指令包含“逐步”短语，但大多数其他指令通过不同的语义含义实现了相当或更好的准确性。

### 5.2.2 BBH

On BBH, the optimization starts from an empty string as the initial instruction by default. The instructions are placed at A_begin when the scorer is PaLM 2-L, and at Q_begin when the scorer is text-bison. For each task, we utilize a subset of 20% examples for prompt optimization, and the rest examples are for testing. We show experimental results on more variants of the instruction position and initialization in Appendix E.

在 BBH 上，优化默认从空字符串作为初始指令开始。 当记分器是 PaLM 2-L 时，指令放置在 A_begin 处；当记分器是 text-bison 时，指令放置在 Q_begin 处。 对于每个任务，我们利用 20% 的示例子集进行即时优化，其余示例用于测试。 我们在附录 E 中展示了指令位置和初始化的更多变体的实验结果。

Figure 5 visualizes the per-task accuracy difference on all 23 BBH tasks compared to the instruction “Let’s think step by step.” (Kojima et al., 2022) and the empty instruction, and we present the concrete accuracies in Table 7 of Appendix E. We show that the instructions found by OPRO outperform “Let’s think step by step.” on almost all tasks by a large margin: our instructions outperform by over 5% on 19/23 tasks with the PaLM 2-L scorer, and on 15/23 tasks with the text-bison scorer. Our prompt optimization algorithm also improves instructions from the empty starting point by over 5% on most tasks: 20/23 with the PaLM 2-L scorer and 15/23 with the text-bison scorer.

图 5 直观地显示了所有 23 个 BBH 任务与“让我们一步一步思考”的指令相比，每个任务的准确性差异。 （Kojima et al., 2022）和空指令，我们在附录 E 的表 7 中给出了具体的准确性。我们表明 OPRO 找到的指令优于“让我们一步一步思考”。 在几乎所有任务上都有很大优势：我们的指令在使用 PaLM 2-L 评分器的 19/23 任务上以及使用 text-bison 评分器的 15/23 任务上表现优于 5% 以上。 我们的即时优化算法还在大多数任务中将空起点的指令改进了 5% 以上：使用 PaLM 2-L 记分器为 20/23，使用 text-bison 记分器为 15/23。

Similar to GSM8K, we observe upward trends in optimization curves on almost all BBH tasks, as shown in Figure 6. See Figure 21 and 22 in Appendix D for more curves on other BBH tasks.

与 GSM8K 类似，我们观察到几乎所有 BBH 任务的优化曲线呈上升趋势，如图 6 所示。有关其他 BBH 任务的更多曲线，请参阅附录 D 中的图 21 和 22。

We next show some examples of instructions found through the course of optimization. On the task ruin_names, starting from the empty instruction (with 64.0 training accuracy), with the text-bison scorer and the PaLM 2-L-IT optimizer, the following instructions are generated:

接下来我们将展示一些通过优化过程找到的指令示例。 在任务 destroy_names 上，从空指令（训练精度为 64.0）开始，使用 text-bison 评分器和 PaLM 2-L-IT 优化器，生成以下指令：

• “Consider the following when editing artist or movie names humorously:” at Step 1 with training accuracy 72.0;

• “When making humorous edits of artist or movie names, you can change one or more letters or even create puns by adding new words that sound similar.” at Step 18 with training accuracy 80.0;

• “We can make humorous edits of artist/movie names by changing letters to create new words that are similar in sound but have different meanings. For example, The Police can be changed to The Polite, The Abyss can be changed to Toe Abyss, and Schindler’s List can be changed to Schindler’s Lost.” at Step 38 with training accuracy 82.0.

• “幽默地编辑艺术家或电影名称时请考虑以下事项：”第 1 步，训练准确度为 72.0；

• “在对艺术家或电影名称进行幽默编辑时，您可以更改一个或多个字母，甚至可以通过添加听起来相似的新单词来创造双关语。” 在第 18 步，训练精度为 80.0；

• “我们可以通过更改字母来创建发音相似但含义不同的新单词，对艺术家/电影名称进行幽默编辑。 例如，《警察》可以改为《礼貌》，《深渊》可以改为《脚趾深渊》，《辛德勒名单》可以改为《辛德勒迷失》。” 在第 38 步，训练准确度为 82.0。

Although the above instructions are semantically similar, a paraphrase by the optimizer LLM offers a notable accuracy improvement. We further highlight this observation in Section 5.2.3.

Below are some instructions generated when performing prompt optimization on temporal_sequences, starting from the empty instruction (with the training accuracy of 64.0):

尽管上述指令在语义上相似，但优化器 LLM 的解释提供了显着的准确性改进。 我们在第 5.2.3 节中进一步强调了这一观察结果。

下面是对temporal_sequences进行提示优化时生成的一些指令，从空指令开始（训练精度为64.0）：

• “To solve this problem, we need to first identify the time period when the person was not seen doing anything else. Then, we need to check if the place they went to was open during that time period. If it was, then that is the time period when they could have gone to that place.” at Step 2 with training accuracy 42.0;

• “To find the time period when a person could have gone to a place, identify the time periods when they were not seen doing anything else and the place was open. If there are multiple time periods that match these criteria, then the person could have gone to the place during any of these time periods.” at Step 18 with training accuracy 54.0;

• “To determine the possible time period when a person went to a place, first identify all the time periods when the person was not seen doing anything else and the place was open. Then, rule out any time periods during which the person was seen doing something else. The remaining time periods are the possible times when the person could have gone to the place.” at Step 41 with training accuracy 72.0.

• “为了解决这个问题，我们首先需要确定没有看到此人做任何其他事情的时间段。 然后，我们需要检查他们去的地方在该时间段内是否开放。 如果是的话，那就是他们可以去那个地方的时间段了。” 第 2 步，训练精度为 42.0；

• “要找到一个人可能去某个地方的时间段，请确定他们没有被看到做任何其他事情且该地方开放的时间段。 如果有多个时间段符合这些条件，那么该人可能在这些时间段中的任何一个时间段内去过该地点。” 在第 18 步，训练精度为 54.0；

• “要确定一个人去过某个地方的可能时间段，首先要确定没有看到该人做任何其他事情且该地方开放的所有时间段。 然后，排除任何被看到该人在做其他事情的时间段。 剩余的时间段是该人可能去过该地点的时间。” 第 41 步，训练准确度为 72.0。

Table 5 presents the best instructions generated on movie_recommendation, ruin_names, and temporal_sequences tasks with different combinations of the optimizer and the scorer LLMs. Again, different optimizer LLMs produce instructions of different styles. See Appendix E for results on more BBH tasks.

表 5 展示了使用优化器和评分器 LLM 的不同组合在 movie_recommendation、ruu_names 和 temporal_sequences 任务上生成的最佳指令。 同样，不同的优化器法学硕士会产生不同风格的指令。 有关更多 BBH 任务的结果，请参阅附录 E。

### 5.2.3 SEMANTICALLY SIMILAR INSTRUCTIONS MAY ACHIEVE DRASTICALLY DIFFERENT ACCURACIES 语义相似的指令可能会获得截然不同的准确度

One challenge of prompt optimization is the sensitivity of model performance to subtle changes in the instruction. For example, with the PaLM 2-L scorer on the GSM8K test set, “Let’s think step by step.” achieves accuracy 71.8, “Let’s solve the problem together.” has accuracy 60.5, while the accuracy of “Let’s work together to solve this problem step by step.” is only 49.4, although it is the semantic combination of the two upper instructions. This behavior increases both the variance across single-step instructions and the oscillation during optimization, and motivates us to generate multiple instructions at each step to improve the optimization stability.

即时优化的挑战之一是模型性能对指令中细微变化的敏感性。 例如，对于 GSM8K 测试集上的 PaLM 2-L 评分器，“让我们一步一步思考。” 达到了71.8的准确率，“让我们一起解决这个问题。” 准确率60.5，而“让我们一起来一步步解决这个问题”的准确率。 只有49.4，虽然是两条上位指令的语义组合。 这种行为增加了单步指令之间的方差和优化期间的振荡，并激励我们在每一步生成多个指令以提高优化稳定性。

### 5.2.4 TRANSFERABILITY OF FOUND INSTRUCTIONS 已找到指令的可转移性

We assess the transferability of found prompts to different datasets of the same domain, where we evaluate the top instructions found for GSM8K on two more math reasoning benchmarks MultiArith (Roy & Roth, 2016) and AQuA (Ling et al., 2017). Table 6 shows that our optimized prompts also outperform baseline prompts with different scorer LLMs on these two benchmarks.

我们评估了找到的提示到同一域的不同数据集的可转移性，其中我们在另外两个数学推理基准 MultiArith (Roy & Roth, 2016) 和 AQuA (Ling et al., 2017) 上评估了 GSM8K 找到的顶级指令。 表 6 显示，在这两个基准测试中，我们的优化提示也优于具有不同计分器 LLM 的基线提示。

### 5.3 ABLATION STUDIES

We use text-bison as the scorer and PaLM 2-L as the optimizer for all ablation studies. The tasks we evaluate are GSM8K (math reasoning) and BBH sports_understanding (non-math reasoning).

Meta-prompt design. The meta-prompt design is crucial in achieving good prompt optimization performance. We investigate the following core design choices:

我们使用 text-bison 作为评分器，使用 PaLM 2-L 作为所有消融研究的优化器。 我们评估的任务是 GSM8K（数学推理）和 BBH sports_understanding（非数学推理）。

元提示设计。 元提示设计对于实现良好的提示优化性能至关重要。 我们研究了以下核心设计选择：

• The order of the previous instructions. We compare the following options: (1) from lowest to highest (our default setting); (2) from highest to lowest; (3) random. Figures 7(a) and 7(b) show that the default setting achieves better final accuracies and converges faster. One hypothesis is that the optimizer LLM output is affected more by the past instructions closer to the end of the meta-prompt. This is consistent with the recency bias observed in Zhao et al. (2021), which states that LLMs are more likely to generate tokens similar to the end of the prompt.

• 先前指令的顺序。我们比较以下选项：（1）从最低到最高（我们的默认设置）；（2）从最高到最低；（3）随机。图7(a)和7(b)显示，默认设置实现更好的最终准确性并更快地收敛。一个假设是，优化器 LLM 的输出受到更接近元提示末尾的过去指令的影响更大。这与Zhao等人（2021）观察到的最近性偏差一致，该偏差表明LLMs更有可能生成类似于提示末尾的标记。

• The effect of instruction scores. In terms of how to present the accuracy scores, we compare three options: (1) rounding the accuracies to integers, which is equivalent to bucketizing the accuracy scores to 100 buckets (our default setting); (2) bucketizing the accuracies to 20 buckets; (3) not showing the accuracies, only showing the instructions in the ascending order. Figures 7(c) and 7(d) show that the accuracy scores assists the optimizer LLM in better understanding the quality difference among previous instructions, and thus the optimizer LLM proposes better new instructions that are similar to the best ones in the input optimization trajectory.

• 指令得分的影响。就如何呈现准确性得分而言，我们比较三个选项：（1）将准确性四舍五入为整数，这等同于将准确性得分分桶为100个桶（我们的默认设置）；（2）将准确性得分分桶为20个桶；（3）不显示准确性，仅按升序显示指令。图7(c)和7(d)显示，准确性得分有助于优化器 LLM 更好地理解先前指令之间的质量差异，因此优化器 LLM 提出更好的新指令，这些新指令与输入优化轨迹中的最佳指令相似。

• The effect of exemplars. We compare three options: (1) showing 3 exemplars from the task (default); (2) showing 10 exemplars from the task; (3) no exemplars. Figures 7(e) and 7(f) show that presenting exemplars in the meta-prompt is critical, as it provides information on what the task looks like and helps the optimizer model phrase new instructions better. However, more exemplars do not necessarily improve the performance, as a few exemplars are usually sufficient to describe the task. In addition, including more exemplars results in a longer meta-prompt with a dominating exemplar part, which may distract the optimizer LLM from other important components like the optimization trajectory.

• 示例的影响。我们比较三个选项：（1）显示来自任务的3个示例（默认）；（2）显示任务的10个示例；（3）没有示例。图7(e)和7(f)显示，将示例呈现在元提示中至关重要，因为它提供了关于任务的外观以及帮助优化模型更好地表达新指令的信息。然而，更多的示例不一定会提高性能，因为通常几个示例就足以描述任务。此外，包括更多的示例会导致更长的元提示，以主导示例部分，这可能会分散优化器 LLM 对其他重要组件（如优化轨迹）的注意力。

The number of generated instructions per step. Computing a mini-batch of gradients reduces the variance of a stochastic gradient descent procedure. Similarly, generating multiple instructions in each step improves the optimization stability with LLMs. On the other hand, to achieve better performance with a fixed budget for the number of instructions to evaluate, the number of per-step instructions should not be too large, so as to allow more optimization steps to incorporate richer information of past instructions with their accuracies. Taking both aspects into consideration, Figure 8 compares the optimization performance of sampling 1 / 2 / 4 / 8 (default) / 16 instructions in each step, showing that sampling 8 instructions at each step overall achieves the best performance.

每步生成的指令数量。计算梯度的小批量减少了随机梯度下降过程的方差。类似地，每一步生成多个指令可以提高LLMs的优化稳定性。另一方面，为了在固定的指令数量评估预算下获得更好的性能，每步指令的数量不应太多，以便允许更多的优化步骤吸收过去指令和它们的准确性的更丰富信息。综合考虑这两个方面，图8比较了在每一步中抽样1/2/4/8（默认值）/16个指令的优化性能，结果显示每步抽样8个指令总体上获得了最佳性能。

Starting point. We study the effect of different initial instructions for prompt optimization. Our default setting is to start from an empty string when the scorer LLM is (instruction-tuned) text-bison, and to start from either the empty string (on BBH tasks) or “Let’s solve the problem.” (on GSM8K) with instruction position A_begin when the scorer LLM is the (pre-trained) PaLM 2-L.

出发点。我们研究了用于提示优化的不同初始指令的影响。我们的默认设置是在评分器LLM是（经过指令调整的）text-bison时从空字符串开始，并且在评分器LLM是（预先训练的）PaLM 2-L时，从空字符串（在BBH任务上）或“让我们解决这个问题。”（在GSM8K上）开始，指令位置是A_begin。

Figure 9(a) shows the performance of text-bison as the scorer LLM with 3 options of initial instructions: (1) the empty string; (2) “Solve the following problem.”; or (3) “Solve the following problem.” and “Let’s solve the problem.”. We observe that the accuracies do not differ much with different starting points. Interestingly, the styles of the generated instructions are also similar. For example, most of the generated instructions starting from (1) and (2) contain the phrase “solve this problem”, like “Let’s work together to solve this problem.” in Step 4 with training accuracy 64.8 from (1), and “Let’s solve the following problems using the given information.” in Step 3 with training accuracy 62.8 from (2).

Figure 9(a)显示了text-bison作为评分器LLM时初始指令的性能，有3个选项：(1) 空字符串；(2) “解决以下问题。”；或(3) “解决以下问题。”和“让我们解决这个问题。”我们观察到，不同的起始点对准确性影响不大。有趣的是，生成的指令的风格也相似。例如，从(1)和(2)开始的大多数生成的指令都包含短语“解决这个问题”，比如从(1)开始的Step 4中的“让我们一起解决这个问题。”，训练准确度为64.8，以及从(2)开始的Step 3中的“让我们使用给定的信息解决以下问题。”，训练准确度为62.8。

Figure 9(b) presents the results of of PaLM 2-L as the scorer LLM with the following options of initial instructions: (1) “Let’s solve the problem.”; (2) the empty string; or (3) “Let’s think step by step.”. We notice that the performance differs much more with different initial instructions, especially at the beginning of the optimization. Specifically, starting from (1) leads to better generated instructions than (2) in the first 30 steps, while the instructions optimized from both (1) and (2) are worse than (3) throughout. A similar observation holds when using PaLM 2-L as scorer and gpt-3.5-turbo as optimizer for BBH tasks, by comparing the results starting from the empty string (Appendix E.2) and from “Let’s solve the problem.” (Appendix E.3). Taking a closer look into the optimization process of (2), we find that although both “solve the problem” and “step by step” show up in generated instructions at Step 5, it takes the optimizer LLM more steps to get rid of worse instructions presented in the meta-prompt when starting from instructions with lower accuracies. Therefore, one direction for future work is to accelerate convergence from weaker starting points.

图9(b)展示了PaLM 2-L作为评分器LLM的结果，有以下初始指令选项：(1)“让我们解决这个问题。”；(2)空字符串；或(3)“让我们逐步思考。”。我们注意到，不同的初始指令在优化的开始阶段会有更大的性能差异。具体而言，从(1)开始在前30个步骤内会导致比(2)更好的生成指令，而从(1)和(2)都优化的指令在整个过程中都比(3)差。当使用PaLM 2-L作为评分器，gpt-3.5-turbo作为BBH任务的优化器时，通过比较从空字符串开始(Appendix E.2)和从“让我们解决这个问题。”开始(Appendix E.3)的结果，我们发现相似的观察结果。仔细观察(2)的优化过程，我们发现虽然“解决问题”和“逐步思考”都出现在第5步生成的指令中，但当从准确度较低的初始指令开始时，优化器LLM需要更多的步骤来摆脱出现在元提示中的较差指令。因此，未来工作的一个方向是加快从较弱的起始点收敛的速度。

Diversity per step. We evaluate the following temperatures of the optimizer LLM: {0.0, 0.5, 1.0 (default), 1.5, 2.0}. Figure 10 shows the default temperature 1.0 achieves the best performance. Specifically, optimizations with smaller temperatures (0.0 and 0.5) lack exploration and thus creativity, and the optimizer LLM often gets stuck at the same instruction for tens of steps, resulting in flat optimization curves. On the other hand, with larger temperatures (1.5 and 2.0), the optimizer LLM more often ignores the trajectory of previous instructions presented in the meta-prompt and thus lacks exploitation, therefore the optimization curve does not have a steady upward trend.

每步的多样性。我们评估了优化器LLM的以下温度：{0.0, 0.5, 1.0 (默认), 1.5, 2.0}。图10显示默认温度1.0实现了最佳性能。具体来说，**较小的温度（0.0和0.5）的优化缺乏探索和创造力**，优化器LLM经常在同一个指令上停滞了几十个步骤，导致平坦的优化曲线。另一方面，**较大的温度（1.5和2.0）下，优化器LLM更容易忽略元提示中呈现的先前指令的轨迹，因此缺乏利用**，因此优化曲线没有稳定的上升趋势。

# 6 RELATED WORK

Prompt optimization. Prior works have developed soft prompt-tuning methods that optimize the prompt represented as task-specific continuous vectors (Lester et al., 2021; Li & Liang, 2021; Liu et al., 2021; Qin & Eisner, 2021), as well as performing discrete prompt optimization by gradient-guided search (Shin et al., 2020; Wen et al., 2023; Gao et al., 2020; Chen et al., 2023d) and reinforcement learning (Deng et al., 2022; Zhang et al., 2022). These approaches become inapplicable when there is only API access to the LLM. Other works designed edit-based approaches for gradient-free prompt optimization (Xu et al., 2022; Prasad et al., 2022), where the editing can be done with humandefined operations (e.g., swapping two phrases) (Prasad et al., 2022) or language models (e.g., back translation) (Xu et al., 2022). Some recent works investigate LLMs for prompt optimization (Zhou et al., 2022b; Pryzant et al., 2023; Xu et al., 2023). Specifically, APE (Zhou et al., 2022b) first uses the LLM to generate initial instructions. Afterwards, APE selects top instructions with the highest accuracies, then prompts the LLM with each individual instruction to generate a semantically similar variant of the initial instruction. APO (Pryzant et al., 2023) in each step instructs the LLM to produce text feedback on how to update an old instruction. Different from edit-based approaches, the optimizer LLM in our work directly generates new instructions at each optimization step, and the optimizer LLM is merely asked to improve the task accuracy without being required to imitate past instructions. Compared to Zhou et al. (2022b) and Pryzant et al. (2023), our optimization process incorporates the past generated instructions with their scores in the meta-prompt, enabling the optimizer LLM to discover common patterns of high-quality instructions.

提示优化。之前的工作已经开发了软提示调整方法，通过优化以任务特定的连续向量表示的提示（Lester等，2021；Li＆Liang，2021；Liu等，2021；Qin＆Eisner，2021），以及通过渐变引导搜索（Shin等，2020；Wen等，2023；Gao等，2020；Chen等，2023d）和强化学习（Deng等，2022；Zhang等，2022）进行离散提示优化。当只能通过API访问LLM时，这些方法不适用。其他工作设计了基于编辑的无渐变提示优化方法（Xu等，2022；Prasad等，2022），其中编辑可以通过人定义的操作（例如，交换两个短语）（Prasad等，2022）或语言模型（例如，回译）（Xu等，2022）来完成。一些最近的工作调查了LLMs用于提示优化（Zhou等，2022b；Pryzant等，2023；Xu等，2023）。具体来说，APE（Zhou等，2022b）首先使用LLM生成初始指令。然后，APE选择具有最高准确性的顶级指令，然后用每个单独的指令提示LLM生成初始指令的语义相似变体。APO（Pryzant等，2023）在每一步中指示LLM生成有关如何更新旧指令的文本反馈。与基于编辑的方法不同，我们的工作中的优化器LLM直接在每个优化步骤中生成新的指令，优化器LLM仅被要求提高任务准确性，而不需要模仿过去的指令。与Zhou等（2022b）和Pryzant等（2023）相比，我们的优化过程将以前生成的指令与它们的得分合并到元提示中，使优化器LLM能够发现高质量指令的常见模式。

Prompting with natural language feedback. A recent line of work investigates approaches to improve the LLM performance by prompting with natural language feedback to revise the model output, which has shown effectiveness in reducing harmful LLM outputs (Bai et al., 2022; Ganguli et al., 2023), improving reasoning (Shinn et al., 2023; Madaan et al., 2023) and code generation performance (Chen et al., 2023e; Olausson et al., 2023; Shinn et al., 2023; Chen et al., 2023b), dialogue applications (Nair et al., 2023; Madaan et al., 2023; Yuan et al., 2023), and so on (Kim et al., 2023; Wang et al., 2023). Specifically, Yuan et al. (2023) develops a human-in-the-loop framework for deriving system-level feedback from a collection of instance-level feedback, which is then used for refining data. In our work, the optimizer LLM utilizes the optimization trajectory in the prompt, which implicitly requires the LLM to summarize the common characteristics among solutions with similar scores. We consider incorporating explicit natural language feedback on generated solutions for later optimization steps as future work.

使用自然语言反馈的提示。最近的一系列工作研究了通过使用自然语言反馈来修正模型输出来提高LLM性能的方法，已经证明在减少有害的LLM输出（Bai等，2022；Ganguli等，2023）、提高推理（Shinn等，2023；Madaan等，2023）和代码生成性能（Chen等，2023e；Olausson等，2023；Shinn等，2023；Chen等，2023b）、对话应用（Nair等，2023；Madaan等，2023；Yuan等，2023）等方面非常有效。具体来说，Yuan等（2023）开发了一个人在循环中的框架，用于从一系列实例级反馈中获取系统级反馈，然后用于改进数据。在我们的工作中，优化器LLM利用提示中的优化轨迹，这隐含地要求LLM总结具有相似得分的解决方案之间的共同特征。我们考虑在后续的优化步骤中将显式自然语言反馈纳入生成的解决方案，作为未来的工作。

Tuning language models for optimization. Some previous works tune or prompt language models to behave as mutation and crossover operators in evolutionary algorithms. Meyerson et al. (2023) utilizes language models with few-shot exemplars to propose evolutionary cross-overs on tasks such as image and code generation. In Lehman et al. (2022), the large language model trained on code diff generation is used as the mutation operator, and they further design a fine-tuning method to improve performance in the Sodarace domain for robot simulation. EvoPrompting (Chen et al., 2023a) uses large language models to evolve neural network architectures, where they combine evolutionary search with soft prompt tuning. With respect to taking the trajectory as the input for optimization, OptFormer (Chen et al., 2022) trains a transformer model on large collections of hyperparameter optimization data. On the other hand, our work performs optimization solely by prompting without additional training.

调整语言模型以进行优化。一些先前的工作调整或提示语言模型，使其行为类似于进化算法中的突变和交叉算子。Meyerson等人（2023）利用具有少量示例的语言模型提出了在图像和代码生成等任务上的进化交叉操作。在Lehman等人（2022）中，训练用于代码差异生成的大型语言模型被用作突变算子，并且他们进一步设计了一种微调方法，以提高机器人模拟领域的Sodarace性能。EvoPrompting（Chen等人，2023a）使用大型语言模型来演化神经网络架构，其中他们将进化搜索与软提示调整相结合。就以轨迹作为优化输入而言，OptFormer（Chen等人，2022）在大量超参数优化数据上训练了一个Transformer模型。另一方面，我们的工作仅通过提示而无需额外的训练来执行优化。

# 7 CONCLUSION

We embark on employing LLMs as optimizers, where the LLM progressively generates new solutions to optimize an objective function. We first motivate OPRO with linear regression and traveling salesman problems, then proceed to prompt optimization as a concrete application. Our evaluation demonstrates that LLMs have the capacity of gradually improving the generated solutions based on the past optimization trajectory. Interestingly, on small-scale traveling salesman problems, OPRO performs on par with some hand-crafted heuristic algorithms. For prompt optimization, optimized prompts outperform human-designed prompts on GSM8K and Big-Bench Hard by a significant margin, sometimes over 50%.

我们开始利用LLMs作为优化器，其中LLM逐渐生成新的解决方案以优化目标函数。我们首先通过线性回归和旅行推销员问题来激发OPRO的动机，然后继续进行提示优化作为一个具体的应用。我们的评估表明，LLMs有能力根据过去的优化轨迹逐渐改善生成的解决方案。有趣的是，在小规模的旅行推销员问题上，OPRO的性能与一些手工制作的启发式算法相当。对于提示优化，经过OPRO优化的提示在GSM8K和Big-Bench Hard上明显优于人工设计的提示，有时超过50%。

A number of unresolved questions are open for future research on LLMs for optimization. In general, how to reduce the sensitivity to initialization and better balance exploitation with exploration remains a challenge. Specifically, for prompt optimization, one limitation of our current implementation is that the optimizer LLM does not effectively utilize error cases in the training set to infer promising directions to improve the generated instructions. In our experiments, we tried including error cases in the meta-prompt rather than randomly sampling from the training set at each optimization step, but the results are similar, indicating that the error cases alone are not informative enough for the optimizer LLM to grasp the cause of the wrong prediction. Another limitation is that prompt optimization requires a training set to compute the accuracy that guides the optimization process. Currently the training set at least contains tens of samples, so that the optimized prompt does not severely overfit to the training samples. A promising direction is to incorporate richer feedback about the error cases besides the aggregated accuracy, and summarize the key features that distinguish between high-quality and low-quality generated prompts in the optimization trajectory. Such information may inform the optimizer LLM of how to more efficiently improve over the past generated instructions, and potentially further reduce the example set size needed for prompt optimization

未来研究中还有许多未解决的关于LLMs用于优化的问题。总体来说，如何减少对初始化的敏感性以及更好地平衡开发和探索仍然是一个挑战。具体来说，对于提示优化，我们目前实现的一个限制是，优化器LLM未能有效地利用训练集中的错误案例来推断改进生成的指令的有希望的方向。在我们的实验中，我们尝试将错误案例包含在元提示中，而不是在每个优化步骤中随机从训练集中抽样，但结果类似，表明仅有错误案例对于优化器LLM来说并不足够信息丰富，无法掌握错误预测的原因。另一个限制是，提示优化需要一个训练集来计算指导优化过程的准确性。目前，训练集至少包含数十个样本，以使优化的提示不会严重过拟合到训练样本。一个有希望的方向是除了聚合的准确性之外，还将更丰富的关于错误案例的反馈纳入考虑，并总结出区分高质量和低质量生成提示的关键特征在优化轨迹中。这些信息可以告诉优化器LLM如何更有效地改进过去生成的指令，可能进一步减少提示优化所需的示例集大小。