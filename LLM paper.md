LLM paper

# LARGE LANGUAGE MODELS AS OPTIMIZERS  

## Introduction

Optimizers: iterative, initial solution->update solution -> optimize objective function

traditional optimize: define optimization problem and derive update step. need to customize algorithm (because of decision space and the performance landscape, especially for derivative-free optimization)

LLM Optimizers: describe optimization problem in natural language; generate new solutions based on problem description and previously found solutions

## Implementation

OPRO

![image-20231017003044119](D:\coursefile\LLM\typorapic\image-20231017003044119.png)

meta prompt:

1\. Optimization problem description: meta-instructions

problem itself, objective function, solution constraints

2\. Previous prompts with accuracies:

ascending order

LLM can identify **similarities** between **solutions** **with** **high** **scores**, then generate potentially better ones

3.

Not all solutions achieve high scores and

monotonically improve over prior ones.

Solution: generate multiple solutions each step

4.Exploration-Exploitation trade-off problem

tune LLM sampling temperature (more or less random)

## **Mathmatical** **Optimization**

### Linear Regression:

choose wtrue and btrue 

([10, 20] × [10, 20], “near outside”  and “far outside”)

sample y = wtruex + btrue + ϵ 

start with 5 pairs, prompt new pair 8 times (each time one pair) each iteration

best 20 pairs

**black box optimization(?)** I think it means don't write any formula in meta-prompt, and just input numbers. Otherwise LLM will calculate it

![image-20231017005004705](D:\coursefile\LLM\typorapic\image-20231017005004705.png)

### Results:

number of unique (w, b) pairs < exhaustive search, means LLM can do black-box optimization

compare the numbers and propose a **descent direction**

text-bison and gpt-4 converge faster

ground truth farther, more steps

## TSP:

![image-20231017005228640](D:\coursefile\LLM\typorapic\image-20231017005228640.png)

sampling n nodes with both x and y coordinates in [-100, 100]  

also 5 init, 8 new each iter

compare to Gurobi Optimizer, Nearest Neighbor, Farthest Insertion

