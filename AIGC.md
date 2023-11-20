

### 目录
- [目录](#目录)
- [2 大模型基础](#2-大模型基础)
  - [2.1 构建框架](#21-构建框架)
  - [2.2 预训练模型](#22-预训练模型)
    - [2.2.1 数据](#221-数据)
    - [2.2.2 文本分词](#222-文本分词)
    - [2.2.3 Bert](#223-bert)
    - [2.2.4 GPT](#224-gpt)
    - [2.2.5 BART](#225-bart)
    - [2.2.6 T5](#226-t5)
    - [2.2.7 Unilm](#227-unilm)
    - [2.2.8 GLM](#228-glm)
    - [2.2.9 LLaMA](#229-llama)
  - [2.3 有监督微调](#23-有监督微调)
    - [2.3.1 BitFit](#231-bitfit)
    - [2.3.2 Prompt-Tuning](#232-prompt-tuning)
    - [2.3.3 P-Tuning](#233-p-tuning)
    - [2.3.4 Prefix-Tuning](#234-prefix-tuning)
    - [2.3.5 Lora](#235-lora)
    - [2.3.6 IA3](#236-ia3)
    - [2.3.7 Adapter](#237-adapter)
  - [2.4 强化学习](#24-强化学习)
    - [2.4.1 奖励模型](#241-奖励模型)
    - [2.4.2 RLHF](#242-rlhf)
- [3 多模态](#3-多模态)
  - [3.1 数据](#31-数据)
  - [3.2 多模态模型](#32-多模态模型)
    - [3.2.1 ViT](#321-vit)
    - [3.2.2 CLIP](#322-clip)
    - [3.2.3 Stable Diffusion](#323-stable-diffusion)
    - [3.2.3 ControlNet](#323-controlnet)
    - [3.2.4 Imagen](#324-imagen)
    - [3.2.5 Dreambooth](#325-dreambooth)
- [4 分布式训练](#4-分布式训练)
  - [4.1 概述](#41-概述)
  - [4.2 并行策略](#42-并行策略)
    - [4.2.1 数据并行](#421-数据并行)
    - [4.2.2 模型并行](#422-模型并行)
    - [4.2.3 张量并行](#423-张量并行)
  - [4.3 DeepSpeed](#43-deepspeed)
- [5 大模型应用框架](#5-大模型应用框架)
- [6 NLP 任务](#6-nlp-任务)

### 2 大模型基础

#### 2.1 构建框架

![大模型框架图](./pic/2-1.png "大模型框架图")

#### 2.2 预训练模型

##### 2.2.1 数据

##### 2.2.2 文本分词
- **分词类性**：
  - word 分词， OOV问题
  - character分词，过长，失去了单词的联系
  - Subword：
    "unfortunately" = "un" + "for" + "tun" + "ate" + "ly"
    常用算法：BPE，SentencePiece，WordPiece等
- **BPE**：
  Byte-Pair Encoding。先拆分为character, 然后统计token pair频率，合并， GPT
- **WordPiece**：
  与BPE 很相近，过程类似，但是会考虑单个token的频率，如果单个token的频率很高，也不会合并。如un-able， un 和 able 概率都很高，即使un-able pair很高，也不会合并
- **Unigram**：
  从一个巨大的词汇表出发，再逐渐删除trimdown其中的词汇，直到size满足预定义。初始的词汇表可以采用所有预分词器分出来的词，再加上所有高频的子串。每次从词汇表中删除词汇的原则是使预定义的损失最小。
  训练文档所有词为$x_1, x_2, ..., X_n$, 每个词token的方法是一个集合$S(x_i)$, 当一个词汇表确定时，每个词tokenize的方法集合$S(x_i)$ 就是确定的，每种方法对应一个概率$P(x)$, 依据损失进行删除。
- **SentencePiece**：
  把一个句子看作一个整体，再拆成片段，而没有保留天然的词语的概念。一般地，它把空格space也当作一种特殊字符来处理，再用BPE或者Unigram算法来构造词汇表。与Unigram算法联合使用
  
##### 2.2.3 Bert

    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    https://arxiv.org/pdf/1810.04805.pdf
- pretrain:
  - Masked LM
  - Next Sentence Prediction
    
![Bert](./pic/2/bert.jpg "Bert")
![Bert-2](./pic/2/bert-2.jpg "Bert-2")

##### 2.2.4 GPT
    GPT1：Improving Language Understanding by Generative Pre-Training
    https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf
    GPT2：Language Models are Unsupervised Multitask Learners
    https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
    GPT3：Language Models are Few-Shot Learners
    https://arxiv.org/pdf/2005.14165.pdf
    InstructGPT：Training language models to follow instructions with human feedback
    https://arxiv.org/pdf/2203.02155.pdf
- GPT1:
![GPT1](./pic/2/gpt1.jpg "GPT1")
自回归方式训练，预训练阶段窗口设置，需要主要的是fune-tuning时，损失函数包含预训练损失

- GPT2：
	堆数据，堆网络参数，网络norm层有变化

- GPT3：
	![GPT3](./pic/2/gpt3.jpg "GPT3")
	模型使用spare attention, 175B参数

- InstructGPT
	![GPT3](./pic/2/gpt3.jpg "GPT3")

##### 2.2.5 BART

    BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
    https://arxiv.org/pdf/1910.13461.pdf


  ![BART1](./pic/2/bart.jpg "BART1") 

- Pre-training:
  - Token Masking
  - Token Deletion
  - Text Infilling
  - Sentence Permutation
  - Document Rotation

![BART-2](./pic/2/bart-2.jpg "BART-2")

- Fine-Tuning：
	在做machine translation时，先预训练参数，只更新initialized encoder参数，后完全更新参数。

 ![BART-3](./pic/2/bart-3.jpg "BART-3")

##### 2.2.6 T5

    Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    https://arxiv.org/pdf/1910.10683.pdf

   ![t5](./pic/2/t5.jpg "t5")
添加前缀


##### 2.2.7 Unilm

    Unified Language Model Pre-training for Natural Language Understanding and Generation
    https://arxiv.org/pdf/1905.03197.pdf

##### 2.2.8 GLM

##### 2.2.9 LLaMA

    LLaMA: Open and Efficient Foundation Language Models
    https://arxiv.org/pdf/2302.13971.pdf
	Llama 2: Open Foundation and Fine-Tuned Chat Models
    https://arxiv.org/pdf/2307.09288.pdf

- **LLaMA-1**：
  - decoder only  
  - Pre-normalization
  - SwiGLU activation function
  - Rotary Embeddings

- **LLaMA-2**：

#### 2.3 有监督微调
有监督微调（Supervised Finetuning, SFT）又称指令微调（Instruction Tuning），是指在已经训练好的语言模型的基础上，通过使用有标注的特定任务数据进行进一步的微调，从而使得模型具备遵循指令的能力。经过海量数据预训练后的语言模型虽然具备了大量的“知识”，但是由于其训练时的目标仅是进行下一个词的预测，此时的模型还不能够理解并遵循人类自然语言形式的指令。

![有监督微调图](./pic/3-1.jpg "有监督微调图")

微调技术综述：

	Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning
	https://arxiv.org/pdf/2303.15647.pdf

##### 2.3.1 BitFit

	只调节神经网络的bias参数

![BitFit](./pic/3/bitfit.png "BitFit")

论文：

	BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models
	https://arxiv.org/pdf/2106.10199v2.pdf


##### 2.3.2 Prompt-Tuning

![BitFit](./pic/3/prompt-tuning.png "BitFit")

论文：

	The Power of Scale for Parameter-Efficient Prompt Tuning
	https://arxiv.org/pdf/2104.08691.pdf

算法原理：

$$ \hat{Y}=argmax_YPr_{\theta, {\theta}_p}(Y|[P;X]) $$

- ${\theta}$  model parameters, ${\theta_p}$ prompt 参数

- $Y$ output, a sequence of tokens

- $X$ input, a sequence of tokens

- $P$ prompt, a series of tokens prepended to the input 

当输入是$n$ 个tokens, 表示为${x_1, x_2,...,x_n}$, 模型会将这些input通过embedding层转换为一个矩阵$X_e \in R^{n \times e}$, 这里的$e$ 是embedding space 维度。同时, $P_e \in R^{p \times e}$, 这里的$p$ 是prompt的长度。将两者拼接起来，得到$[P_e;X_e] \in R^{(p+n)\times e}$。训练时，只更新$\theta_p$.

Design Decision:
- random
- embedding from model's vocabulary
- embeddings that enumerate the output classes


##### 2.3.3 P-Tuning
论文：

	GPT Understands, Too
	https://arxiv.org/pdf/2103.10385.pdf
	P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks
	https://arxiv.org/pdf/2110.07602.pdf

原理：

##### 2.3.4 Prefix-Tuning
论文：

	Prefix-Tuning: Optimizing Continuous Prompts for Generation
	https://arxiv.org/pdf/2101.00190.pdf

原理：

![prefix](./pic/3/prefix.jpg "prefix")

- autoregressive LM: $[Prefix; x]$
- encode-decode: $[Prefix; x; {Prefix}^,; y]$


##### 2.3.5 Lora

  论文：

	LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
	https://arxiv.org/pdf/2106.09685.pdf
    AdaLoRA: ADAPTIVE BUDGET ALLOCATION FOR PARAMETEREFFICIENT FINE-TUNING
    https://arxiv.org/pdf/2303.10512.pdf
    https://github.com/QingruZhang/AdaLoRA
    QLORA: Efficient Finetuning of Quantized LLMs
    https://arxiv.org/pdf/2305.14314.pdf

原理：
- **lora**：
  
   $h=W_0 x + \Delta W * x  = W_0 x + BAx = (W_o + BA) x$， 其中 $W_0 \in R^{d \times k}$, $B \in R^{d \times r}$, $A \in R^{r \times k}$, $ r << min(d, k)$。

   初始时, $A$: random Gaussion initialization, $B$: zero， 同时为 $\Delta W $ 添加缩放系数 $ \frac{\alpha} {r}$。
   

![lora](./pic/3/lora.jpg "lora")

- **adalora**：

   - $h=W_0 x + \Delta W * x  = W_0 x + P\Lambda Q x$

   - 训练过程：
  
![adalora0](./pic/3/adalora-0.jpg "adalora0")

   - 损失函数及参数更新：
  
![adalora1](./pic/3/adalora-1.jpg "adalora1")

![adalora2](./pic/3/adalora-2.jpg "adalora2")

   - 三元组参数重要重要性计算：
  
   ![adalora3](./pic/3/adalora-3.jpg "adalora3")

- **qlora**：
![qlora](./pic/3/qlora.jpg "qlora")

##### 2.3.6 IA3
	Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
	https://arxiv.org/pdf/2205.05638.pdf

##### 2.3.7 Adapter

论文:

	Parameter-Efficient Transfer Learning for NLP
	https://arxiv.org/pdf/1902.00751.pdf
    AdapterFusion:Non-Destructive Task Composition for Transfer Learning
    https://arxiv.org/pdf/2005.00247.pdf
    MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer

原理：
- **adapter**：
![adapter](./pic/3/adapter.jpg "adapter")

- **adapterFusion**
	- knowledge extraction stage
	- knowledge composition step
![adapterfusion](./pic/3/adapterfusion.jpg "adapterfusion")
![adapterfusion-2](./pic/3/adapterfusion-2.jpg "adapterfusion-2")
  ![adapterfusion-sta](./pic/3/adapterfusion-sta.jpg "adapterfusion-sta")
  ![adapterfusion-mta](./pic/3/adapterfusion-mta.jpg "adapterfusion-mta")
  ![adapterfusion-stage2](./pic/3/adapterfusion-stage2.jpg "adapterfusion-stage2")
  ![adapterfusion-stage2-2](./pic/3/adapterfusion-stage2-2.jpg "adapterfusion-satge2-2")
  
- **MAD-X**


#### 2.4 强化学习
```
https://huggingface.co/docs/trl/example_overview
https://huggingface.co/blog/rlhf
```

![rlhf](./pic/4/rlhf-overview.jpg "rlhf")


##### 2.4.1 奖励模型

```
https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
```

![reward](./pic/4/reward.jpg "reward")

##### 2.4.2 RLHF
```
ppo:
https://github.com/huggingface/trl/blob/main/examples/scripts/ppo.py
```

![rlhf](./pic/4/rlhf.jpg "rlhf")

### 3 多模态

#### 3.1 数据

#### 3.2 多模态模型

##### 3.2.1 ViT
    An image is worth 16x16 words: Transformers for image recognition at scale
    https://arxiv.org/pdf/2010.11929.pdf
![vit](./pic/6/vit.jpg "vit")

##### 3.2.2 CLIP
    Learning Transferable Visual Models From Natural Language Supervision
    https://arxiv.org/pdf/2103.00020.pdf
![clip](./pic/6/clip.jpg "clip")

##### 3.2.3 Stable Diffusion
    High-Resolution Image Synthesis with Latent Diffusion Models
    https://arxiv.org/pdf/2112.10752.pdf
![stable-diffusion](./pic/6/stable-diffusion.jpg "stable-diffusion")

##### 3.2.3 ControlNet
    Adding Conditional Control to Text-to-Image Diffusion Models
    https://arxiv.org/pdf/2302.05543.pdf
![ControlNet-1](./pic/6/controlnet.jpg "ControlNet-1")
![ControlNet-1](./pic/6/controlnet-1.jpg "ControlNet-1")
![ControlNet-1](./pic/6/controlnet-2.jpg "ControlNet-1") 

##### 3.2.4 Imagen
    Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
    https://arxiv.org/pdf/2205.11487.pdf
![imagen](./pic/6/imagen.jpg "imagen") 

##### 3.2.5 Dreambooth
    DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation
    https://arxiv.org/pdf/2208.12242.pdf

### 4 分布式训练

#### 4.1 概述
#### 4.2 并行策略

##### 4.2.1 数据并行
- torch DDP
##### 4.2.2 模型并行
![imagen](./pic/8/model.jpg "imagen")
![imagen](./pic/8/model-2.jpg "imagen")
![imagen](./pic/8/model-3.jpg "imagen")
##### 4.2.3 张量并行
![imagen](./pic/8/tensor.jpg "imagen")
![imagen](./pic/8/tensor-2.jpg "imagen")
#### 4.3 DeepSpeed
![imagen](./pic/8/deepspeed1.jpg "imagen")


### 5 大模型应用框架
**LangChain**
![imagen](./pic/7/langchain.jpg "imagen")
LangChain 的提供了以下 6 种标准化、可扩展的接口并且可以外部集成的核心模块：
  - 模型输入/输出（Model I/O）与语言模型交互的接口；
  - 数据连接（Data connection）与特定应用程序的数据进行交互的接口；
  - 链（Chains）用于复杂的应用的调用序列；
  - 智能体（Agents）语言模型作为推理器决定要执行的动作序列；
  - 记忆（Memory）用于链的多次运行之间持久化应用程序状态；
  - 回调（Callbacks）记录和流式传输任何链式组装的中间步骤。


### 6 NLP 任务











