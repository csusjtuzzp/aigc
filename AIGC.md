## 1 目录

[TOC]


## 2 大模型基础

### 2.1 构建框架

![大模型框架图](./pic/2-1.png "大模型框架图")



### 2.2 预训练模型

#### 2.2.1 文本分词

- 分词类性：
  - word 分词， OOV 问题
  - character分词，过长，失去了单词的联系
  - Subword：
  
    "unfortunately" = "un" + "for" + "tun" + "ate" + "ly"  
    
    常用算法：BPE，SentencePiece，WordPiece等
  
- BPE：
  
  Byte-Pair Encoding。先拆分为character, 然后统计token pair频率，合并， GPT
- WordPiece：
  
  与BPE 很相近，过程类似，但是会考虑单个token的频率，如果单个token的频率很高，也不会合并。如un-able， un 和 able 概率都很高，即使un-able pair很高，也不会合并

- Unigram：
  
  从一个巨大的词汇表出发，再逐渐删除trimdown其中的词汇，直到size满足预定义。初始的词汇表可以采用所有预分词器分出来的词，再加上所有高频的子串。每次从词汇表中删除词汇的原则是使预定义的损失最小。

  训练文档所有词为$x_1, x_2, ..., X_n$, 每个词token的方法是一个集合$S(x_i)$, 当一个词汇表确定时，每个词tokenize的方法集合$S(x_i)$ 就是确定的，每种方法对应一个概率$P(x)$, 依据损失进行删除。

- SentencePiece：
  
  把一个句子看作一个整体，再拆成片段，而没有保留天然的词语的概念。一般地，它把空格space也当作一种特殊字符来处理，再用BPE或者Unigram算法来构造词汇表。与Unigram算法联合使用

  
#### 2.2.2 Bert
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    https://arxiv.org/pdf/1810.04805.pdf

- pretrain:

  - Masked LM
  - Next Sentence Prediction
    
![Bert](./pic/2/bert.jpg "Bert")
![Bert-2](./pic/2/bert-2.jpg "Bert-2")

#### 2.2.3 GPT

    GPT1：Improving Language Understanding by Generative Pre-Training
    https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf
    GPT2：Language Models are Unsupervised Multitask Learners
    https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
    GPT3：Language Models are Few-Shot Learners
    https://arxiv.org/pdf/2005.14165.pdf
    InstructGPT：Training language models to follow instructions with human feedback
    https://arxiv.org/pdf/2203.02155.pdf


GPT1:
![GPT1](./pic/2/gpt1.jpg "GPT1")
自回归方式训练，预训练阶段窗口设置，需要主要的是fune-tuning时，损失函数包含预训练损失

GPT2：

堆数据，堆网络参数，网络norm层有变化

GPT3：
![GPT3](./pic/2/gpt3.jpg "GPT3")
模型使用spare attention, 175B参数

InstructGPT
![GPT3](./pic/2/gpt3.jpg "GPT3")

#### 2.2.4 BART

    BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
    https://arxiv.org/pdf/1910.13461.pdf

  ![BART1](./pic/2/bart.jpg "BART1") 

Pre-training:
  - Token Masking
  - Token Deletion
  - Text Infilling
  - Sentence Permutation
  - Document Rotation

![BART-2](./pic/2/bart-2.jpg "BART-2")
Fine-Tuning：
![BART-3](./pic/2/bart-3.jpg "BART-3")

在做machine translation时，先预训练参数，只更新initialized encoder参数，后完全更新参数。

#### 2.2.5 T5

    Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    https://arxiv.org/pdf/1910.10683.pdf

![t5](./pic/2/t5.jpg "t5")
添加前缀


#### 2.2.6 Unilm

    Unified Language Model Pre-training for Natural Language Understanding and Generation
    https://arxiv.org/pdf/1905.03197.pdf
#### 2.2.7 GLM

#### 2.2.8 LLaMA


### 2.3 有监督微调

有监督微调（Supervised Finetuning, SFT）又称指令微调（Instruction Tuning），是指在已经训练好的语言模型的基础上，通过使用有标注的特定任务数据进行进一步的微调，从而使得模型具备遵循指令的能力。经过海量数据预训练后的语言模型虽然具备了大量的“知识”，但是由于其训练时的目标仅是进行下一个词的预测，此时的模型还不能够理解并遵循人类自然语言形式的指令。



![有监督微调图](./pic/3-1.jpg "有监督微调图")



微调技术综述：



	Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning

	https://arxiv.org/pdf/2303.15647.pdf




#### 2.3.1 BitFit

	只调节神经网络的bias参数

![BitFit](./pic/3/bitfit.png "BitFit")


论文：

	BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models
	https://arxiv.org/pdf/2106.10199v2.pdf

代码：

```
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
```
#### 2.3.2 Prompt-Tuning

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

代码：
```
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit

# Soft Prompt
# config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)
# config
# Hard Prompt
config = PromptTuningConfig(
	task_type=TaskType.CAUSAL_LM, 
	prompt_tuning_init=PromptTuningInit.TEXT, 
	prompt_tuning_init_text="下面是一段人与机器人的对话。",                       num_virtual_tokens=len(tokenizer("下面是一段人与机器人的对话。")["input_ids"]),  
	tokenizer_name_or_path="Langboat/bloom-1b4-zh")

model = get_peft_model(model, config)

# inference
from peft import PeftModel
peft_model = PeftModel.from_pretrained(model=model, model_id="./chatbot/checkpoint-500/")
peft_model = peft_model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(peft_model.device)
print(tokenizer.decode(peft_model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True))

```

#### 2.3.3 P-Tuning

论文：

	GPT Understands, Too
	https://arxiv.org/pdf/2103.10385.pdf
	P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks
	https://arxiv.org/pdf/2110.07602.pdf

原理：



#### 2.3.4 Prefix-Tuning

论文：

	Prefix-Tuning: Optimizing Continuous Prompts for Generation
	https://arxiv.org/pdf/2101.00190.pdf

原理：

![prefix](./pic/3/prefix.jpg "prefix")


- autoregressive LM: $[Prefix; x]$
- encode-decode: $[Prefix; x; {Prefix}^,; y]$
$$h_i=$$




代码：
```
from peft import PrefixTuningConfig, get_peft_model, TaskType
config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10, prefix_projection=True)

model = get_peft_model(model, config)

# 核心代码
if peft_config.peft_type == PeftType.PREFIX_TUNING:
    past_key_values = self.get_prompt(batch_size)
    return self.base_model(
        input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
    )

class PrefixEncoder(torch.nn.Module):
    r"""
    The `torch.nn` model to encode the prefix.

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example:

    ```py
    >>> from peft import PrefixEncoder, PrefixTuningConfig

    >>> config = PrefixTuningConfig(
    ...     peft_type="PREFIX_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_hidden_size=768,
    ... )
    >>> prefix_encoder = PrefixEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The two-layer MLP to transform the prefix embeddings if
          `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (`batch_size`, `num_virtual_tokens`)

    Output shape: (`batch_size`, `num_virtual_tokens`, `2*layers*hidden`)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

if peft_config.peft_type == PeftType.PREFIX_TUNING:
    prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
    if peft_config.inference_mode:
        past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
    else:
        past_key_values = prompt_encoder(prompt_tokens)
    if self.base_model_torch_dtype is not None:
        past_key_values = past_key_values.to(self.base_model_torch_dtype)
    past_key_values = past_key_values.view(
        batch_size,
        peft_config.num_virtual_tokens,
        peft_config.num_layers * 2,
        peft_config.num_attention_heads,
        peft_config.token_dim // peft_config.num_attention_heads,
    )
    if peft_config.num_transformer_submodules == 2:
        past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
        peft_config.num_transformer_submodules * 2
    )
    if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
        post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
        past_key_values = post_process_fn(past_key_values)
    return past_key_values
```


#### 2.3.5 Lora

  论文：

	LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
	https://arxiv.org/pdf/2106.09685.pdf
    AdaLoRA: ADAPTIVE BUDGET ALLOCATION FOR PARAMETEREFFICIENT FINE-TUNING
    https://arxiv.org/pdf/2303.10512.pdf
    https://github.com/QingruZhang/AdaLoRA
    QLORA: Efficient Finetuning of Quantized LLMs
    https://arxiv.org/pdf/2305.14314.pdf

原理：
- lora：
   $h=W_0 x + \Delta W * x  = W_0 x + BAx = (W_o + BA) x$， 其中 $W_0 \in R^{d \times k}, B \in R^{d \times r}, A \in R^{r \times k}, r<<min(d, k)$。

   初始时，$A$: random Gaussion initialization， $B$: zero， 同时为$\Delta W$ 添加缩放系数 $\frac{\alpha} {r}$。
   

![lora](./pic/3/lora.jpg "lora")

- adalora：

   - $h=W_0 x + \Delta W * x  = W_0 x + P\Lambda Q x$

   - 训练过程：
   ![adalora-0](./pic/3/adalora-0.jpg "adalora-0")

   - 损失函数及参数更新：
   ![adalora-1](./pic/3/adalora-1.jpg "adalora-1")
   ![adalora-2](./pic/3/adalora-2.jpg "adalora-2")

   - 三元组参数重要重要性计算：
   ![adalora-3](./pic/3/adalora-3.jpg "adalora-3")

- qlora：
    ![qlora](./pic/3/qlora.jpg "qlora")
代码：

```
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=[".*\.1.*query_key_value"], modules_to_save=["word_embeddings"])

```


#### 2.3.6 IA3

	Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
	https://arxiv.org/pdf/2205.05638.pdf



#### 2.3.7 Adapter

论文:

	Parameter-Efficient Transfer Learning for NLP
	https://arxiv.org/pdf/1902.00751.pdf
    AdapterFusion:Non-Destructive Task Composition for Transfer Learning
    https://arxiv.org/pdf/2005.00247.pdf
    MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer

原理：



- adapter：
![adapter](./pic/3/adapter.jpg "adapter")
```
  def feedforward_adapter(input_tensor, hidden_size=64, init_scale=1e-3):
    """A feedforward adapter layer with a bottleneck.

    Implements a bottleneck layer with a user-specified nonlinearity and an
    identity residual connection. All variables created are added to the
    "adapters" collection.

    Args:
        input_tensor: input Tensor of shape [batch size, hidden dimension]
        hidden_size: dimension of the bottleneck layer.
        init_scale: Scale of the initialization distribution used for weights.

    Returns:
        Tensor of the same shape as x.
    """
    with tf.variable_scope("adapters"):
        in_size = input_tensor.get_shape().as_list()[1]
        w1 = tf.get_variable(
            "weights1", [in_size, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=init_scale),
            collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
        b1 = tf.get_variable(
            "biases1", [1, hidden_size],
            initializer=tf.zeros_initializer(),
            collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
        net = tf.tensordot(input_tensor, w1, [[1], [0]]) + b1

        net = gelu(net)

        w2 = tf.get_variable(
            "weights2", [hidden_size, in_size],
            initializer=tf.truncated_normal_initializer(stddev=init_scale),
            collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
        b2 = tf.get_variable(
            "biases2", [1, in_size],
            initializer=tf.zeros_initializer(),
            collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
        net = tf.tensordot(net, w2, [[1], [0]]) + b2

    return net + input_tensor

```
- adapterFusion
![adapterfusion](./pic/3/adapterfusion.jpg "adapterfusion")
![adapterfusion-2](./pic/3/adapterfusion-2.jpg "adapterfusion-2")
    - knowledge extraction stage
  ![adapterfusion-sta](./pic/3/adapterfusion-sta.jpg "adapterfusion-sta")
  ![adapterfusion-mta](./pic/3/adapterfusion-mta.jpg "adapterfusion-mta")
    - knowledge composition step
  ![adapterfusion-stage2](./pic/3/adapterfusion-stage2.jpg "adapterfusion-stage2")
  ![adapterfusion-stage2-2](./pic/3/adapterfusion-stage2-2.jpg "adapterfusion-satge2-2")

    ```
    model = BertModelWithHeads.from_pretrained("bert-base-uncased")
    model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False)
    model.load_adapter("sts/qqp@ukp", with_head=False)
    model.load_adapter("nli/qnli@ukp", with_head=False)
    model.add_classification_head("cb")

    adapter_setup = Fuse("multinli", "qqp", "qnli")
    model.add_fusion(adapter_setup)
    model.set_active_adapters(adapter_setup)
    model.train_fusion(adapter_setup)
    ```
  
- MAD-X


### 2.4 强化学习

#### 2.4.1 奖励模型

	https://zhuanlan.zhihu.com/p/595579042

#### 2.4.2 RLHF

### 2.5 LangChain



## 3 扩散模型





## 4 NLP 任务



## 5 视觉



## 6 模型训练







