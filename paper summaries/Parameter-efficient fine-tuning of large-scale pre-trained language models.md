#  Parameter-efficient fine-tuning of large-scale pre-trained language models
(https://www.nature.com/articles/s42256-023-00626-4)

The paper discuss and analyse the design and performances of parameter-efficient architectures (here called "delta-tuning" methods, reffering to the portion ("delta") of parameters of the adapter that are changed during training).
The paper categorizes delta-tuning methods into addition-based, specification-based, and reparameterization-based approaches, and evaluates their performance, convergence, and efficiency across over 100 NLP tasks.


## Hint on terminology:

Most of the times on the literature the term "adapter" refers only to those "bottleneck" architectures, where two layers, the first that reduces the dimension of the input, and the second that projects the input in the original dimension, are added at the end of the attention block of the transformer. This is the case of the present paper.
But in most rare cases, the term "adapter" refers to all those methods that introduce a small number of new parameters and only update these while keeping the pre-trained model weights fixed (e.g., LoRA, prompt-tuning, Compacter etc.).

Usually those methods are called in literature PEFT (Parameter-Efficient Fine-Tuning) or in the case of the present paper, Delta-Tuning.

## Types of Delta-tuning (PEFT) Methods

### Addition-based methods 

Which introduce additional parameters to the NN. To the addition-based methods belong:

#### Adapter-based tuning

Consists in injecting a small-scale neural module, called adapter, to the transformer layers and only tune these adapters for model adaptation. One adapter module contains a down-projection and a up-projection to the original dimension.
In each block, the adapter modules are separately inserted after the multi-head self-attention and the feed-forward network sublayers.

Some methods attempt even a more rigorous saving strategy by introducing inductive biases into the structure of the adapter layer: Compacter

The advantage of adapters is that you can place multiple adapter on a single PLM. For example, adapterFusion first pre-trains task-specific adapters and then combines the representations of the pre-trained adapters to leverage the cross-task knowledge  and enhance the performance of transfer learning (what does this mean?)
60% faster than fine-tuning, and only 4-6% convergence slower

Adapter-based tuning more robust and could perform better than fine-tuning on few-shot and cross-lingual scenarios.

#### Prompt-based tuning

Prompt-based tuning work by adding to the original input additional context. This strategy obtained good results in low-data settings. There are two main architecture that belong to this category:

- Prefix-tuning, which prepends trainable continuous tokens (prefixes) to the input and hidden states of each transformer layer. 

- Prompt-tuning, which adopts a simplified strategy by prepending soft prompts to the input layer. During training, the parameters of soft prompts are updated while the PLM is left frozen. 

The difference in performance between prompt-tuning and prefix-tuning decreases with the increase of size of the model. 

A big promblem of prompt-tuning is that it's difficult to optimize, and even if soft-prompts can be trained successfullly, they converge slower than full-parameter fine-tuning and other delta-tuning methods.

### Specification-based methods

Specification-based methods don't introduce any new parameters to the model, bur rather select a section of the model parameters to be optimized. 

A study by (https://arxiv.org/abs/1911.03090) only fine-tunes 1/4 of the final layers of BERT and RoBERTa and produce 90% of the performance of full-parameter fine-tuning. BitFit instead proves that by only optimizing the bias term inside the model, it could reproduce 95% of the performance on several benchmarks.

### Reparameterization-based methods

This branch of methods seeks to transform the parameters of the model in a more parameter-efficient form. It stems from the hypothesis that fine-tuning of most downstream tasks can be reduced to a low rank approximation. An example of reparameterization-based methods is LoRA.


## Results

They measured performance, convergence and efficacy of 4 fine-tuning methods: prompt-tuning, prefix-tuning, LoRA, and adapters.


### Experiments

- Evaluated four representive delta-tuning methods against vanilla fine-tuning:

- Prompt-tuning (PT)
- Prefix-tuning (PF)
- LoRA (LR)
- Adapter (AP)


- 100 representative tasks from Huggingface datasets, such as text classification (sentiment analysis, NLI...), question asnwering, conditional generation (summarization, dialogue...) etc.

### Analysis of performance
Performance of the delta-tuning emthods are in general comparable to fine-tuning

If we were to make a list of the most performing methods according to the avarage results:

1. Fine-tuning
2. Lora
3. Adapter
4. Prefix-tuning
5. Prompt-tuning

More tunable parameters does not always lead to better performance, and the design of the structure of delta-tuning plays a greater role
Prompt-tuning performs much worse that other delta-tuning methods, especially with smaller PLM. However, when the size of the PLM is bigger, performance improves sharply.

#### Analysis of convergence

1. From the fastest to the slowest method to converge:

1. Finetuning
2. Adapter and LoRA
3. Prefix-tuning
4. Prompt-tuning (far behind the 3 other methods)

2. Generally, performance and convergence is not that sensitive to the size, but rather to the structure of the methods, and as the size of the PLM increases, the convergence accelerates

### Efficiency Analysis

1. Delta-tuning generally reduces memory and computational costs.
2. Combining delta-tuning methods often yields better results than using a single method, with the optimal combination varying by task and model.

## Conclusion

Delta-tuning presents a promising approach for adapting large-scale PLMs efficiently. The paper highlights the importance of method structure over the number of tunable parameters and evaluates delta-tuning methods with various datasets and tasks.
