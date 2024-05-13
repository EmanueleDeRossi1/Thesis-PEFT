## What's prefix tuning?

Prefix-tuning add a sequence of vectors to the input (called prefix). It draws inspiration from prompting, where additional context or instruction are used to steer a LM to solve a NLG task. But how well does using real-language instruction works for pretrained LMs? 

Instead of using real tokens like in propting, prefix-tuning uses free parameters (continous word embeddings) appended at the beginning of each sequence (the prefix), which are optimized during training. This approach seems to give better results that prompting. And while prompting optimizes only the embedding layer, prefix-tuning optimizes the activations of all the layers.

Prefix-tuning is very efficient computationally: we only need to store one single copy of the large LM and a learned task-specific prefix.


## Datasets

#### For table-to-table task:

Evaluated on 3 standard neural generation datasets:
- E2E (restaurant reviews)
- WebNLG (14 domains)
- Dart (open domain - Wikipedia)

#### For summarization:
- XSUM dataset (an abstractive summarization dataset on news articles)

## Evaluation and Results

Prefix-tuning evaluated on table-to-text generation (using CPT-2) and abstractive summarization (using BART).

#### For table-to-text generation:

Compared prefix-tuning against:
- full fine-tuning
- fine-tuning only the top 2 layers
- adapter-tuning

#### For summarization:

Compared prefix-tuning against BART

When trained on full datasets, prefix-tuning and fine-tuning obtain similar results for table-to-text. In the summarization task, prefix tuning perform slighly worse. (u should put some numbers in here)

In low-data settings, instead, prefix-tuning outperforms fine-tuning on both tasks, and extrapolates better to tables (for table-to-text) and articles (for summarization) with unseen topics (put some numbers)


## Results:

#### Table-to-text Generation:

- Prefix-tuning, with just 0.1% task-specific parameter updates, outperforms other lightweight baselines like ADAPTER and FT-TOP2, even with updating 30 times fewer parameters, and achieves comparable results to full fine-tuning.

- When parameters are matched to 0.1%, prefix-tuning significantly outperforms ADAPTER, showing a 4.1 BLEU improvement per dataset on average.
- Even compared to full fine-tuning (100%) and adapter-tuning (3.0%), which update significantly more parameters than prefix-tuning, prefix-tuning achieves comparable or better results, showcasing its Pareto efficiency.
- Prefix-tuning shows promising results in generalizing to tables with diverse domains and a large number of relations, as demonstrated by good performance on DART dataset.

#### Summarization:

- Prefix-tuning achieves slightly lower performance compared to full fine-tuning in summarization tasks, particularly with only 0.1% of parameters.
- The advantages of prefix-tuning observed in table-to-text generation may not directly translate to summarization tasks due to differences in dataset characteristics, such as length and complexity.

#### Low-data Setting:

- Prefix-tuning demonstrates a comparative advantage over fine-tuning, especially in low-data settings, outperforming fine-tuning by 2.9 BLEU on average.

- The gap between prefix-tuning and fine-tuning narrows as the dataset size increases.
- Qualitatively, prefix-tuning tends to be more faithful than fine-tuning, especially in low-data regimes.

#### Extrapolation:

- Prefix-tuning exhibits better extrapolation performance to unseen topics compared to fine-tuning, in both table-to-text and summarization tasks.
- Adapter-tuning also achieves good extrapolation performance, comparable to prefix-tuning, suggesting that preserving LM parameters positively impacts extrapolation.


## Intrinsic Evaluation

Using real words relevant to the task improves performance significantly compared to random initialization. However, in full data settings, initialization has no significant impact.

- Prefix Length: Longer prefixes increase expressive power initially, but beyond a certain threshold (200 for summarization, 10 for table-to-text), performance drops due to potential overfitting.

- Full vs. Embedding-only: Tuning only the embedding layer (embedding-only) significantly reduces performance compared to full prefix-tuning, indicating that it lacks expressiveness. Discrete prompt optimization is even less expressive than embedding-only.

- Prefix-tuning vs. Infix-tuning: Prefix-tuning (activations placed at the beginning) outperforms infix-tuning (activations placed between x and y), possibly because prefix-tuning can influence both x and y activations, whereas infix-tuning only affects y activations.

- Initialization: Initialization of the prefix significantly impacts performance, especially in low-data settings. Initializing with real words relevant to the task improves performance significantly compared to random initialization. However, in full data settings, initialization has no significant impact.

- Data Efficiency: Prefix-tuning demonstrates better performance compared to full fine-tuning when using more than 20% of the data. In low-data scenarios (10% of the data), prefix-tuning with random initialization performs similarly to or slightly worse than full fine-tuning, requiring the initialization trick to enhance performance.
