# UDAPTER - Efficient Domain Adaptation Using Adapters

## Dataset
The evaluation approach uses two representative datasets with different tasks, both in English. Each dataset has 5 domains, resulting in 20 domain adaptation scenarios per dataset. The details of the datasets are as follows:

- **AMAZON**: The Multi Domain Sentiment Analysis Dataset contains Amazon product reviews for five different types of products (domains): Apparel (A), Baby (BA), Books (BO), Camera_Photo (C), and Movie Reviews (MR). Each review is labeled as positive or negative.
- **MNLI**: The Multigenre Natural Language Inference (MNLI) corpus contains hypothesis–premise pairs covering a variety of genres: Travel (TR), Fiction (F), Telephone (TE), Government (G), and Slate (S). Each pair of sentences is labeled Entailment, Neutral, or Contradiction.

## Models
Two methods are proposed to enhance parameter efficiency in unsupervised domain adaptation (UDA) using pre-trained language models:

1. **Two-Steps Domain and Task Adapter (TS-DT Adapter)**:
   - Consists of two separate adapters: a domain adapter and a task adapter.
   - The domain adapter is trained first to generate domain-invariant representations.
   - The task adapter is then stacked on top of the domain adapter and trained separately while the domain adapter is left frozen.
   - The domain adapter uses MK-MMD to calculate the distribution discrepancy between the source and target data.
   - The task adapter uses cross-entropy loss between the source label and the source prediction.

2. **Joint Domain Task Adapter**:
   - Uses a single adapter to learn domain-invariant and task-specific representations jointly.
   - The trade-off between the two losses is regularized by an adaptation factor.

## Experiments
The two methods are evaluated on the following tasks:

- **Sentiment Analysis** using the Multi Domain Sentiment Analysis Dataset.
- **Natural Language Inference** using the MNLI corpus.

For each dataset, 20 domain adaptation scenarios are considered, resulting in a total of 40 scenarios across both datasets. Each experiment is run three times, and the mean and standard deviation of the F1 scores are reported.

## Results
The methods, TS-DT Adapter and Joint Domain Task Adapter, perform well in both datasets:

- **AMAZON**: The methods outperform other baselines, including DANN and DSN, and achieve competitive performance with fully fine-tuned UDA methods.
- **MNLI**: Similar results are observed, with the methods performing close to fully fine-tuned models and outperforming other adapter-based and UDA methods.

## Analysis
Several key findings and observations are made from the experiments:

- **Adapter Reduction Factor**: The bottleneck size of the adapters plays an important role in the final performance. Smaller reduction factors generally perform well in both datasets.
- **Removal of Adapters from Continuous Layer Spans**: Adapters added to higher layers are more effective, and removing adapters from the first few layers still preserves performance.
- **Composability**: The two-step method TS-DT Adapter shows composability, where task adapters can be reused for different domain pairs with minimal performance loss.

## Further Analysis
- **t-SNE Plots**: Visualizations of the t-SNE plots show that the lower layers of the pretrained model are domain-invariant, while higher layers are domain-variant. The methods effectively reduce divergence in higher layers.
- **Comparison with Baselines**: Simply replacing feature extractors in existing UDA methods with adapters is not sufficient. The proposed methods demonstrate better performance with minimal modifications to hyperparameters.

## Conclusion
UDAPTER is proposed to make unsupervised domain adaptation more parameter-efficient. The methods outperform strong baselines and perform competitively with other UDA methods while fine-tuning only a fraction of the parameters. This work demonstrates the potential of adapters for efficient domain adaptation in NLP tasks.
