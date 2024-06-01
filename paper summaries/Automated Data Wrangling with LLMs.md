# Directions Towards Efficient and Automated Data Wrangling with Large Language Models


This paper explores the use of large language models (LLMs) for data wrangling tasks such as entity matching, error detection, and data imputation. It evaluates parameter-efficient fine-tuning (PEFT) methods to reduce the computational and storage costs associated with full model fine-tuning. The study finds that PEFT methods achieve performance close to full fine-tuning while being more parameter-efficient. However, parameter-efficient does not necessarily mean compute-efficient: the training time still remains high because PEFT models still require backpropagation through all model layers.  

## Dataset
The study evaluates ten benchmark datasets from various domains, including:
- **Entity Matching**: Beer, iTunes-Amazon, Fodors-Zagats, Walmart-Amazon, Amazon-Google, DBLP-ACM, DBLP-Google.
- **Error Detection**: Hospital.
- **Data Imputation**: Buy, Restaurant.

## Models
The study uses Google T5 in three variants (T5-small, T5-base, T5-large) and compares different parameter-efficient fine-tuning (PEFT) methods:
- **Prompt-Tuning**
- **P-Tuning**
- **Prefix-Tuning**
- **LoRA**

Baselines:
- **Zero-shot GPT-3** (OpenAI)
- **AutoML (autogluon)**
- **Fully fine-tuned T5 models**

## Experiments
The study conducts empirical evaluations to assess:
- **Prediction Quality**: F1 and accuracy scores across datasets.
- **Parameter Efficiency**: Number of parameter updates required.
- **Computational Efficiency**: Training and inference times on a single GTX 1080 ti GPU.

## Results
- **Prediction Quality**: PEFT methods generally outperform zero-shot GPT-3 and AutoML baselines. LoRA achieves the highest mean performance, often outperforming full fine-tuning for T5-small and T5-base.
- **Training Time**: PEFT methods incur high compute costs. Prefix-Tuning is the fastest but still comparable to full fine-tuning times.
- **Inference Throughput**: Prefix-Tuning provides the highest throughput, closely followed by LoRA. T5-small has significantly higher throughput compared to T5-base and T5-large.

## Analysis
- **Efficiency vs. Performance**: PEFT methods achieve performance close to full fine-tuning with fewer parameter updates but still have high training compute costs.

  
## Zero-Match

Talk about this.
