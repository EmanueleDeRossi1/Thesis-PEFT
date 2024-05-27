# Towards Parameter-Efficient Automation of Data Wrangling Tasks with Prefix-Tuning

This paper explores the use of prefix-tuning as a lightweight alternative to fine-tuning large language models (LLMs) for data wrangling tasks such as entity matching, error detection, and data imputation. Prefix-tuning automates the learning of continuous prompts without updating the original LLM parameters. A more extensive paper is (Directions Towards Efficient and Automated Data Wrangling with Large Language Models
)
## Dataset
The study evaluates ten benchmark datasets from various domains, including:
- **Entity Matching**: Beer, iTunes-Amazon, Fodors-Zagats, Walmart-Amazon, Amazon-Google, DBLP-ACM, DBLP-Google.
- **Error Detection**: Hospital.
- **Data Imputation**: Buy, Restaurant.

## Models
The study uses the T5-base model and compares prefix-tuning against:
- **Full Fine-Tuning**: Updating all model parameters.
- **Zero-Shot Prompting**: Using the GPT-3 model.

## Experiments
The study conducts extensive empirical evaluations to assess:
- **Prediction Quality**: F1 scores for entity matching and error detection, accuracy for data imputation.
- **Parameter Efficiency**: Number of parameter updates required.
- **Computational Efficiency**: Training and inference times on a single GTX 1080 ti GPU.

## Results
- **Prediction Quality**: In five out of ten datasets, prefix-tuning is within 2.3% of the performance of fine-tuning. In eight out of ten datasets, it is within 5.2%.
- **Training Time**: Prefix-tuning incurs high compute costs similar to fine-tuning, with training times varying significantly between T5-small, T5-base, and T5-large.
- **Inference Throughput**: Prefix-tuning provides high throughput, making it suitable for large-scale applications.

## Analysis
- **Efficiency vs. Performance**: Prefix-tuning achieves performance close to full fine-tuning with only 0.39% of the parameter updates required for fine-tuning. This makes it a viable, scalable solution for data wrangling tasks.
- **Challenges**: Data imputation tasks are more challenging for prefix-tuning, and certain datasets like Amazon-Google and Walmart-Amazon show lower relative performance.
- **Future Vision (ZEROMATCH)**: The study proposes further exploration into zero-shot settings and the use of control prefixes to improve automation and efficiency in data wrangling with LLMs.

## Conclusion
Prefix-tuning offers a parameter-efficient alternative to full fine-tuning for data wrangling tasks, demonstrating promising results in terms of both performance and scalability. The approach has potential for further optimization and application in large-scale, real-world scenarios.
