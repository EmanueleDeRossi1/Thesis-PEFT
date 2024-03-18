# TITLE OF THE PAPER

## What the apper says in few words

## Datasets 
There are two datasets used:

- The benchmark datasets from DeepMatcher and Magellan, which covers various domains such as product, citation, and restaurant. Each dataset comprises entities from two relational tables with multiple attributes and a set of labeled matching/non-matching entity pairs.

For instance, the DBLP-Scholar (DS) dataset consists of two tables extracted from DBLP and Scholar, each with four aligned attributes (Title, Authors, Venue, Year). This dataset contains 28,707 entity pairs, including 5,347 matching pairs.


- The WDC product datasets, collected from e-commerce websites, each containing four categories: computers, cameras, watches, and shoes, with 1100 labeled entity pairs per category.

### Dataset split and evaluation

There are two datasets: a labeled source dataset and an unlabeled target dataset. The model is trained on both datasets, using the source dataset to learn the task-specific features, and the target dataset to learn domain-specific features. Since we train the model on the unlabeled target training set, we can use that set to perform the evaluation of the model at the end. This means that the target dataset is split into a validation set (10%) and a test/training set (90%), which is used for training (since it's unlabeled) and for the evaluation (using the labels).

Model snapshots are evaluated on the validation set, and the best-performing snapshot is chosen. The optimized model is then used to make predictions on the test set for evaluation.
The evaluation metric used is the F1 score, which combines precision (the ratio of true positive to the actual positive cases: TP / (TP + FP)) and recall (the ratio of the true positive to all the cases identified as positive: TP + (TP + FN)) into a single value. 

## Framework 

The framework consists of three main modules. 

(1) A Feature Extractor, which converts entity pairs to high-dimensional vectors (a.k.a. features). 
The Feature Extractor used is a Bert model (bert-base-multilingual-cased) with 12 transformer layers and output a 768 dimensional feature vector. The max sequence length fed into Bert is 256 (for WDC datasets) and 128 (for other datasets)


(2) A Matcher, a binary classifier that takes the features of entity pairs as input, and predicts whether they match or not. The Matcher consists of one fully connected layer and a Softmax output layer.

(3) A Feature Aligner, the key module for domain adaptation, which is designed to alleviate the effect of domain shift. To achieve this, Feature Aligner adjusts Feature Extractor to align distributions of source and target ER datasets, which then reduces domain shift between source and target. Moreover, it updates Matcher accordingly to minimize the matching errors in the adjusted feature space.

Feature Aligner is implemented by three categories of solutions: (1) discrepancy-based, (2) adversarial-based, and (3) reconstruction-based.

- For implementing discrepancy-based Feature Aligner, MMD and K-order are calculated directly on the ouptu layer of Feautre Extractor F

- For adversial-based Feature Aligner, one fully connected layer with Sigmod activation function in GRL is used, which is followed by three fully connected layers with LeakyReLU as activation function and a Sigmod layer for InvGAN and InvGAN+KD

- For reconstruction-based Feature Aligner, a pre-trained model Bart with its default settings is used to realize the reconstruction task in ED

A more extensive description of the Feature Aligner architectures is presented in the next section.

Validation set of target is used for choosing hyper-parameters. More information on hyper-parameters tuning on page 9, left-top corner. (ADD)

## Description of Feature Aligner methods

### Discrepancy-based Methods (MMD and 𝐾-order MMD)

Using statistical metrics to minimize the domain distribution discrepancy between source and target. The Feature Aligner is basically a fixed function to calculate the discrepancy value and does not have parameters.

During training, the Feature Extractor generates features Xs and Xt which have feature space Ps and Pt. The Feature Aligner then computes a distrubution discrepancy (Alignment Loss) between Ps and Pt. Finally, a Matcher gives ER label prediction and computes the matching loss Lm over labeled data Ds. The goal is to reduce simultaneously La and Lm.

MMD is a method used to measure the discrepancy between two probability distributions. It quantifies the difference in mean embeddings of data points from two distributions in a reproducing kernel Hilbert space (RKHS). The 𝐾-order MMD extends this concept by considering higher-order moments in addition to the mean, providing a more comprehensive measure of distribution discrepancy.

### Adversial-based Methods (GRL, InvGAN, InvGAN+KD)

Using a domain classifier to distinguish features from source or target. Feature Aligner is a binary classifier implemented by fully-connected layers. Basically the goal is to generate features that are more similar so that the feature aligner cannot correctly predict them (explain this better)

GRL (Gradient Reversal Layer):   GRL incorporates a gradient reversal layer into the model architecture. The GRL has no parameters to update, and acts as an identity transformation in forward-propagation.

The Feature Aligner predicts the domain of Xs and Xt and computes domain classification loss La. During back-propagation, the gradient from the Feature Aligner multiplies a negative constant in the GRL. Meanwhile, the labeled Xs is inputted into the Matcher and computes the matching loss Lm.

During training, the Feature Extractor should generate features that confuse the Domain Classifier, while the Domain Classifier aims to correctly classify the domain of the input features. This is why it's called adversial training, and the training continues until the Feature Extractor correctly fools the Domain Classifier.

InvGAN: GAN is made of two parts: a Generator, which generates "fake data" and a Discriminator, which has to distinguish between the fake data and the real data. In this scenario, the "real data" is the target features, and the "fake data" is the source features. The generator (the feature extractor) is trained to make the features from both domains indistinguishable, while training a Discriminator (Feature Aligner) to differentiate between them.

InvGAN+KD (Inverted Labels GAN + Knowledge Distillation): InvGAN+KD extends the InvGAN approach by incorporating knowledge distillation (KD). KD involves training a new "student" model (F') from an existing "teacher" model (F) to retain the classification ability of the teacher model. KD helps to prevent the loss of discriminative information in the features generated by the generator (F') while maintaining domain invariance. It achieves this by ensuring that the features generated by F' can still be distinguished by the original model (M), while also learning domain-invariant features through adversarial training.

### Reconstruction-based Methods (ED)

In Reconstruction-based Methods, the Feature Aligner is realized as a Decoder, which reconstructs the input data Ds and Dt. The reconstruction learns a shared hidden representation space between domains. The recostrunction loss Lrec ensures that the shared Feature Extracor extracts the most important information from both domains, while the matching loss Lm ensures that M works for the shared feature.

Encoder-Decoder (ED): in the Encoder-Decoder approach, the Feature Extractor is treated as Encoder, and the Feature Aligner as Decoder. The Feature Extractor (Encoder) generates hidden represetations for the original input text, which are then inputted into the Feature Aligner (Decoder) to generate the reconstructed text. The reconstruction loss Lrec can be calculated between the generated and the original text, i.e., the entity pairs. This approach is similar to Bart.


## Results

Two settings are considered for evaluation:
- Similar Domains.
- Different Domains.
In both settings, DA demonstrates significant improvements compared to NoDA (no domain adaptation), even when datasets are from similar domains (sometimes in similar domains there can be variations in attributes or textual styles, causing domain shift effects). But in different domains the improvements are more noticeble. 

t-SNE is used for visualizing high-dimensional feature distributions of source and target datasets

- But improvement is not always significant. In some cases, such as DBLP-Scholar → DBLP-ACM, DA yields no improvement over NoDA because models trained on the source dataset already perform well on the target dataset. 

- Evaluation on WDC datasets reveals that the gain from DA is not prominent, indicating that the data distribution among different datasets is very similar.
- NoDA achieves high performance on target datasets, even outperforming state-of-the-art models trained on their own training sets. This suggests that domain shift may not be significant in this scenario, limiting the potential for improvement through DA.


- Using MMD, we can calculate the distance between dataset (smaller MMD = similar datasets)
- F1 scores are higher when MMD is smaller

- When source and target datasets are from similar domains, DA gets better results than from different domains (duh). 

## Analysis

### Evalution of Feature Aligner

- Succesful Cases Analysis:

- MMD and InvGAN+KD outperform NoDA in nearly all the cases
- Discrepancy-based DA performs well to be convergent with enough training epochs and achieves obvious improvements, while adversarial-based DA may be oscillate. The oscillation may be reduced by reducing learning rate, which may lead to more training epochs
- InvGAN+KD is sensitive to the learning rate

- Failure Cases Analysis

- InvGAN is worse than NoDA is many cases. This is because the Feature Extractor in InvGAN make the target features as similar to the source features as possible, whether the features thus obtained are discriminative or not. This is proved by the fact that the Matcher becomes much worse even on the source dataset. InvGAN+KD, instead, is much more stable and performative.

- ED also achieves inferior performance in all cases. Maybe enconder-deconder approach fail to capture and reconstruct the textual information of original entity pairs

- GRL is generally good, but sometimes NoDa outperforms it (e.g., Book2 → Zomato-Yelp.). GRL training is generally not stable.

### Evaluation on Feature Extractor

Two F.E. used: Bert and a bidirectional RNN, but Bert outperforms RNN in all 3 datasets. RRN fails to transfer efficiently, so that the model trained on source Ds relies heavily on itself and does not work well on the target dataset

