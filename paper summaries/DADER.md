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

### Discrepancy-based Methods

Using statistical metrics to minimize the domain distribution discrepancy between source and target. The Feature Aligner is basically a fixed function to calculate the discrepancy value and does not have parameters.

Training Procedure:

Feature Extractor generates features Xs and Xt which have feature space Ps and Pt. 

Feature Aligner computes a distrubution discrepancy (Alignment Loss) between Ps and Pt.

Matcher gives ER label prediction and computes the matching loss Lm over labeled data Ds. The goal is to reduce simultaneously La and Lm

(descibe what those methods are)
Maximum Mean Discrepancy (MMD)
𝐾-order
(                              )

### Adversial-based Methods

Using a domain classifier to distinguish features from source or target. Feature Aligner is a binary classifier implemented by fully-connected layers. Basically the goal is to generate features that are more similar so that the feature aligner cannot correctly predict them (explain this better)

GRL (Gradient Reversal Layer):   GRL incorporates a gradient reversal layer into the model architecture. The GRL has no parameters to update, and acts as an identity transformation in forward-propagation.

The Feature Aligner predicts the domain of Xs and Xt and computes domain classification loss La. During back-propagation, the gradient from the Feature Aligner multiplies a negative constant in the GRL. Meanwhile, the labeled Xs is inputted into the Matcher and computes the Mtching loss Lm

During training, the Feature Extractor should generate features that confuse the Domain Classifier, while the Domain Classifier aims to correctly classify the domain of the input features. This is why it's called adversial training, and the training continues until the Feature Extractor correctly fools the Domain Classifier.

InvGAN: GAN is made of two parts: a Generator, which generates "fake data" and a Discriminator, which has to distinguish between the fake data and the real data. In this scenario, the "real data" is the target features, and the "fake data" is the source features. The generator (the feature extractor) is trained to make the features from both domains indistinguishable, while training a Discriminator (Feature Aligner) to differentiate between them.

InvGAN+KD (Inverted Labels GAN + Knowledge Distillation): InvGAN+KD extends the InvGAN approach by incorporating knowledge distillation (KD). KD involves training a new "student" model (F') from an existing "teacher" model (F) to retain the classification ability of the teacher model. KD helps to prevent the loss of discriminative information in the features generated by the generator (F') while maintaining domain invariance. It achieves this by ensuring that the features generated by F' can still be distinguished by the original model (M), while also learning domain-invariant features through adversarial training.

### Reconstruction-based Methods

In Reconstruction-based Methods, the Feature Aligner is realized as a Decoder, which reconstructs the input data Ds and Dt. The reconstruction learns a shared hidden representation space between domains. The recostrunction loss Lrec ensures that the shared Feature Extracor extracts the most important information from both domains, while the matching loss Lm ensures that M works for the shared feature.

Encoder-Decoder (ED): in the Encoder-Decoder approach, the Feature Extractor is treated as Encoder, and the Feature Aligner as Decoder. The Feature Extractor (Encoder) generates hidden represetations for the original input text, which are then inputted into the Feature Aligner (Decoder) to generate the reconstructed text. The reconstruction loss Lrec can be calculated between the generated and the original text, i.e., the entity pairs. This approach is similar to Bart.


## DADER Algorithm

- Initialize Feature Extractor, Matcher and Feature Aligner
- For each iteration:
	- sample one mini-batch from labeled source and one mini-batch from unlabeled target
	- compute loss of the Matcher (equation 4?)
	- use back-propagation to tune the Matcher

	if Discrepancy or Reconstruction-based methods then:
	- Procedure 1 NoAdvAdapt: 
		- Compute Alignment Loss (using MMD or K-order)
		- Backpropagate to tune Feature Aligner if FA is a NN, otherwise if FA is a fixed function do not execute backpropagation
		- Backpropagate to tune Feature Extractor
	if GRL-based method then:
	- Procedure 2 GRLAdapt:
		- Compute Aligment Loss (equation 9?)
		- Backpropagate to tune Feature Aligner
		- Backpropagate to tune Feature Extractor, where gradient for F.E. from F.A. has been reversed by multiplying a parameter −{beta}
 
