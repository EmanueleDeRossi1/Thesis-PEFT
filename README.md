# Thesis-Adapters

# Scheduling

### Replicate DADER results
- Create a comparison table between the results from the paper and those reproduced by us (report the settings)
- Perform two identical runs with one seed, one dataset -> does the F1 remain the same?
- Perform run on one dataset, with all 5 seeds -> what's the difference between the results of the run and the results showed on the paper?
- Calculate runtime for DADER
- Select the best method from DADER
- Save model storage size, number of parameters, and compute inference time (i.e., time for generating a single prediction).


### Replicate UDAPTER results
- Use their datasets and create a comparison table similar to the previous point (report the settings).

### Execute UDAPTER on DADER datasets
- Save F1 scores and runtime

### Literature
- Read the paper by the Sebastian Schelter group (Amsterdam):
  - Zeyu Zhang, Paul Groth, Iacer Calixto, and Sebastian Schelter. "Directions Towards Efficient and Automated Data Wrangling with Large Language Models"
  - [Link to the paper](https://www.wis.ewi.tudelft.nl/assets/files/dbml2024/DBML24_paper_1.pdf) and [GitHub repository](https://github.com/Jantory/cpwrangle)
  - Which models are they using?
      - [PEFT, from HuggingFace](https://huggingface.co/docs/peft/index)

### Next
- Start experimenting with LoRA

## Dataset

The datasets are the one use on DADER. However, I changed its structure to make it more similar to the datasets used on UDAPTER, so we can more easily use the dataset of DADER on UDAPTER. 
The files have the same name and the same extension of the UDAPTER datasets, and the variables are called the same. Additionaly, the division of data into training, testing, and development sets follows the same proportions as those in UDAPTER.

### Content of the data directory:
  - different_domains: benchmark datasets from DeepMatcher and Magellan. Include source-target dataset pairs from different domains (e.g., Movies-Product)
  - similar_domains: benchmark datasets from from DeepMatcher and Magellan. Include source-target dataset pairs from the same domains (e.g., Product-Product)
  - wdc: WDC product datasets from e-commerce websites. Four categories: computers, cameras, watches and shoes. Source-Target dataset pairs are all permutations of the distict 4 categories (excluding Source-Target pairs like computers-computers)

The datasets are partitioned into three subsets: train, test, and dev, with proportions of 60%, 20%, and 20%, respectively.

Each source_target directory contains the files:

- train_source.csv : Training set of the source dataset.
- dev_source.csv : Development (dev) set of the source dataset.
- test_source.csv : Test set of the source dataset.

- target_unlabelled.csv : Training set of the target dataset (here the labels are dropped)
- dev_target.csv : Development (dev) set of the target dataset.
- test_target.csv : Test set of the target dataset.

### Abbreviations

- ab: Abt-Buy
- wa1: Walmart-Amazon
- ds: DBLP-Scholar
- da: DBLP-ACM
- fz: Fodors-Zagats
- dzy: Zomato-Yelp
- ia: iTunes-Amazon
- ri: RottenTomatoes-IMDB
- b2: Books2
- computers: WDC-Computers
- cameras: WDC-Cameras
- watches: WDC-Watches
- shoes: WDC-Shoes
