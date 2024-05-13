# Thesis-Adapters

# Scheduling

- Replicare risultati DADER: inserire una tabella di confronto tra risultati del paper e quelli riprodotti da noi (riportare i settings)
- Replicare risultati UDAPTER (sui loro dataset): tabella analoga alla punto precedente (riportare i settings)
- Eseguire Udapter sui dataset di DADER
  - Salvare F1 scores
  - Salvare i runtimes
- Calcolare runtime per DADER
- Sperimentazione con Lora


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
