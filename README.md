# Thesis-Adapters

## Dataset

Content of the data directory:
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
