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

