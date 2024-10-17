# Thesis-Adapters

# Scheduling
- Change the F1 on UDAPTER from 'macro' to 'binary' and re-execute udapter on all the datasets 
- Implement sampling on x instances of training dataset
- Check DADER: only source or also target?
- Try LoRA on some dataset
- Execute DADER and UDAPTER with 200 samples

- For DADER, UDAPTER and your model: save the results on a .csv file
- An example for how to save them:

``` 
if os.path.exists('results.csv'):
    old_results = pd.read_csv('results.csv')
    results = pd.concat([old_results, result])
else:
    results = result.copy()
    results.to_csv('results.csv', index=False)
```
- For next week, take 200 instances from the training set from UDAPTER and DADER and train the model only on those (remember to use stratify=True)

- Check out how MK-MMD metric works. What are gaussian kernels?
- Change the code so that it saves f1, std_dev, inference and training time


# Future ideas
- Look on the Huggingface-PEFT library for other methods beside LoRA
- Now you're training the model without source labels. What happens if we use only a small part of the source labels for training?
- In which layers is Lora applied (LoraConfig, layers_to_transform)? Should you apply LoRA on fewer layers?
- Think about implementing a new method for your thesis, like TIP-Adapter
- The deadline is 12.12.2024. You should start writing the thesis at least in November

### Literature
  - Zeyu Zhang, Paul Groth, Iacer Calixto, and Sebastian Schelter. "Directions Towards Efficient and Automated Data Wrangling with Large Language Models"
  - [Link to the paper](https://www.wis.ewi.tudelft.nl/assets/files/dbml2024/DBML24_paper_1.pdf) and [GitHub repository](https://github.com/Jantory/cpwrangle)

