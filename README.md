# Thesis-Adapters

# Scheduling

- Check out how MK-MMD metric works. What are gaussian kernels?
- Change the code so that it saves f1, std_dev, inference and training time
- What hparams could you change in the hparams tuning, and why?
- Train LoRA and save the results on overleaf. Also add a summary about the hyperparams and settings you used 
- Template for thesis?

# Future ideas
- Look on the Huggingface-PEFT library for other methods beside LoRA
- Now you're training the model without source labels. What happens if we use only a small part of the source labels for training?
- In which layers is Lora applied (LoraConfig, layers_to_transform)? Should you apply LoRA on fewer layers?
- Think about implementing a new method for your thesis, like TIP-Adapter
- The deadline is 12.12.2024. You should start writing the thesis at least in November

### Literature
  - Zeyu Zhang, Paul Groth, Iacer Calixto, and Sebastian Schelter. "Directions Towards Efficient and Automated Data Wrangling with Large Language Models"
  - [Link to the paper](https://www.wis.ewi.tudelft.nl/assets/files/dbml2024/DBML24_paper_1.pdf) and [GitHub repository](https://github.com/Jantory/cpwrangle)

