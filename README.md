# Conditional-Adversarial-Domain-Generalization-with-Single-Discriminator
Pytorch implementation of Conditional Adversarial Domain Generalization With a Single Discriminator for Bearing Fault Diagnosis [1].



## Script execution

There is a configuration file named `train_config.yml`.  This file contains the following variables that must me configured:

- `train_set`: Path to the training set `.csv` file.
- `val_set`: Path to the validation set `.csv` file.
- `checkpoint`: Checkpoint name to save the model.
- `train_set`: Name of the model to be used as feature extractor. It uses the `timm` package.

Once the file is properly configured, you must execute the following command:

```bash
python train.py -c train_config.yml
```



## References

[1] Q. Zhang *et al*., "Conditional Adversarial Domain Generalization With a Single Discriminator for Bearing Fault Diagnosis," in *IEEE Transactions on Instrumentation and Measurement*, vol. 70, pp. 1-15, 2021, Art no. 3514515, doi: 10.1109/TIM.2021.3071350.