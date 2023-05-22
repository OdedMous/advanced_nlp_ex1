# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1, fine tuning pretrained models to perform sentiment analysis on the SST2 dataset.

# Install
``` pip install -r requirements.txt ```

# Fine-Tune and Predict on Test Set
Run:

``` python ex1.py <number of seeds> <number of training samples> <number of validation samples> <number of prediction samples> ```

Generated files are res.txt, showing model's performance, training time and prediction time, and prediction.txt, containing prediction results for all test samples.

# Plots

Plots produced by Weights & Biases platform.

![alt text](https://github.com/OdedMous/advanced_nlp_ex1/blob/main/train_loss.png)

![alt text](https://github.com/OdedMous/advanced_nlp_ex1/blob/main/GPU_Utilization.png)

# Run on Google Colab

In order to utilize google colab GPU, one can simply paste the code from ex1.py to a code snippet (need only change the command-line arguments manually).
