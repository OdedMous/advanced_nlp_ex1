import os
import sys
import numpy as np
import evaluate
import wandb
import torch

from datasets.arrow_dataset import Dataset
from datasets import load_dataset

from transformers import (TrainingArguments,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          EvalPrediction,
                          Trainer)


# Set this to TRUE to report results in Weights&Biases
ENABLE_WANDB = False



def reduce_dataset_size(dataset, max_samples : int):

    if max_samples > 0:
        dataset = dataset.select(range(max_samples))

    return dataset

def preprocess_function(examples, tokenizer):

    # no padding here since we want DYNAMIC padding
    # we don't insert return_tensors='pt' cuz the trainer will convert the inputs to tensors by itself
    result = tokenizer(examples["text"], truncation=True) # , return_tensors='pt'
    return result


def preprocess_datasets(raw_datasets, tokenizer: AutoTokenizer, max_train_samples, max_eval_samples, max_test_samples):

    raw_datasets = raw_datasets.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer}, batched=True,
                                                                    batch_size=None)

    train_dataset = reduce_dataset_size(raw_datasets["train"], max_train_samples)
    eval_dataset = reduce_dataset_size(raw_datasets["validation"], max_eval_samples)
    test_dataset = reduce_dataset_size(raw_datasets["test"], max_test_samples)

    return train_dataset, eval_dataset, test_dataset

def return_metric_func(metric):
    """
    This is a function that returns a "compute_metrics" function.
    """

    def compute_metrics(p: EvalPrediction):
        """
        This is a custom compute_metrics function.
        :param p: an `EvalPrediction` object (a namedtuple with a predictions and label_ids field)
        :return: a dictionary string to float (keys are strings and values are floats).
        """
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result

    return compute_metrics

def do_finetuning(model_name : str,
                  seeds_num: int,
                  tokenizer: AutoTokenizer,
                  train_dataset: Dataset,
                  eval_dataset: Dataset,
                  device):

    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    best_trainer = None
    best_accuracy = 0
    acc_list = []
    train_times = []

    for seed in range(seeds_num):

        print(f"Finetuning {model_name} with seed {seed}...")

        if ENABLE_WANDB:
            # set the wandb project where this run will be logged
            wandb.init(project="my_anlp", name=f"{model_name}_seed_#{seed}")
            report_to = "wandb"
        else:
            report_to = 'none'

        # Define metric
        accuracy_metric = evaluate.load("accuracy")
        compute_metrics_func = return_metric_func(metric=accuracy_metric)

        training_args = TrainingArguments(seed=seed,
                                          do_train=True,
                                          output_dir=os.getcwd(),
                                          report_to=report_to,
                                          logging_steps=500,  # after how many step to push loss value into wandb
                                          save_total_limit=1 #  save only the last checkpoint (note that the loss will be still calculated after each logging_steps and will be reported to Weights&Biases)
                                          )  # save_strategy="no # We don't need eval during training in our ex1

        # Note that trainer creates automatically DataLoaders (with batch size,shuffle,  etc.) from the given datasets
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset, # we dont need to evaluate during training in ex1
            compute_metrics=compute_metrics_func,
            tokenizer=tokenizer,
        )

        train_result = trainer.train()

        train_times.append(train_result.metrics['train_runtime'])

        # Evaluation on validation set
        # note that "model.eval()" is already done inside trainer.evaluate/predict functions
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        if ENABLE_WANDB:
           wandb.log(metrics)
           wandb.finish()

        acc = metrics['eval_accuracy']
        acc_list.append(acc)

        if acc > best_accuracy:
            best_accuracy = acc
            best_trainer = trainer

    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)

    # Save model
    # trainer.save_model(model_name+"_seed_#"+str(seed)) # note that this saves the tokenizer too for easy upload

    return best_trainer, mean_acc, std_acc, train_times


def do_predict(trainer: Trainer, test_dataset) -> float:

    training_args = TrainingArguments(seed=trainer.args.seed,        # use the same seed as used for training
                                      output_dir=os.getcwd(),
                                      per_device_eval_batch_size=1)  # enforce batch_size=1 for predictions

    trainer.args = training_args

    prediction_results = trainer.predict(test_dataset.remove_columns(['label', 'label_text']))
    raw_preds = prediction_results.predictions  # probabilities for each class
    preds = np.argmax(raw_preds, axis=-1)

    # Write predictions to file
    with open('predictions.txt', 'w', encoding="utf-8") as f:
        for org_sentence, label in zip(test_dataset['text'], preds):
            f.write(f'{org_sentence}###{label}\n')

    test_runtime = prediction_results.metrics["test_runtime"]

    return test_runtime

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available device:", device)

    if ENABLE_WANDB:
       # take key from your username page in: https://wandb.ai
       wandb.login() # key="???"

    args = sys.argv
    seeds_num = int(args[1])
    max_train_samples = int(args[2])
    max_eval_samples = int(args[3])
    max_test_samples = int(args[4])

    raw_datasets = load_dataset("SetFit/sst2")

    all_train_times = []
    best_mean_acc = 0
    best_trainer = None

    open('res.txt', 'w').close()

    for model_name in  ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']:

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset, eval_dataset, test_dataset = preprocess_datasets(raw_datasets, tokenizer, max_train_samples,
                                                                                                 max_eval_samples,
                                                                                                 max_test_samples)

        trainer, mean_acc, std_acc, train_times = do_finetuning(model_name, seeds_num, tokenizer, train_dataset, eval_dataset, device)
        all_train_times.extend(train_times)

        with open('res.txt', 'a') as f:
            f.write(f'{model_name},{mean_acc} +- {std_acc}\n')

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_trainer = trainer

    total_train_time = sum(all_train_times)

    test_runtime = do_predict(best_trainer, test_dataset)

    with open('res.txt', 'a') as f:
        f.write('----\n')
        f.write(f'train time, {total_train_time}\n')
        f.write(f'predict time, {test_runtime}\n')


if __name__ == '__main__':

    main()




