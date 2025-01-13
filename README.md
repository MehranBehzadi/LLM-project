# LLM Project

## Project Task
The LLM Sentiment Analysis Project focuses on leveraging a language model to perform sentiment analysis on a provided dataset. The goal is to classify text into different sentiment categories, such as positive, negative, and neutral, using natural language processing (NLP) techniques and machine learning.

## Dataset
Given that the project focuses on sentiment analysis, the dataset that was chosen for it was imdb data to fullfil objectives of this project.

## Pre-trained Model
The pre-trained model that was used for this project is lvwerra/distilbert-imdb which can be found from https://huggingface.co/lvwerra/distilbert-imdb, with accuracy of 0.928 and loss of 0.1903.
For preprocessing tokenization has been performed

## Performance Metrics
The following were the metric results

training_loss=0.1424872226922541
train_runtime': 1328.5631
'train_samples_per_second': 18.817
'train_steps_per_second': 1.176
'total_flos': 3311684966400000.0
'train_loss': 0.1424872226922541 
'epoch': 1.0

Here is the evaluation results

'eval_loss': 0.24149523675441742
 'eval_model_preparation_time': 0.0029
 'eval_accuracy': 0.92868
 'eval_runtime': 380.7321
 'eval_samples_per_second': 65.663
 'eval_steps_per_second': 8.208


## Hyperparameters
The following hyperparameters were used during the training:

learning_rate=2e-5,
per_device_train_batch_size=16
per_device_eval_batch_size=16
num_train_epochs=1
