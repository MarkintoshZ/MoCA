#!/usr/bin/env python
# coding: utf-8

# # Momentum Calibration for Text Generation
# https://arxiv.org/abs/2212.04257

get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0,1')

from datetime import datetime
import copy

from tqdm.notebook import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import transformers
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoTokenizer, pipeline
import evaluate
from evaluate import evaluator


# load dataset
cnn_test = load_dataset("cnn_dailymail", '3.0.0', split="test")

# load model and its associated components
checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


def preprocess_function(examples):
    inputs = examples["article"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["highlights"], max_length=142, 
                       truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_test = cnn_test.map(preprocess_function, batched=True)

# evaluation setup
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="BART",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=10,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=cnn_train,
    eval_dataset=cnn_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.evaluate(tokenized_test)