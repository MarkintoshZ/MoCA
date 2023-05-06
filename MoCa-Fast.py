#!/usr/bin/env python
# coding: utf-8

# # Momentum Calibration for Text Generation
# https://arxiv.org/abs/2212.04257

get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0,1')
# get_ipython().run_line_magic('env', 'CUDA_LAUNCH_BLOCKING=1')

from datetime import datetime
import copy

from tqdm import tqdm
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
cnn_train = load_dataset("cnn_dailymail", '3.0.0', split="train")
cnn_test = load_dataset("cnn_dailymail", '3.0.0', split="test")

# load dataset
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


tokenized_train = cnn_train.map(preprocess_function, batched=True)
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


# create two models
device ='cuda:0'

M = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
G = copy.deepcopy(M)
M = M.to(device)
M.train()
G = G.to(device)
G.eval()


summarizer = pipeline("summarization", model=G, tokenizer=tokenizer, device=device)


def gen_samples(pipe, text, n_samples=8, return_tensors=False):
    res = pipe(text, num_return_sequences=n_samples, diversity_penalty=5.5, 
               num_beam_groups=4, num_beams=n_samples, return_tensors=return_tensors,
               truncation=True)
    print(res)
    if return_tensors:
        return [r['summary_token_ids'] for r in res]
    return [r['summary_text'] for r in res]


# training setup
from torch.optim import AdamW
from transformers import get_scheduler

optimizer = AdamW(M.parameters(), lr=1e-4)

# skip scheduler for now
num_epochs = 20
num_step_per_epoch = 8 * 10
num_gradient_accum = 8
# num_training_steps = num_epochs * num_step_per_epoch / num_gradient_accum
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=10,
#     num_training_steps=num_training_steps,
# )
# print(f'number of training steps: {num_training_steps}')

checkpoint_dir = './checkpoints'

vocab_size = tokenizer.vocab_size
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
pred_max_length = 142
n_samples = 8
use_metrics = lambda x: x['rouge1']

# hyperparameters
lambda_param = 0.001
alpha = 2.0
beta = 10
m = 0.995

from torch.nn import CrossEntropyLoss

def loss_fct(logits, labels):
    output = torch.log_softmax(logits, axis=-1)
    scores = torch.gather(output, 1, labels.unsqueeze(-1)).squeeze(-1)
    label_mask = (labels != pad_token_id).float()
    return -(scores * label_mask).sum(-1) / (label_mask).sum(-1) ** alpha

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
    model=G,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

sample_testset = np.random.choice(tokenized_test, 64)

metrics = trainer.evaluate(sample_testset)
print(metrics)

def shuffle_dataset(dataset):
    while True:
        indicies = np.arange(len(dataset))
        np.random.shuffle(indicies)
        for idx in indicies:
            yield dataset[int(idx)]
                             

dataset = shuffle_dataset(tokenized_test)

for epoch in range(num_epochs):
    # generate samples
    batch = [next(dataset) for _ in range(num_step_per_epoch)]
    
    %%time
    try:
        candidates = gen_samples(summarizer, 
                                 [b['article'] for b in batch[:1]][0], 
                                 return_tensors=True)
        candidates = gen_samples(summarizer, 
                                 [b['article'] for b in batch[:1]][0], 
                                 return_tensors=True)
    except:
        pass
    
    # create dataset
    
    # create trainer and run
    
    # update G
    
    # eval
    
    # repeat
    
    
    print(f'### Epoch {epoch}', end=' ')
    for step in range(num_step_per_epoch):
        # print('.', end='')
        inputs = next(dataset)
        
        inputs = tokenized_train[0]
        
        input_ids = torch.as_tensor(inputs['input_ids']) \
            .to(device) \
            .repeat(n_samples + 1, 1)
        attention_mask = torch.as_tensor(inputs['attention_mask']) \
            .to(device) \
            .repeat(n_samples + 1, 1)
        
        # generate samples
        summarizer.model.to(device)
        samples = gen_samples(summarizer, inputs['article'], return_tensors=True)

        # calculate the rouge score for each sample
        labels = inputs['labels']
        rouge_scores = []
        for sample in samples:
            score = compute_metrics(([sample], [labels]))
            rouge_scores.append(use_metrics(score))
        samples = [sample for score, sample in 
                   sorted(zip(rouge_scores, samples), key=lambda x: x[0])]

        # forward pass
        sample_text = tokenizer.batch_decode(samples, skip_special_tokens=True)
        true_label = inputs['highlights']
        sample_text.append(true_label)
        labels = tokenizer(sample_text, max_length=142, truncation=True,
                           padding=True, return_tensors='pt')
        labels = labels.input_ids.to(device)
        outputs = M(input_ids=input_ids, attention_mask=attention_mask, 
                    labels=labels.to(device))

        # calculate loss
        logits = outputs.logits[:-1] # (batch_size * seq_len * vocab_size)
        s_theta = [F.cross_entropy(logits[i], labels[i], reduction='mean') 
                   for i in range(n_samples)]

        # pairwise_ranking_loss
        pairwise_ranking_loss = 0
        for j in range(n_samples):
            for i in range(j):
                pairwise_ranking_loss += \
                    torch.clip(s_theta[j] - s_theta[i] + (j - i) * lambda_param, 0)

        # MLE loss
        logits = outputs.logits[-1]
        mle_loss = F.cross_entropy(logits, labels[-1], reduction='mean')
        
        # final loss
        loss = pairwise_ranking_loss + beta * mle_loss
        loss.backward()
        
        if (step + 1) % num_gradient_accum == 0 or (step + 1) == num_step_per_epoch:
            print('[Model M] PR loss: {:.2f} MLE loss: {:.2f} total: {:.2f}'.format(
                pairwise_ranking_loss,
                mle_loss.round(decimals=2),
                (pairwise_ranking_loss + mle_loss).round(decimals=2)
            ))
            optimizer.step()
            optimizer.zero_grad()
            
        
    print()
        
    # update G
    for G_param, M_param in zip(G.parameters(), M.parameters()):
        G_param.data.copy_(m * M_param.data + (1.0 - m) * M_param.data)
    G.to(device)

    summarizer = pipeline("summarization", model=G, tokenizer=tokenizer, device=device)
        
    # save model checkpoint
    G.save_pretrained(checkpoint_dir + datetime.now().isoformat())

    trainer = Seq2SeqTrainer(
        model=G,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(sample_testset)
    print(metrics)
