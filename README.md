### Introduction

This repository contains a comparative study of fine-tuning pre-trained transformer models for sentence similarity tasks. We compare three models: RoBERTa, DeBERTa, and GPT-2. The main goal of this project is to gain hands-on experience with finetuning hyperparameters to get the best performance. The finetuning of BERT for sentence similarity task was performed in this [Hugging Face (HF) tutorial](https://huggingface.co/learn/llm-course/chapter3/3?fw=pt), though no hyperparameter optimization was performed. So our goal is to beat the results achieved in this tutorial by trying different hyperparameters.

### Data

We use the [Microsoft Research Paraphrase Corpus (MRPC)](https://aclanthology.org/I05-5002.pdf) dataset consisting of $5801$ pairs of sentences, which is available through [HF datasets](https://huggingface.co/datasets/nyu-mll/glue/viewer/mrpc?views%5B%5D=mrpc_train). Almost 63% of this dataset is taken to be the training dataset, almost 7% is taken as the validation dataset, and almost 30% is taken as the test dataset. This dataset was also used in the aforementioned HF tutorial. 


### LLMs as classifiers

The LLM models (RoBERTa, DeBERTa, and GPT-2) are pretrained to predict the next token. However, we now need to use them to classify whether two sentences are similar. To address this, the head of the pre-trained model can be replaced with a randomly initialized binary classifier (i.e., another head that returns two logits), which can be done using HF transformer library by calling `AutoModelForSequenceClassification.from_pretrained` function. In simple words, the pre-trained model (without the trained head) provides us a feature representation of the input sequence, and then this representation is used as an input to the classifier head.

Now given two sentences, we concatenate them together and take it as an input sequence of the model. (Both sentences are separated by the end of the text token.) The full model (i.e. both the pre-trained model and the new classifier head) is then fine-tuned. 

### Experiments

In this project, we focus on three hyperparameters: learning rates, learning rate schedules, and random initialization of classifiers.

Therefore, for each of the models (RoBERTa, DeBERTa, and GPT-2), we perform experiments with three different random initializations of the classifier. For each random initialization, we perform six experiments with different choices of learning rates. Three of these experiments are with constant learning rates of $1\times 10^{-5}$, $3\times 10^{-5}$, and $5\times 10^{-5}$, whereas the other three experiments are with a linearly decreasing learning rate with initial values of $1\times 10^{-5}$, $3\times 10^{-5}$, and $5\times 10^{-5}$. Therefore, in total we perform 18 experiments for each model. 

All other hyperparameters are kept fixed in our experiments. For example, batch size is taken to be 8 and number of epochs is taken to be 3 (though we observe overfitting in the third epoch in most of the experiments.)

All experiments are run on T4 GPU available for free on Google Colab. The implementations of the experiments are in notebooks finetune_Roberta_for_sentence_similarity.ipynb, finetune_deberta_for_sentence_similarity.ipynb, and finetune_gpt2_for_sentence_similarity.ipynb.

### Results

Based on our experiment outputs, we make the following observations. 

1. For **DeBERTa** model, we observe that training with a learning rate which starts from $5\times 10^{-5}$ and linearly decreases to zero leads to the best result if we stop after the second epoch. We achieve the validation (test) accuracy of 90.4% (87.2%) and validation (test) loss of 0.228 (0.323).
2. For **RoBERTa** model, we observe that training with a learning rate which starts from $3\times 10^{-5}$ and linearly decreases to zero leads to the best result if we stop after the second epoch. We achieve the validation (test) accuracy of 88.5% (86.7%) and validation (test) loss of 0.258 (0.318).
3. For **GPT2** model, we observe that training with a constant learning rate of $5\times 10^{-5}$ leads to the best result if we only train for two epochs. We achieve the validation (test) accuracy of 81.1% (80.0%) and validation (test) loss of 0.408 (0.435).

The following table summarizes the results:

| Model Name  | Validation Loss | Validation Accuracy |Test Loss | Test Accuracy |
|-------|----------|----------|----------|----------|
| DeBERTa | 0.228  | 90.4% | 0.323| 87.2%|
| RoBERTa   | 0.258  | 88.5 %   | 0.318| 86.7%|
| GPT2 | 0.408  | 81.1%    | 0.435| 80.0%|

Interestingly, we note that our trained DeBERTa or RoBERTa model perform better than the fine-tuned BERT model in this [HF tutorial](https://huggingface.co/learn/llm-course/chapter3/3?fw=pt), where the validation accuracy of 85.8 was reported. 

It is also interesting to note that our experiments suggest that the *encoder-only* models like DeBERTa and RoBERTa perform better for sentence similarity tasks than the *decoder-only* models like GPT-2. This is probably because the encoder-only models use 'bidirectional' attention, and hence, can attend to both sentences together. On the other hand, the decoder-only models use causal (unidirectional) attention and cannot attend to both sentences simultaneously.

### Saved Weights

The trained model weights for all three models are available on the HF model repo: [https://huggingface.co/mudassirmoosa/sentence-similarity-transformer-comparison/tree/main](https://huggingface.co/mudassirmoosa/sentence-similarity-transformer-comparison/tree/main)

### Future Directions

After doing this project, we wonder whether it would be better to only train the weights of the classifier head (i.e. keeping the weights of the pre-trained model fixed or 'frozen'.) It seems that this would be a special case of a [Parameter-efficient finetuning (PEFT)](https://github.com/huggingface/peft) technique for efficient finetuning. It will be interesting to explore this direction next. 
