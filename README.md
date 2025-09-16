# Introduction

This is the repository for a comparative study of finetuning various pre-trained transformers models (RoBERTa, DeBERTa, and GPT-2) for sentence similarity tasks. The main goal of this project is to gain hands-on experience with finetuning hyperparameters to get the best performance. The finetuning of BERT for sentence similarity task was performed in this [Hugging Face (HF) tutorial](https://huggingface.co/learn/llm-course/chapter3/3?fw=pt), though no hyperparameter optimization was performed. So our goal is to beat the results achieved in this tutorial by trying different hyperparameters.

### Data

We use the [Microsoft Research Paraphrase Corpus (MRPC)](https://aclanthology.org/I05-5002.pdf) dataset consisting of $5801$ pair of sentences, which is available through [HF](https://huggingface.co/datasets/nyu-mll/glue/viewer/mrpc?views%5B%5D=mrpc_train). This dataset was also used in the aforementioned HF tutorial. 

### LLMs as classifiers

The LLM models (RoBERTa, DeBERTa, and GPT-2) are pretrained to predict the next token. However, we now need to use them to classify whether two sentences are similar. To remedy this, the head of the pre-trained model can be replaced with a randomly initialized binary classifier (i.e., another head that returns two logits), which can be done using HF transformer library by calling `AutoModelForSequenceClassification.from_pretrained` function. In simple words, we use the pretrained model to give us a feauture representation of the pair of sentences, and then we take that representation as an input of the classifier. Finally, we modify all the weights (both of the pre-trained model and the new classifier head) while training the model. 

### Experiments

In this project, we focus on three hyperparameters: learning rates, learning rate schedules, and random initialization of classifiers.

Therefore, for each of the models (RoBERTa, DeBERTa, and GPT-2), we perform experiments with three different random initialization of the classifier. For each of the random initializations, we perform six experiments with different choices of learning rates. Three of these experiments are with constant learning rates of $1\times 10^{-5}$, $3\times 10^{-5}$, and $5\times 10^{-5}$, whereas the other three experiments are with a linearly decreasing learning rate with initial values of $1\times 10^{-5}$, $3\times 10^{-5}$, and $5\times 10^{-5}$. Therefore, in total we perform 18 experiments for each model, and each experiment runs for 3 epochs.

The experiments are performed in notebooks finetune_Roberta_for_sentence_similarity.ipynb, finetune_deberta_for_sentence_similarity.ipynb, and finetune_gpt2_for_sentence_similarity.ipynb.

### Results

Based on our experiments outputs, we make the following observations. 

1. For **DeBERTa** model, we observe that training with a learning rate which starts from $5\times 10^{-5}$ and linearly decreases to zero leads to the best result if we stop after the second epoch. We achieve the validation (test) accuracy of 90.4% (87.2%) and validation (test) loss of 0.228 (0.323).
2. For **RoBERTa** model, we observe that training with a learning rate which starts from $3\times 10^{-5}$ and linearly decreases to zero leads to the best result if we stop after the second epoch. We achieve the validation (test) accuracy of 88.5% (86.7%) and validation (test) loss of 0.258 (0.318).
3. For **GPT2** model, we observe that training with a constant learning rate of $5\times 10^{-5}$ leads to the best result if we only train for two epochs. We achieve the validation (test) accuracy of 81.1% (80.0%) and validation (test) loss of 0.408 (0.435).

Interestingly, we note that our trained DeBERTa or RoBERTa model perform much better than the trained BERT model in this [HF tutorial](https://huggingface.co/learn/llm-course/chapter3/3?fw=pt), where the validation accuracy of 85.8 was reported. 

It is also interesting to note that our experiments suggest that the *encoder-only* models like DeBERTa and RoBERTa perform better for sentence similarity task than then *decoder-only* models like GPT-2. 

### Saved Weights

The trained model weights for all three models are available on the HF model repo: [https://huggingface.co/mudassirmoosa/sentence-similarity-transformer-comparison/tree/main](https://huggingface.co/mudassirmoosa/sentence-similarity-transformer-comparison/tree/main)

### Future Directions

After finishing the main experiments in this project, we realize if it would be better to only train the weights of the classifier head (i.e. keeping the weights of the pre-trained model fixed or 'frozen'.) It seems that this would be a special case of a [Parameter-efficient finetuning (PEFT)](https://github.com/huggingface/peft) technique for efficient finetuning. We hope to explore this direction next. 
