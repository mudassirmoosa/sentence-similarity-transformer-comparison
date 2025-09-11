"""

This file contains codes
(1) to run training loops and 
(2) for loading DataLoaders 
for sentence similarity task.

"""

import torch
from transformers import get_scheduler, set_seed
from datasets import load_dataset
from transformers import DataCollatorWithPadding, set_seed
from torch.utils.data import DataLoader


"""
A custom trainer object to train 
an LLM for sentence similarity task.
"""

class sentence_similarity_trainer():

  def __init__(self, model, train_dataloader, val_dataloader, device, trainer_config, stopping_condition = None):

    '''
      Args:
            model : Transformer model
            train_dataloader : Training data as DataLoader object
            val_dataloader   : Validation data as DataLoader object
            test_dataloader  : Test data as DataLoader object
            device : device
            trainer_config (dict) :
                    trainer_config['optimizer'] : the optimizer function, eg. AdamW
                    trainer_config['num_epochs'] : no of epochs to run
                    trainer_config['learning_rate'] : learning_rate
                    trainer_config['lr_schedule'] (bool) : whether to use learning_rate scheduler
            stopping_condition (either None or a dict):
                    stopping_condition['epoch'] : epoch on which to stop the training
                    stopping_condition['step']  : step on which to stop the training
    '''

    # Initializing the model
    self.model = model
    self.device = device
    self.model.to(device)

    # Initializing the data
    self.train_data = train_dataloader
    self.val_data = val_dataloader

    # Initializing the optimizer
    self.config = trainer_config
    self.optimizer = trainer_config['optimizer'](self.model.parameters(),
                                                   lr = trainer_config['learning_rate'],
                                                   )

    # Initializing training settings
    self.num_epochs = trainer_config['num_epochs']
    self.lr_scheduler = trainer_config['lr_scheduler']
    if self.lr_scheduler:
      self.scheduler = get_scheduler(
          "linear",
          self.optimizer,
          num_warmup_steps=0,
          num_training_steps=self.num_epochs*len(self.train_data),
      )


    # Initializing the stopping condition
    if stopping_condition == None:
        self.stopping_condition = {'epoch': None, 'step': None}
    else:
        self.stopping_condition = stopping_condition
        

  
  # Function to calculate training 
  # loss during training
  def training_evaluation(self):

    self.model.eval()
    with torch.no_grad():

      train_losses = []

      for i, batch in enumerate(self.train_data):
        
        # Getting the batch loss
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        train_losses.append(outputs.loss.item())

      avg_train_loss = sum(train_losses) / len(train_losses)

    self.model.train()

    return avg_train_loss
    
  # Function to calculate valiadation loss 
  # and accuracy during training
  def validation_evaluation(self):

    self.model.eval()
    with torch.no_grad():

      val_losses = []
      val_accuracies = []

      for i, batch in enumerate(self.val_data):
        
        # Getting the batch loss
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        val_losses.append(outputs.loss.item())
        # Getting the batch accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        val_accuracy = (predictions == batch['labels']).float().mean()
        val_accuracies.append(val_accuracy.item())

      avg_val_loss = sum(val_losses) / len(val_losses)
      avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)

    self.model.train()

    return avg_val_loss, avg_val_accuracy
    
  # Running the training loops  
  def train(self):

    self.model.train()
    best_accuracy = 0.0
    best_accuracy_step = 0
    best_accuracy_epoch = 0
    stopping_condition_achieved = False

    for epoch in range(self.num_epochs):

      if stopping_condition_achieved:
          break

      for step, batch in enumerate(self.train_data):

        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
          self.scheduler.step()
        self.optimizer.zero_grad()

        if ((step) % 50 == 0) or (step == len(self.train_data) - 1):
          current_val_loss, current_val_accuracy = self.validation_evaluation()
          current_train_loss = self.training_evaluation()  
          print(f"Epoch {epoch} Step {step} -- training loss: {current_train_loss} --validation loss: {current_val_loss} -- validation accuracy {current_val_accuracy}")
          if current_val_accuracy > best_accuracy:
            best_accuracy = current_val_accuracy
            best_accuracy_epoch = epoch
            best_accuracy_step = step

        if self.stopping_condition['epoch'] == epoch and self.stopping_condition['step'] == step:
            stopping_condition_achieved = True
            break

      
    # Final result
    print(f"The best accuracy was {best_accuracy} after step {best_accuracy_step} of epoch {best_accuracy_epoch}.")
    

    self.model.eval()

    
  def save_model(self, path):

    # saving only the weights of the models
    torch.save(self.model.state_dict(), path)



"""
A custom data loader object for 
sentence similarity task.
"""

class sentence_similarity_dataloaders():

  def __init__(self, tokenizer):

    self.tokenizer = tokenizer
    self.raw_datasets = load_dataset("glue", "mrpc")

  
  def tokenize_function(self, example):
    return self.tokenizer(example["sentence1"], example["sentence2"], truncation=True)

  def get_dataloaders(self):

    # Tokenize datasets and clean up
    tokenized_datasets = self.raw_datasets.map(self.tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # Convert into DataLoader Object
    data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    val_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator)
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=8, collate_fn=data_collator)

    return train_dataloader, val_dataloader, test_dataloader
