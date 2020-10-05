import random
import torch
import numpy as np

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def encode_sentences(sentences, labels, max_length, tokenizer):
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = max_length,   # Pad & truncate all sentences.
                          truncation = True,
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt')     # Return pytorch tensors.

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def train_val_loaders(sentences, labels, tokenizer, val_split = True, train_prop = 0.9, batch_size = 32, max_length = 32):
    input_ids, attention_masks, labels = encode_sentences(sentences, labels, max_length, tokenizer)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    if val_split:
        # Calculate the number of samples to include in each set.
        train_size = int(train_prop * len(dataset))
        val_size = len(dataset) - train_size
        
        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # train dataloader
        train_dataloader = DataLoader(train_dataset,
                                      sampler = RandomSampler(train_dataset),
                                      batch_size = batch_size)

        validation_dataloader = DataLoader(val_dataset,
                                           sampler = SequentialSampler(val_dataset),
                                           batch_size = batch_size)

        return train_dataloader, validation_dataloader

    else:
        # train dataloader
        train_dataloader = DataLoader(dataset,
                                      sampler = RandomSampler(dataset),
                                      batch_size = batch_size)

        return train_dataloader, None


def train(model, data, labels, tokenizer, val_split = True, train_prop = 0.9, batch_size = 32, 
          max_length = 64, epochs = 2, random_seed = None, device = torch.device("cuda")):

    train_dataloader, eval_dataloader = train_val_loaders(data, 
                                                        labels,
                                                        tokenizer,
                                                        val_split, 
                                                        train_prop, 
                                                        batch_size, 
                                                        max_length)

    optimizer = AdamW(model.parameters(),
                    lr = 2e-5,
                    eps = 1e-8)

    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0,
                                              num_training_steps = total_steps)

    # set random seed on all devices
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


    for epoch in range(epochs):
        total_train_loss = 0
        model.train()

        # Train
        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and step != 0:
                print(step, round(total_train_loss / step, 2))
      
            # 1,2. Unpack our data inputs and labels and load onto GPU (device)
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
      
            # 3. Clear out the gradients calculated in the previous pass.
            #    In pytorch the gradients accumulate by default (useful for things like RNNs) 
            #    unless you explicitly clear them out.
            model.zero_grad()
      
            # 4. Forward pass
            loss, logits = model(batch_input_ids, 
                                 token_type_ids = None, 
                                 attention_mask = batch_input_mask, 
                                 labels = batch_labels)
      
            # 5. Backward pass
            loss.backward()
      
            # Clip gradients to 1.0 to avoid exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      
            # 6. Tell the network to update parameters with optimizer.step()
            optimizer.step()
            scheduler.step()
      
            # 7. Track variables for monitoring progress
            total_train_loss += loss.item()
      
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

        # calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(len(train_dataloader), round(avg_train_loss))

        # Evaluation
        print('\nEpoch', epoch + 1, 'Validation')

        total_eval_loss = 0
        total_eval_accuracy = 0

        model.eval()

        if val_split:
            for batch in eval_dataloader:
          
                # 1,2. Unpack our data inputs and labels and load onto GPU (device)
                batch_input_ids = batch[0].to(device)
                batch_input_mask = batch[1].to(device)
                batch_labels = batch[2].to(device)
          
                # 3. Forward pass (feed input data through the network)
                # stop calculating gradients
                with torch.no_grad():
            
                    # 4. Compute loss on our validation data and track variables for monitoring progress
                    # (forward pass)
                    loss, logits = model(batch_input_ids, 
                                         token_type_ids = None, 
                                         attention_mask = batch_input_mask,
                                         labels = batch_labels)
                
                total_eval_loss += loss.item()
          
                # Move logits and labels to
                logits = logits.detach().cpu().numpy()
                label_ids = batch_labels.to('cpu').numpy()
          
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            # average validation loss and Accuracy
            avg_eval_accuracy = total_eval_accuracy / len(eval_dataloader)
            avg_eval_loss = total_eval_loss / len(eval_dataloader)
        
            print('Validation Loss:', round(avg_eval_loss, 2))
            print('Validation Accuracy:', round(avg_eval_accuracy, 2))
        print()


def predict(model, sentences, labels, tokenizer, batch_size = 32, max_length = 32, 
            device = torch.device("cuda")):
   
    # preprocess sentences
    input_ids, attention_masks, labels = encode_sentences(sentences, labels, max_length, tokenizer)
  
    # Set the batch size.   
    batch_size = batch_size  
  
    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, 
                                       sampler = prediction_sampler, 
                                       batch_size = batch_size)
  

    ### Prediction
  
    # Put model in evaluation mode
    model.eval()
  
    predictions = []
  
    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        batch_input_ids, batch_input_mask, batch_labels = batch
        
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(batch_input_ids, token_type_ids = None, 
                            attention_mask = batch_input_mask)
    
        logits = outputs[0]
    
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions += [np.argmax(scores) for scores in logits]
        labels = list(labels.numpy())

    return predictions, labels