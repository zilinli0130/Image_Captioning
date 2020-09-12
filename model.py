import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer that turns words into embedded token vectors of a specified size
        self.Embedding = nn.Embedding(vocab_size, embed_size)

    
        # LSTM layer accepts the embedded token vectors and process them to get an output 
        self.LstmLayer = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True) #If batch_first = True, then the input and output tensors are provided as (batch, seq, input_size)

      
        # turns the output vector in vacab_size with distribution scores
        self.OutputDim2VocabSize = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        
        # map the captions to embedded token vectors with specified size
        embed = self.Embedding(captions[:,:-1])  
        

        # embedded token vectors as inputs
        inputs = torch.cat((features.unsqueeze(dim=1),embed), dim=1)
    

        # LSTM layer outputs
        lstm_out, _ = self.LstmLayer(inputs);   

        # turns the output vector in vacab_size with distribution scores
        outputs = self.OutputDim2VocabSize(lstm_out);    
        
        # outputs.shape = torch.size[(batch_size captions.shape[1] vocab_size])
        return outputs 
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # initialize the empty caption
        caption = []
        
        # initialize hidden states
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))

       
        
        # batch_size = 1, sequence_length = 1
        for i in range(max_len):
            
            # size of inputs = [batch_size sequence_length embed_size]
            # size of lstm_output = [batch_size caption_length hidden_size]
            lstm_output, hidden = self.LstmLayer(inputs, hidden) 
            #print('iteration %d', i)
            #print('The size of inputs is: \n', inputs.shape)
            #print('The size of lstm_output is \n:', lstm_output.shape)

          
            
            
            # size of outputs = [batch_size sequence_length vocab_size]
            outputs = self.OutputDim2VocabSize(lstm_output)
            #print('The size of outputs is \n:', outputs.size()) 
            #print('--- \n')
            
            # size of outpus = [sequence_length vocab_size]
            outputs = outputs.squeeze(1) 
            
            # size of outputs = [sequence_length 1] since the word with highest probability is choosen
            choosen_word  = outputs.argmax(dim=1)    
            
            # append the result to caption 
            caption.append(choosen_word.item())
            
            # ready for next loop
            inputs = self.Embedding(choosen_word.unsqueeze(0))  
          
        return caption