import random

import torch
import torch.nn as nn
import torch.optim as optim

# import custom models



class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################

        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, source, out_seq_len = None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        seq_len = source.shape[1]
        if out_seq_len is None:
            out_seq_len = seq_len

        
        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden weights being fed into the decoder   #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################
        
        outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size)

        if self.decoder.model_type == "RNN":
            output, hidden = self.encoder(source)        
            hidden = hidden[:,-1,:].unsqueeze(0)
            input = source[0,:1].unsqueeze(0)
        
            for ts in range(out_seq_len):
                output, hidden = self.decoder(input, hidden)
                outputs[:,ts] = output
                hidden = hidden[:,-1,:].unsqueeze(0)
                input = output.argmax(1).unsqueeze(0)

        
        if self.decoder.model_type == "LSTM":
            output, (hidden,cell) = self.encoder(source)
            input = source[:,0].unsqueeze(-1)
            
            for i in range(out_seq_len):
                output, (hidden,cell) = self.decoder(input, (hidden,cell))
                outputs[:,i] = output
                input = output.argmax(1).unsqueeze(-1)
                
                
                
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs



        

