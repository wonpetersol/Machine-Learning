import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization


    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes as you wish here.                      #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   You also need to include correct activation functions                      #
        #   Initialize the gates in the order above!                                   #
        #   Initialize parameters in the order they appear in the equation!            #
        ################################################################################
        
        #i_t: input gate
        self.ii_weight = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.ii_bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.hi_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.hi_bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.ii_sigmoid = torch.nn.Sigmoid()

        # f_t: the forget gate
        self.if_weight = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.if_bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.hf_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.hf_bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.if_sigmoid = torch.nn.Sigmoid()

        # g_t: the cell gate
        self.ig_weight = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.ig_bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.hg_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.hg_bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.ig_tanh = torch.nn.Tanh()
        
        # o_t: the output gate
        self.io_weight = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.io_bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.ho_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.ho_bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.io_sigmoid = torch.nn.Sigmoid()

        self.c_t_tanh = torch.nn.Tanh()

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################

        h_t = torch.zeros(self.hidden_size,self.hidden_size)
        c_t = torch.zeros(self.hidden_size,self.hidden_size)
        
        for t in range(len(x)-1):
            self.i_t = self.ii_sigmoid((x[:,t] @ self.ii_weight) + self.ii_bias + (h_t @ self.hi_weight) + self.hi_bias)
            self.f_t = self.if_sigmoid((x[:,t] @ self.if_weight) + self.if_bias + (h_t @ self.hf_weight) + self.hf_bias)
            self.g_t = self.ig_tanh((x[:,t] @ self.ig_weight) + self.ig_bias + (h_t @ self.hg_weight) + self.hg_bias)
            self.o_t = self.io_sigmoid((x[:,t] @ self.io_weight) + self.io_bias + (h_t @ self.ho_weight) + self.ho_bias)

            c_t = (self.f_t * c_t) + (self.i_t * self.g_t)
            h_t = (self.o_t * self.c_t_tanh(c_t))

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

