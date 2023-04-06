import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        x_pads = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0))

        N, C, H, W = x.shape
        C_out, C_in, H_ker, W_ker = self.weight.shape

        H_out = int(((H + 2 * self.padding - H_ker) / self.stride) + 1)
        W_out = int(((W + 2 * self.padding - W_ker) / self.stride) + 1)

        out = np.zeros([N, C_out, H_out, W_out])

        for c in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    hstart = h * self.stride
                    hend = hstart + H_ker
                    wstart = w * self.stride
                    wend = wstart + W_ker
                    out[:,c,h,w] = np.sum(x_pads[:, :, hstart:hend, wstart:wend] * self.weight[c, :, :, :],axis=(1,2,3)) + self.bias[c]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        x_pads = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant',  constant_values=(0))
        N, C, H, W = x.shape
        C_out, C_in, H_ker, W_ker = self.weight.shape

        H_out = int(((H + 2 * self.padding - H_ker) / self.stride) + 1)
        W_out = int(((W + 2 * self.padding - W_ker) / self.stride) + 1)

        self.dx = np.zeros_like(x)
        self.dw = np.zeros_like(self.weight)
        self.db = np.sum(dout, axis=(0,2,3))
        dx_pads = np.zeros_like(x_pads)

        for h in range(H_out):
            for w in range(W_out):
                for c in range(C_out):
                    hstart = h * self.stride
                    hend = hstart + H_ker
                    wstart = w * self.stride
                    wend = wstart + W_ker
                    
                    self.dw[c,:,:,:] += np.sum(x_pads[:,:,hstart:hend,wstart:wend]*(dout[:,c,h,w])[:,None,None,None],axis=0)
                    
                for n in range(N):
                    dx_pads[n, :, hstart:hend, wstart:wend] += np.sum(self.weight[:,:,:,:]*(dout[n,:,h,w])[:,None,None,None],axis=0)
        self.dx = dx_pads[:,:,self.padding:-self.padding,self.padding:-self.padding]
                

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
