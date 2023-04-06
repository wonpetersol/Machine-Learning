import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################

        N, C, H, W = x.shape

        H_out = int(((H - self.kernel_size) / self.stride) + 1)
        W_out = int(((W - self.kernel_size) / self.stride) + 1)

        out = np.zeros((N, C, H_out, W_out))

        for h in range(H_out):
            for w in range(W_out):
                hstart = h * self.stride
                hend = hstart + self.kernel_size
                wstart = w * self.stride
                wend = wstart + self.kernel_size

                out[:,:,h,w] = np.max(x[:,:,hstart:hend, wstart:wend],axis=(2,3))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        N, C, H, W = x.shape

        H_out = int(((H - self.kernel_size) / self.stride) + 1)
        W_out = int(((W - self.kernel_size) / self.stride) + 1)

        self.dx = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):            
                        hstart = h * self.stride
                        hend = hstart + self.kernel_size
                        wstart = w * self.stride
                        wend = wstart + self.kernel_size
                        print("######################")
                        print(dout.shape)
                        print(self.dx[n,c,hstart:hend, wstart:wend].shape)
                        print("######################")
                        self.dx[n,c,hstart:hend, wstart:wend] += (dout[n,c,h,w]*(x[n,c,hstart:hend, wstart:wend] == np.max(x[n,c,hstart:hend, wstart:wend])))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
