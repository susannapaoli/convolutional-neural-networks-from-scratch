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
        kernel = self.kernel_size 
        stride = self.stride

        H_out = int(((H-kernel) / stride ) + 1)
        W_out = int(((W-kernel) / stride ) + 1)
        out = np.zeros((N, C, H_out, W_out))
        
        for i in range(N):
          for j in range(C):
            for l in range(0, H_out):
              for m in range(0, W_out):
                pool = x[i, j, l*stride:l*stride+kernel, m*stride:m*stride+kernel]
                value = np.amax(pool)

                out[i,j,l,m] = value
          

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
        N,C,H,W = x.shape
        dx = np.zeros((N,C,H,W))
        
        for i in range(N):
          for j in range(C):
            for l in range(H_out):
              for m in range(W_out):
                #max_ind = np.argmax(x[i,j,l*self.stride:l*self.stride+self.kernel_size, m*self.stride:m*self.stride+self.kernel_size])
                #a_g, d_g, i_g, j_g,  = np.unravel_index(max_ind, (x.shape[0], x.shape[1], self.kernel_size, self.kernel_size))
                i_g, j_g = np.where(np.max(x[i,j,l*self.stride:l*self.stride+self.kernel_size, m*self.stride:m*self.stride+self.kernel_size]) == x[i,j,l*self.stride:l*self.stride+self.kernel_size, m*self.stride:m*self.stride+self.kernel_size])
                #print(dx[i,j,l*self.stride:l*self.stride+self.kernel_size:,m*self.stride:m*self.stride+self.kernel_size])
                dx[i,j,l*self.stride:l*self.stride+self.kernel_size:,m*self.stride:m*self.stride+self.kernel_size][i_g[0], j_g[0]] = dout[i,j,l,m]

        self.dx = dx 
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
