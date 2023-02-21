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
        N, C, H, W = x.shape
        kernel = self.kernel_size
        pd = self.padding
        stride = self.stride

        HH = int(((H - kernel + 2*pd) / stride ) + 1)
        WW = int(((W - kernel + 2*pd) / stride ) + 1)

        x_pad= np.pad(x, ((0,0), (0,0), (pd, pd), (pd, pd)))
        out = np.zeros((N, self.out_channels, HH, WW))

        for i in range(N):
          for j in range(self.out_channels):
            v = 0
            for l in range(0, H, stride):
              t = 0
              for m in range(0, W, stride):
                out[i,j,v,t] = np.sum(x_pad[i, :, l:l+kernel, m:m+kernel] * self.weight[j, :, :, :]) + self.bias[j]
                t += 1
              v += 1
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
        N, C, H, W = x.shape
        F, J, HH, WW = self.weight.shape
        Q = self.bias.shape

        kernel = self.kernel_size
        pd = self.padding
        stride = self.stride

        dx = np.zeros((N,C,H,W))
        dw = np.zeros((F, J, HH, WW))
        db = np.zeros(Q)

        x_pad = np.pad(x, ((0,0), (0,0), (pd, pd), (pd,pd)))
        dx_pad = np.pad(dx, ((0,0), (0,0), (pd, pd), (pd, pd)))

        for i in range(N):
          for j in range(self.out_channels):
            for l in range(0, H, stride):
              for m in range(0, W, stride):
                dx_pad[i, :, l:l+HH, m:m+WW] += self.weight[j, :, :, :] * dout[i,j,l,m]
                dw[j, :, :, :] += x_pad[i, :, l:l+HH, m:m+WW] * dout[i,j,l,m]
                db[j] += dout[i,j,l,m]
        
        
        dx_pad = dx_pad[:,:, 1:-1, 1:-1]
        dx = dx_pad
        self.dx = dx
        self.dw = dw
        self.db = db

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################