from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        self.velocity_weight = {}
        self.velocity_bias = {}

        for idx, m in enumerate(model.modules):
          if hasattr(m, 'weight'):
            self.velocity_weight[m] =  [0.0 for i in range(m.weight.shape[1]) for j in range(m.weight.shape[0])]
          if hasattr(m, 'bias'):
            self.velocity_bias[m] = [0.0 for i in range(m.bias.shape[0])]

        # initialize the velocity terms for each weight

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        #self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                dw = m.dw 
                self.velocity_weight[m] = [self.momentum * v - self.learning_rate * dw_i for v, dw_i in zip(self.velocity_weight[m], dw)]
                m.weight += self.velocity_weight[m]
              
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                db = m.db
                self.velocity_bias[m] = [self.momentum * v - self.learning_rate * db_i for v, db_i in zip(self.velocity_bias[m], db)]
                m.bias += self.velocity_bias[m]
              
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
