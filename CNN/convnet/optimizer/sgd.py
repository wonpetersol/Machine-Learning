from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################

                self.grad_tracker[idx]["dw"] = (self.momentum * self.grad_tracker[idx]["dw"] - self.learning_rate * m.dw)
                m.dw = self.grad_tracker[idx]["dw"]
                m.weight = m.weight + self.grad_tracker[idx]["dw"]

                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################

                self.grad_tracker[idx]["db"] = (self.momentum * self.grad_tracker[idx]["db"] - self.learning_rate * m.db)
                m.db = self.grad_tracker[idx]["db"]
                m.bias = m.bias + self.grad_tracker[idx]["db"]

                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
