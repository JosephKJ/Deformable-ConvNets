import _init_paths
from utils.symbol import Symbol

class Student(Symbol):

    def __init__(self):
        Symbol.__init__(self)

    def get_symbol(self, cfg, is_train=True):
        pass

    def init_weights(self, cfg, arg_params, aux_params):
        pass

