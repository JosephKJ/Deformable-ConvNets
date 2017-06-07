import _init_paths
from utils.symbol import Symbol


class student_symbol(Symbol):

    def __init__(self):
        Symbol.__init__(self)
        print 'inside student_symbol __init__'

    def get_symbol(self, cfg, is_train=True):
        print 'inside get_symbol'

    def init_weights(self, cfg, arg_params, aux_params):
        print 'inside init_weights'
