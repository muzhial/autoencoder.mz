import os

from ae.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.print_interval = 100
        self.eval_interval = 101
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
