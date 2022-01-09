import os

from ae.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.print_interval = 25
        self.eval_interval = 20
        self.max_epoch = 300
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.mem_type = 'vq'
        self.n_embeddings = 10800
        self.mem_dim = 256
        self.commitment_cost = 0.25
        self.decay = 0.999
        self.epsilon = 1e-5
        self.chnum_in = 1
