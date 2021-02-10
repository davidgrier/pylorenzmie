class Estimator(object):

    def __init__(self,
                 z_p=None,
                 a_p=None,
                 n_p=None):
        self.z_p = z_p or 200.
        self.a_p = a_p or 1.
        self.n_p = n_p or 1.5

    def predict(self, *args):
        return dict(z_p=self.z_p,
                    a_p=self.a_p,
                    n_p=self.n_p)
