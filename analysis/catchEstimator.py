from CATCH import Estimator
import numpy as np


class catchEstimator(Estimator):

    def __init__(self, **kwargs):
        super(catchEstimator, self).__init__(**kwargs)

    def predict(self, image):
        p = Estimator.predict(self, [(100.*image).astype(np.uint8)])
        print(p)
        return p[0]
