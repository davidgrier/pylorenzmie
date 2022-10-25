from CATCH import Estimator


class catchEstimator(Estimator):

    def predict(self, image):
        return Estimator.predict(self, [image])[0]
