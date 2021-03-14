from CATCH import Localizer


class catchLocalizer(Localizer):

    def __init__(self, **kwargs):
        super(catchLocalizer, self).__init__(**kwargs)

    def detect(self, image):
        d = Localizer.detect(self, [100*image])
        return d[0]
