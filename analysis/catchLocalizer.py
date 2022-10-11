from CATCH import Localizer


class catchLocalizer(Localizer):

    def __init__(self, **kwargs):
        super(catchLocalizer, self).__init__(**kwargs)

    def detect(self, image):
        if image is None:
            return []
        return Localizer.detect(self, [100*image])[0]
