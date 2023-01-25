from CATCH import Localizer


class catchLocalizer(Localizer):

    def detect(self, image):
        return Localizer.detect(self, [image])[0]
