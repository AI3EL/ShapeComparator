from unittest import TestCase
from main import *


class TestShape(TestCase):
    def test_shape(self):
        lion_histos = []
        for i in range(1, 6):
            print("Beginning lion histo {0}".format(i))
            lion = Shape("../elephant-poses/elephant-0{0}.obj".format(i))
            lion_histos.append(lion.compute_histograms(2, 2, 100, 10))
        Shape.plot_histos(lion_histos)
