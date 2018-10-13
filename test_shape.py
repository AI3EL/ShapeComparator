from unittest import TestCase
from main import *


class TestShape(TestCase):
    def test_shape(self):
        lion = Shape("../lion-poses/lion-reference.obj")
        lion.compute_histograms(2, 2, 100, 0.1)
