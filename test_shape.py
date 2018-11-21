from unittest import TestCase
from main import *


class TestShape(TestCase):
    def test_shape(self):
        plot_animals(["lion", "elephant"], [6, 3])
