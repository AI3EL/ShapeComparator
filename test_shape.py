from unittest import TestCase
from main import *


class TestShape(TestCase):
    def test_shape(self):
        plot_gps_mds("../test")
        plt.show()
