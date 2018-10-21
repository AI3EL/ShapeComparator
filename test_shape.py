from unittest import TestCase
from main import *
import matplotlib.pyplot as plt

class TestShape(TestCase):
    def test_shape(self):
        lion = Shape("../lion-poses/lion-reference.obj")
        resultat = lion.compute_histograms(2, 2, 100, 10)
        print(resultat)
        plt.plot(resultat[0],resultat[1])