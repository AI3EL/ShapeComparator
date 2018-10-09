from unittest import TestCase
from main import *


class TestShape(TestCase):
    def test_shape(self):
        d = 15
        m = 8
        s = Shape("../lion-poses/lion-reference.obj")
        lambdas, vectors = linalg.eigh(s.M, np.diag(s.S), eigvals=(0, 4))
        vectors = vectors.transpose()
        # Project in dimension d
        vectors = [vectors[i][:d] for i in range(len(vectors))]
        # TODO : check vectors are orthogonal according to S