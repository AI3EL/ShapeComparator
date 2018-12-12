import random

import numpy as np
from scipy.sparse import linalg
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from scipy import sparse
import os

# TODO : faire un truc pour sauvegarder/lire
# TODO : bien choisir le max
# TODO : comprendre les warnings rouges habituels
# TODO : comprendre le warning "RuntimeError: Factor is exactly singular"
# TODO : mettre des labels pour les couleurs
# TODO : comparaison avec shape-DNA
# TODO : optimiser la construction de l'histogramme
# TODO : but final : faire tourner sur http://segeval.cs.princeton.edu/

# max is not included : last interval is [max-2*step, max-step]
class Histogram :
    def __init__(self, step, min, max):
        #assert (max - min) % step == 0
        #De step^min à step^max
        self.step = step
        self.min = min
        self.max = max
        self.size = (self.max - self.min)
        self.bins = [0] * int((max-min)/step)

    def get_occurence(self, v):
        return self.bins[v]

    def add_occurence(self, v, n=1):
        if self.max > v >= self.min:
            self.bins[int((v-self.min)/self.step)] += n

    @staticmethod
    def dist(h1, h2):
        assert h1.step == h2.step
        assert h1.min == h2.min
        assert h1.max == h2.max
        #min = max(h1.min, h2.min)
        #max = min(h1.max, h2.max)
        return sum([(h1.get_occurence(i) - h2.get_occurence(i))**2 for i in range(0, int((h1.max-h1.min)/h1.step))])**0.5

class Point :
    def __init__(self, coord):
        self.coord = coord
        self.dim = len(coord)

    @staticmethod
    def dist(a, b):
        assert a.dim == b.dim
        return sum([(a.coord[i] - b.coord[i]) ** 2 for i in range(a.dim)]) ** 0.5

    @staticmethod
    def scal(a, b):
        assert a.dim == b.dim
        return sum([a.coord[i] * b.coord[i] for i in range(a.dim)])

    def norm(self):
        return Point.dist(self, Point([0] * self.dim))

    def opp(self):
        return Point([-self.coord[i] for i in range(self.dim)])

    def unit(self):
        norm = self.norm()
        return Point([self.coord[i] / norm for i in range(self.dim)])

    def __add__(self, other):
        assert self.dim == other.dim
        return Point([self.coord[i] + other.coord[i] for i in range(self.dim)])

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        assert self.dim == other.dim
        return Point([self.coord[i] - other.coord[i] for i in range(self.dim)])

    def __rsub__(self, other):
        return (other - self).opp()

    def __truediv__(self, scal):
        assert type(scal) in [float, int]
        return Point([self.coord[i] / scal for i in range(self.dim)])

    def __mul__(self, scal):
        assert type(scal) in [float, int]
        return Point([self.coord[i] * scal for i in range(self.dim)])

    def __rmul__(self, other):
        return self * other

    @staticmethod
    def cross(a, b):
        assert a.dim == 3 and b.dim == 3
        return Point([a.coord[1] * b.coord[2] - a.coord[2] * b.coord[1],
                       b.coord[0] * a.coord[2] - b.coord[2] * a.coord[0],
                       a.coord[0] * b.coord[1] - a.coord[1] * b.coord[0]])

    def __str__(self):
        return str(self.coord)


def blank_split(line):
    i = 0
    res = [""]
    while i < len(line):
        if line[i] == ' ':
            res.append("")
            while line[i] == ' ':
                i+=1
        else:
            res[-1] += line[i]
            i+=1
    return res


class Triangle:
    def __init__(self, points, normal=None, q=None):
        self.a = points[0]
        self.b = points[1]
        self.c = points[2]
        self.normal = normal
        self.points = points  # Indexes of the points
        self.q = q

    def __str__(self):
        return "Points : {0} , {1} , {2}, q = {3}, normal = {4}".format(self.a, self.b, self.c, self.q, self.normal)


class TrianglePair:
    def __init__(self, a, b, c, q1, q2):
        self.a = a
        self.b = b
        self.c = c
        self.q1 = q1
        self.q2 = q2


class Shape:
    def __init__(self, path):
        self.p = []  # Points (around 10⁴ on examples)
        self.normals = []  # Points normals (around 10⁴ on examples)
        self.triangles = []  # Each triangle is a size 3 list of points ~10⁴
        self.tglIdOfPt = [] # tglIdOfPt[i] = ids of trianlges where point of id i appears
        self.crclOfPt = [] # crclOfPt[i] contains all the triangles in tglIdOfPt[i] with a==i and b and c are consecutives TODO : delete
        self.tglPairOfPt = []
        self.M = [[]]
        self.S = []

        self.read(path)
        self.find_q()
        self.build_triangle_pairs()
        self.build_matrices_from_pairs()

    # Assumes vertices are all defined before triangles
    def read(self, path):
        print("Reading")
        with open(path, "r") as file:
            read_vertices = False
            for line in file:
                if line[0] == "v" and line[1] == " ":
                    assert not read_vertices
                    words = blank_split(line)
                    assert len(words) == 4
                    self.p.append(Point([float(words[1]), float(words[2]), float(words[3])]))

                elif line[1] == "n":
                    if not read_vertices:
                        read_vertices = True
                        self.tglIdOfPt = [[] for i in range(len(self.p))]
                    words = blank_split(line)
                    assert len(words) == 4
                    self.normals.append(Point([float(words[1]), float(words[2]), float(words[3])]))

                elif line[0] == "f":
                    words = blank_split(line)
                    assert len(words) == 4
                    words = words[1:]
                    words[2] = words[2][0:-1]
                    for i in range(len(words)):
                        words[i] = words[i].split("//")

                    self.triangles.append(Triangle([int(words[i][0]) - 1 for i in range(3)]))
                    for j in range(3):
                        self.tglIdOfPt[int(words[j][0]) - 1].append(len(self.triangles)-1)
        print("Number of points : " + str(len(self.p)))
        print("Number of triangles : " + str(len(self.triangles)))

    # If the triangle is obtuse, q j is chosen to be the midpoint of the edge opposite to the obtuse angle
    def find_q(self):
        for i, triangle in enumerate(self.triangles):

            # Find normal to the triangle :
            triangle.normal = Point.cross(self.p[triangle.b] - self.p[triangle.a],
                                          self.p[triangle.c] - self.p[triangle.a])

            # Check if triangle is obtuse
            if Point.scal(self.p[triangle.b] - self.p[triangle.a], self.p[triangle.c] - self.p[triangle.a]) <= 0:
                triangle.q = (self.p[triangle.b] + self.p[triangle.c]) / 2
            elif Point.scal(self.p[triangle.c] - self.p[triangle.b], self.p[triangle.a] - self.p[triangle.b]) <= 0:
                triangle.q = (self.p[triangle.c] + self.p[triangle.a]) / 2
            elif Point.scal(self.p[triangle.a] - self.p[triangle.c], self.p[triangle.b] - self.p[triangle.c]) <= 0:
                triangle.q = (self.p[triangle.a] + self.p[triangle.b]) / 2

            else:
                d = (self.p[triangle.b] - self.p[triangle.a]) / 2
                e = (self.p[triangle.c] - self.p[triangle.a]) / 2
                u = Point.cross(triangle.normal, self.p[triangle.b] - self.p[triangle.a])
                v = Point.cross(triangle.normal, self.p[triangle.c] - self.p[triangle.a])
                A = np.array([u.coord, v.coord]).transpose()
                b = np.array((d-e).coord)
                x, y = np.linalg.lstsq(A, b)[0]
                triangle.q = d + x*u

    def build_triangle_circles(self):
        print("Building triangle circles")
        for i in range(len(self.p)):
            not_in_circle = self.tglIdOfPt[i][1:]
            ref_triangle = self.triangles[self.tglIdOfPt[i][0]]
            origin_neighbours = ref_triangle.points[:]
            origin_neighbours.remove(i)
            circle = [Triangle([i] + origin_neighbours, ref_triangle.normal, ref_triangle.q)]

            edge_point = origin_neighbours[1]

            while edge_point != origin_neighbours[0]:
                j = 0
                while edge_point not in self.triangles[not_in_circle[j]].points:
                    j += 1
                ref_triangle = self.triangles[not_in_circle[j]]
                neighbours =ref_triangle.points[:]
                neighbours.remove(i)
                neighbours.remove(edge_point)
                circle.append(Triangle([i, edge_point, neighbours[0]], ref_triangle.normal, ref_triangle.q))
                edge_point = neighbours[0]
                not_in_circle.pop(j)
            assert len(not_in_circle) == 0  # Double check
            self.crclOfPt.append(circle)

    def build_triangle_pairs(self):

        def next_triangle(point, circle):
            for i in range(len(circle)):
                triangle = self.triangles[circle[i]]
                if point in triangle.points:
                    return circle[i]
            return -1

        print("Building triangle pairs")
        for i in range(len(self.p)):
            circle = self.tglIdOfPt[i][:]
            triangle_pairs = []

            while circle:
                ref_triangle = self.triangles[circle.pop()]
                origin_neighbours = ref_triangle.points[:]
                origin_neighbours.remove(i)

                next_id = next_triangle(origin_neighbours[0], circle)
                if next_id != -1:
                    dual = self.triangles[next_id]
                    dual_points = dual.points[:]
                    dual_points.remove(i)
                    dual_points.remove(origin_neighbours[0])
                    triangle_pairs.append(TrianglePair(dual_points[0], origin_neighbours[0], origin_neighbours[1],
                                                       dual.q, ref_triangle.q))

                next_id = next_triangle(origin_neighbours[1], circle)
                if next_id != -1:
                    dual = self.triangles[next_id]
                    dual_points = dual.points[:]
                    dual_points.remove(i)
                    dual_points.remove(origin_neighbours[1])
                    triangle_pairs.append(TrianglePair(origin_neighbours[0], origin_neighbours[1], dual_points[0],
                                                       ref_triangle.q, dual.q))

            self.tglPairOfPt.append(triangle_pairs)

    # Build matrices M and S
    def build_matrices(self):
        print("Building matrices")
        self.M = sparse.lil_matrix((len(self.p), len(self.p)))
        self.S = [0. for i in range(len(self.p))]
        for i in range(len(self.p)):
            circle = self.crclOfPt[i]
            s = 0.
            for j in range(len(circle)):
                # M
                left = circle[j]
                right = circle[(j + 1) % len(circle)]
                assert left.c == right.b  # Check reorder worked
                left_adj = abs(Point.scal(self.p[left.a] - self.p[left.b], (self.p[left.c] - self.p[left.b]).unit()))
                left_hyp = (self.p[left.b] - self.p[left.a]).norm()
                assert left_hyp > left_adj
                if left_adj == 0:  # When left triangle has a 90° angle at b
                    alpha = 0.
                else:
                    alpha = 1 / ((left_hyp / left_adj)**2 - 1) ** 0.5
                right_adj = abs(Point.scal(self.p[right.a] - self.p[right.c], (self.p[right.b] - self.p[right.c]).unit()))
                right_hyp = (self.p[right.c] - self.p[right.a]).norm()
                assert right_hyp > right_adj
                if right_adj == 0:  # When right triangle has a 90° angle at c
                    beta = 0.
                else:
                    beta = 1 / ((right_hyp / right_adj)**2 - 1) ** 0.5
                self.M[i, left.c] = (alpha + beta) / 2

                # S
                d = (self.p[left.b] - self.p[left.a]) / 2
                e = (self.p[left.c] - self.p[left.a]) / 2
                s +=(self.p[left.a] - d).norm() * (left.q - d).norm() / 2 \
                  + (self.p[left.a] - e).norm() * (left.q - e).norm() / 2
            self.S[i] = s

        self.M.tocsr()
        tmp_sum = sparse.csr_matrix((1, len(self.p)))
        for i in range(len(self.p)):
            tmp_sum += self.M[i]
        for i in range(len(self.p)):
            self.M[i, i] = tmp_sum[0, i]

    # Build matrices M and S
    def build_matrices_from_pairs(self):
        print("Building matrices")
        self.M = sparse.lil_matrix((len(self.p), len(self.p)))
        self.S = [0. for i in range(len(self.p))]
        for i in range(len(self.p)):
            triangle_pairs = self.tglPairOfPt[i]
            s = 0.
            for tgl_pair in triangle_pairs:
                # M
                left_adj = abs(
                    Point.scal(self.p[i] - self.p[tgl_pair.a], (self.p[tgl_pair.b] - self.p[tgl_pair.a]).unit()))
                left_hyp = (self.p[tgl_pair.a] - self.p[i]).norm()
                assert left_hyp > left_adj
                if left_adj == 0:  # When left triangle has a 90° angle at b
                    alpha = 0.
                else:
                    alpha = 1 / ((left_hyp / left_adj) ** 2 - 1) ** 0.5
                right_adj = abs(
                    Point.scal(self.p[i] - self.p[tgl_pair.c], (self.p[tgl_pair.b] - self.p[tgl_pair.c]).unit()))
                right_hyp = (self.p[tgl_pair.c] - self.p[i]).norm()
                assert right_hyp > right_adj
                if right_adj == 0:  # When right triangle has a 90° angle at c
                    beta = 0.
                else:
                    beta = 1 / ((right_hyp / right_adj) ** 2 - 1) ** 0.5
                self.M[i, tgl_pair.b] = (alpha + beta) / 2

                # S
                d = (self.p[tgl_pair.b] + self.p[i]) / 2
                s += (self.p[i] - d).norm() * (tgl_pair.q1 - d).norm() / 2 \
                     + (self.p[i] - d).norm() * (tgl_pair.q2 - d).norm() / 2
            self.S[i] = s

        self.M.tocsr()
        tmp_sum = sparse.csr_matrix((1, len(self.p)))
        for i in range(len(self.p)):
            tmp_sum += self.M[i]
        for i in range(len(self.p)):
            self.M[i, i] = - tmp_sum[0, i]
        self.M = -self.M

    # Return the gps of point of index p
    @staticmethod
    def gps(eigs, vectors, p):
        assert len(eigs) == len(vectors[0])
        return Point([vectors[p][i]/(eigs[i]**0.5) for i in range(len(eigs))])

    # Returns the gps of n randomly sampled points among the vertices (avec remise)
    def sample_gps(self, eigs, vectors, n) :
        points = [random.randint(0, len(self.p)-1) for i in range(n)]
        return [Shape.gps(eigs, vectors, points[i]) for i in range(n)]

    # Returns the histogram of all the distances between cloud and cloud2 with step "step"
    @staticmethod
    def compute_histogram(cloud, cloud2, step, min_, max_):
        distances = []
        for i in range(len(cloud)):
            for j in range(len(cloud2)):
                distances.append(Point.dist(cloud[i], cloud2[j]))
        res = Histogram(step, min_, max_)
        for v in distances:
            res.add_occurence(v)
        return res

    # Returns a matrix (list of list) with the histograms constructed via the balls construction
    # Params :
    # - d : dimensions kept in GPS = eigenvalues kept
    # - m : number of balls
    # - n : number of points sampled among the vertices
    # - step : step used to build the histogram
    # - max : max norm of the points taken into account (determines the radius of the biggest sphere)
    def compute_histograms(self, d, m, n, step, max_):
        print("Diagonalizing")
        sphere_radius = max_/m
        eigs, vectors = linalg.eigsh(self.M, d, sparse.dia_matrix((self.S, [0]), shape=(len(self.p), len(self.p))), sigma=0, which='LM')  # Computes the d smallest eigenvalues
        eigs = eigs[1:]
        vectors = vectors[:, 1:]

        # Normalize vector
        for i in range(vectors.shape[1]):
            vectors[:, i] /= Point(vectors[:, i]).norm()

        gps_points = self.sample_gps(eigs, vectors, n)
        gps_points.sort(key=lambda x: x.norm())
        clouds = [[gps_points[j] for j in range(n) if sphere_radius*i < gps_points[j].norm() < sphere_radius*(i+1)]for i in range(m)]
        print("Calculs histogrammes")
        histos = [[None]*m for i in range(m)]
        for i in range(m):
            for j in range(i+1):
                histos[i][j] = Shape.compute_histogram(clouds[i], clouds[j], step, sphere_radius*max(0, (j-i-1)), max_ + sphere_radius*max(0, (j-i-1)))
        for i in range(m):
            for j in range(i):
                histos[j][i] = histos[i][j]
        #print_histos(histos[0])
        return histos

def print_histos(histo):
    couleurs = ['b', 'r', 'g']
    for j in range(1):
        absc = [i*histo[j].step for i in range(int((histo[j].max -histo[j].min)/histo[j].step))]
        ords = [histo[j].bins[i] for i in range(len(histo[j].bins))]
        plt.plot(absc, ords, couleurs[j])
    plt.show()

def compute_histos(pathes):
    res = []
    max_ = 3
    step = max_ / 100.
    for path in pathes:
        print("Beginning histogram {0}".format(path))
        shape = Shape(path)
        res.append(shape.compute_histograms(15, 8, 1000, step, max_))
    return res


def compute_dists(histos):
    print("Calcul des distances")
    n = len(histos)
    m = len(histos[0])
    dists = [[0]*n for i in range(n)]
    for i in range(n):
        for j in range(i):
            for k in range(m):
                dists[i][j] += sum(Histogram.dist(histos[i][k][h], histos[j][k][h]) for h in range(m))
    return dists

def plot_gps_mds(path):
    print("Begin plot of path : " + path)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    dirs = os.listdir(path)
    files = []
    for dir in dirs:
        assert os.path.isdir(path + '/' + dir)
        files.append(os.listdir(path + '/' + dir))
    pathes = []
    for i in range(len(dirs)):
        pathes += [path + '/' + dirs[i] + '/' + file for file in files[i]]
    print(pathes)
    histos = compute_histos(pathes)
    dists = compute_dists(histos)
    print("DIST")
    print(dists)
    embedding = MDS()
    arraydists = np.asarray(dists)
    resultat = embedding.fit_transform(arraydists)
    print(resultat)

    indices = [0]
    for i in range(len(dirs)):
        indices.append(indices[-1] + len(files[i]))
    for i in range(len(dirs)):
        absc = [resultat[i][0] for i in range(indices[i], indices[i+1])]
        ords = [resultat[i][1] for i in range(indices[i], indices[i+1])]
        plt.plot(absc, ords, colors[i] + 'o')
