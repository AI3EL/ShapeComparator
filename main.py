import random
import numpy as np
from scipy.sparse import linalg
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import sparse
import os
import kmeans


# max is not included : last interval is [max-2*step, max-step]
class Histogram :
    def __init__(self, step, min, max):
        self.step = step
        self.min = min
        self.max = max
        self.bins = [0] * (int((max-min)/step)+1)
        self.not_in_range = 0

    def add_occurence(self, v, n=1):
        if self.max > v >= self.min:
            self.bins[int((v-self.min)/self.step)] += n
        else:
            self.not_in_range += 1

    def to_point(self):
        return Point([float(b) for b in self.bins])

    @staticmethod
    def dist(h1, h2):
        assert h1.step == h2.step
        assert h1.min == h2.min
        assert h1.max == h2.max
        return sum([(h1.bins[i] - h2.bins[i])**2 for i in range(0, int((h1.max-h1.min)/h1.step))])**0.5

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
            i += 1
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

        if path[-3:] == 'off':
            self.read_off(path)
        elif path[-3:] == 'obj':
            self.read_obj(path)
        else:
            raise ValueError('Unsupported file extension : .' + path[-3:])
        self.find_q()
        self.build_triangle_pairs()
        self.build_matrices_from_pairs()

    # Assumes vertices are all defined before triangles
    def read_obj(self, path):
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

    def read_off(self, path):
        print("Reading")
        with open(path, "r") as file:
            assert file.readline()[:3] == "OFF"
            num_vertices, num_triangles, _ = [int(w) for w in blank_split(file.readline())]
            self.tglIdOfPt = [[] for i in range(num_vertices)]

            for i in range(num_vertices):
                line = file.readline()
                words = blank_split(line)
                assert len(words) == 3
                self.p.append(Point([float(words[0]), float(words[1]), float(words[2])]))

            for i in range(num_triangles):
                line = file.readline()
                words = blank_split(line)
                assert len(words) == 4
                assert words[0] == '3'
                words = words[1:]
                words = [int(word) for word in words]
                self.triangles.append(Triangle(words))
                for word in words:
                    self.tglIdOfPt[word].append(len(self.triangles) - 1)
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
                x, y = np.linalg.lstsq(A, b, rcond=None)[0]
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
                if left_hyp > left_adj: # does not happen sometime because of numeric errors
                    if left_adj == 0:  # When left triangle has a 90° angle at b
                        alpha = 0.
                    else:
                        alpha = 1 / ((left_hyp / left_adj) ** 2 - 1) ** 0.5
                else:
                    alpha = 0.0
                right_adj = abs(
                    Point.scal(self.p[i] - self.p[tgl_pair.c], (self.p[tgl_pair.b] - self.p[tgl_pair.c]).unit()))
                right_hyp = (self.p[tgl_pair.c] - self.p[i]).norm()
                if right_hyp > right_adj:
                    if right_adj == 0:  # When right triangle has a 90° angle at c
                        beta = 0.
                    else:
                        beta = 1 / ((right_hyp / right_adj) ** 2 - 1) ** 0.5
                else:
                    beta = 0.0
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
            assert max_ > v >= min_
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
        histos = [[None]*m for i in range(m)]
        print("Number of points by sphere :")
        print([len(cloud) for cloud in clouds])
        total_not_in_range = 0
        print("Computing histograms")
        for i in range(m):
            for j in range(i+1):
                histos[i][j] = Shape.compute_histogram(clouds[i], clouds[j], step, sphere_radius*max(0, (i-j-1)), sphere_radius*(i+j+2))
                total_not_in_range += histos[i][j].not_in_range
        print("Total not in range of histograms : " + str(total_not_in_range))
        for i in range(m):
            for j in range(i):
                histos[j][i] = histos[i][j]
        return histos

    def compute_dna(self, d):
        print("Diagonalizing")
        eigs, vectors = linalg.eigsh(self.M, d, sparse.dia_matrix((self.S, [0]), shape=(len(self.p), len(self.p))),
                                     sigma=0, which='LM')  # Computes the d smallest eigenvalues
        eigs = eigs[1:]
        return eigs

def print_histo(histo):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate', 'violet', "khaki", "teal", "grey", "pink", "purple",
              "silver", "gold", "lime", "springgreen", "salmon", "slateblue", "crimson"]
    for j in range(min(len(histo), len(colors))):
        absc = [i*histo[j].step for i in range(int((histo[j].max -histo[j].min)/histo[j].step)+1)]
        ords = [histo[j].bins[i] for i in range(len(histo[j].bins))]
        plt.plot(absc, ords, color=colors[j])
    plt.show()


def compute_dnas(pathes, d):
    res = []
    for path in pathes:
        print("Beginning eigenvalues {0}".format(path))
        shape = Shape(path)
        res.append(shape.compute_dna(d))
    return res


def compute_dists_dna(eigs):
    n = len(eigs)
    d = len(eigs[0])
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(1, d):
            eigs[i][j] = eigs[i][j] / eigs[i][0]
    for i in range(n):
        for j in range(i):
            s = 0
            for k in range(d):
                s += pow(eigs[i][k] - eigs[j][k], 2)
            dists[i][j] += pow(s, 0.5)
            dists[j][i] = dists[i][j]
    return dists


def compute_histos(pathes):
    m = 7
    res = []
    max_ = 1.5
    step = max_ * m / 5000.0
    for path in pathes:
        print("Beginning histogram {0}".format(path))
        shape = Shape(path)
        res.append(shape.compute_histograms(25, m, 1000, step, max_))
    return res


def compute_dists_gps(histos):
    print("Calcul des distances")
    n = len(histos)
    m = len(histos[0])
    dists = [[0]*n for i in range(n)]
    for i in range(n):
        for j in range(i):
            for k in range(m):
                dists[i][j] += sum(Histogram.dist(histos[i][k][h], histos[j][k][h]) for h in range(m))
                dists[j][i] = dists[i][j]

    return dists

def plot_gps_mds(path):
    print("Begin plot of path : " + path)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate', 'violet', "khaki", "teal", "grey", "pink", "purple",
              "silver", "gold", "lime", "springgreen", "salmon", "slateblue", "crimson"]
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
    dists = compute_dists_gps(histos)
    groups = []
    counter = 0
    for i in range(len(dirs)):
        groups.append([])
        for j in range(len(files[i])):
            groups[-1].append(counter)
            counter += 1
    histo_points =[]
    for histo in histos:
        histo_coordinates = []
        for hist in histo:
            for h in hist:
                for b in h.bins:
                    histo_coordinates.append(b)
        histo_points.append(Point(histo_coordinates))
    error = compute_error(histo_points, groups)
    print("Error : " + str(error))
    embedding = MDS()
    arraydists = np.asarray(dists)
    resultat = embedding.fit_transform(arraydists)


    indices = [0]
    for i in range(len(dirs)):
        indices.append(indices[-1] + len(files[i]))
    legend = []
    for i in range(len(dirs)):
        absc = [resultat[i][0] for i in range(indices[i], indices[i+1])]
        ords = [resultat[i][1] for i in range(indices[i], indices[i+1])]
        plt.plot(absc, ords, color=colors[i], marker='o', linestyle='')
        legend.append(mpatches.Patch(color=colors[i], label=dirs[i]))
    plt.legend(handles=legend)


def plot_dna_mds(path):
    d = 200
    print("Begin plot of path : " + path)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate', 'violet', "khaki", "teal", "grey", "pink", "purple",
              "silver", "gold", "lime", "springgreen", "salmon", "slateblue", "crimson"]
    dirs = os.listdir(path)
    files = []
    for dir in dirs:
        assert os.path.isdir(path + '/' + dir)
        files.append(os.listdir(path + '/' + dir))
    pathes = []
    for i in range(len(dirs)):
        pathes += [path + '/' + dirs[i] + '/' + file for file in files[i]]
    print(pathes)

    eigvs = compute_dnas(pathes, d)
    dists = compute_dists_dna(eigvs)  # Changes eigvs !
    groups = []
    counter = 0
    for i in range(len(dirs)):
        groups.append([])
        for j in range(len(files[i])):
            groups[-1].append(counter)
            counter += 1
    eigv_points = []
    for eigv in eigvs:
        eigv_points.append(Point(eigv))
    error = compute_error(eigv_points, groups)
    print("Error : " + str(error))
    embedding = MDS()
    arraydists = np.asarray(dists)
    resultat = embedding.fit_transform(arraydists)

    indices = [0]
    for i in range(len(dirs)):
        indices.append(indices[-1] + len(files[i]))
    legend = []
    for i in range(len(dirs)):
        absc = [resultat[i][0] for i in range(indices[i], indices[i + 1])]
        ords = [resultat[i][1] for i in range(indices[i], indices[i + 1])]
        plt.plot(absc, ords, color=colors[i], marker='o', linestyle='')
        legend.append(mpatches.Patch(color=colors[i], label=dirs[i]))
    plt.legend(handles=legend)


def compute_compactness(distances, groups):
    compactness = 0.0
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            for k in range(j):
                print(distances[j][k])
                compactness += distances[j][k] ** 2
            compactness /= len(groups[i])
    return compactness


# labelled[i] = list of indices of points labelled as category i
def compute_error(points, labelled):
    res = 0.0

    # Defining centroids
    centroids = [Point([0.0]*points[0].dim)]*len(labelled)
    for i in range(len(labelled)):
        for j in range(len(labelled[i])):
            centroids[i] += points[labelled[i][j]]
        centroids[i] /= len(labelled[i])

    # Adding dists to centroids to error
    for i in range(len(labelled)):
        for j in range(len(labelled[i])):
            res += Point.dist(centroids[i], points[labelled[i][j]]) ** 2

    # Adding inverse of dists among centroids
    for i in range(len(centroids)):
        for j in range(i):
            res += 1/Point.dist(centroids[i], centroids[j])

    return res/(len(points))


def find_clusters_off(path):
    print("Begin cluster off files : " + path)
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
    histo_points = []
    for histo in histos:
        histo_coordinates = []
        for hist in histo:
            for h in hist:
                for b in h.bins:
                    histo_coordinates.append(b)
        histo_points.append(Point(histo_coordinates))
    max_coordinates = [max([p.coord[i] for p in histo_points]) for i in range(histo_points[0].dim)]
    compactness, centroids, labels, cluster_sizes = kmeans.kmeans(histo_points, len(dirs), 7, 0.00001, 10, max_coordinates)
    groups = []
    counter = 0
    for i in range(len(dirs)):
        groups.append([])
        for j in range(len(files[i])):
            groups[-1].append(counter)
            counter += 1
    print("Labels :")
    print(labels)
    print("VS groups :")
    print(groups)

def compare_histos(histo_list, names):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate', 'violet', "khaki", "teal", "grey", "pink", "purple",
              "silver", "gold", "lime", "springgreen", "salmon", "slateblue", "crimson"]
    for i in range(len(histo_list[0])):
        for j in range(len(histo_list[0][i])):
            ords_list = []
            legend = []
            for l, histo in enumerate(histo_list):
                absc = [k * histo[i][j].step for k in range(len(histo[i][j].bins))]
                ords_list.append([histo[i][j].bins[k] for k in range(len(histo[i][j].bins))])
                legend.append(mpatches.Patch(color=colors[l], label=names[l]))
            for l, ords in enumerate(ords_list):
                plt.plot(absc, ords, color=colors[l])
            plt.legend(handles=legend)
            plt.show()


def compare_lions(path):
    files = os.listdir(path)
    print(files)
    pathes = [path + '/' + file for file in files]
    histos = compute_histos(pathes)
    compare_histos(histos, [file[:-4] for file in files])


def full_GPS_test(path):
    print("Begin full GPS test : " + path)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate', 'violet', "khaki", "teal", "grey", "pink", "purple",
              "silver", "gold", "lime", "springgreen", "salmon", "slateblue", "crimson"]
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
    dists = compute_dists_gps(histos)
    groups = []
    counter = 0
    for i in range(len(dirs)):
        groups.append([])
        for j in range(len(files[i])):
            groups[-1].append(counter)
            counter += 1
    histo_points = []
    for histo in histos:
        histo_coordinates = []
        for hist in histo:
            for h in hist:
                for b in h.bins:
                    histo_coordinates.append(b)
        histo_points.append(Point(histo_coordinates))

    error = compute_error(histo_points, groups)
    print("Error : " + str(error))
    embedding = MDS()
    arraydists = np.asarray(dists)
    resultat = embedding.fit_transform(arraydists)
    indices = [0]
    for i in range(len(dirs)):
        indices.append(indices[-1] + len(files[i]))
    legend = []
    for i in range(len(dirs)):
        absc = [resultat[i][0] for i in range(indices[i], indices[i + 1])]
        ords = [resultat[i][1] for i in range(indices[i], indices[i + 1])]
        plt.plot(absc, ords, color=colors[i], marker='o', linestyle='')
        legend.append(mpatches.Patch(color=colors[i], label=dirs[i]))
    plt.legend(handles=legend)
    plt.show()

    max_coordinates = [max([p.coord[i] for p in histo_points]) for i in range(histo_points[0].dim)]
    compactness, centroids, labels, cluster_sizes = kmeans.kmeans(histo_points, len(dirs), 15, 0.00001, 20,
                                                                  max_coordinates)

    print("Labels :")
    print(labels)
    print("VS groups :")
    print(groups)























