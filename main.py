import numpy as np
from scipy import linalg

class Point:
    def __init__(self, coord):
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        self.coord = coord

    @staticmethod
    def dist(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    @staticmethod
    def cross(a, b):
        return Point([a.y * b.z - a.z * b.y, b.x * a.z - b.z * a.x, a.x * b.y - a.y * b.x])

    @staticmethod
    def scal(a, b):
        return sum([a.coord[i] * b.coord[i] for i in range(3)])

    def norm(self):
        return Point.dist(self, Point([0, 0, 0]))

    def opp(self):
        return Point([-self.coord[i] for i in range(3)])

    def unit(self):
        norm = Point.dist(self, Point([0, 0, 0]))
        return Point([self.coord[i] / norm for i in range(3)])

    def __add__(self, other):
        return Point([self.coord[i] + other.coord[i] for i in range(3)])

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Point([self.coord[i] - other.coord[i] for i in range(3)])

    def __rsub__(self, other):
        return (other - self).opp()

    def __truediv__(self, scal):
        assert type(scal) in [float, int]
        return Point([self.coord[i] / scal for i in range(3)])

    def __mul__(self, scal):
        assert type(scal) in [float, int]
        return Point([self.coord[i] * scal for i in range(3)])

    def __rmul__(self, other):
        return self * other

class Triangle:
    def __init__(self, points, normal=None, q=None):
        self.a = points[0]
        self.b = points[1]
        self.c = points[2]
        self.normal = normal
        self.points = points  # Indexes of the points
        self.q = q


class Shape:
    def __init__(self, path):
        self.p = []  # Points ~ 10⁴
        self.normals = []
        self.q = []
        self.triangles = []  # Each triangle is a size 3 list of points ~10⁴
        self.tglIdOfPt = []
        self.crclOfPt = []
        self.M = [[]]
        self.S = []
        self.read(path)
        self.find_q()
        self.build_triangle_circles()
        self.build_matrices()

    # Assumes vertices are all defined before triangles
    def read(self, path):
        with open(path, "r") as file:
            read_vertices = False
            for line in file:
                if line[0] == "v" and line[1] == " ":
                    assert not read_vertices
                    words = line.split("   ")
                    assert len(words) == 4
                    self.p.append(Point([float(words[1]), float(words[2]), float(words[3])]))

                elif line[1] == "n":
                    if not read_vertices:
                        read_vertices = True
                        self.tglIdOfPt = [[] for i in range(len(self.p))]
                    words = line.split("   ")
                    assert len(words) == 4
                    self.normals.append(Point([float(words[1]), float(words[2]), float(words[3])]))

                elif line[0] == "f":
                    words = line.split(" ")
                    assert len(words) == 4
                    words = words[1:]
                    words[2] = words[2][0:-1]
                    for i in range(len(words)):
                        words[i] = words[i].split("//")

                    self.triangles.append(Triangle([int(words[i][0]) - 1 for i in range(3)]))
                    for j in range(3):
                        self.tglIdOfPt[int(words[j][0]) - 1].append(len(self.triangles)-1)

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

    def build_matrices(self):
        self.M = [[0. for j in range(len(self.p))] for i in range(len(self.p))]
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
                assert left_hyp >= left_adj
                alpha = 1 / ((left_hyp / left_adj)**2 - 1) ** 0.5
                right_adj = abs(Point.scal(self.p[right.a] - self.p[right.c], (self.p[right.b] - self.p[right.c]).unit()))
                right_hyp = (self.p[right.c] - self.p[right.a]).norm()
                assert right_hyp >= right_adj
                beta = 1 / ((right_hyp / right_adj)**2 - 1) ** 0.5
                self.M[i][left.c] = (alpha + beta) / 2

                # S
                d = (self.p[left.b] - self.p[left.a]) / 2
                e = (self.p[left.c] - self.p[left.a]) / 2
                s +=(self.p[left.a] - d).norm() * (left.q - d).norm() / 2 \
                  + (self.p[left.a] - e).norm() * (left.q - e).norm() / 2
            self.S[i] = s
        for i in range(len(self.p)):
            self.M[i][i] = sum(self.M[i])