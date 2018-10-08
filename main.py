import numpy as np


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.coord = [x,y,z]

class Triangle:
    def __init__(self, points):
        self.a = points[0]
        self.b = points[1]
        self.c = points[2]
        self.points = points  # Indexes of the points


class Shape:
    def __init__(self):
        self.p = []  # Points ~ 10⁴
        self.triangles = []  # Each triangle is a size 3 list of points ~10⁴
        self.trianglesOfPoint = []

    # Assumes vertices are all defined before triangles
    def read(self, path):
        with open(path, "r") as file:
            readVertices = False
            for line in file :
                if line[0] == "v" and line[1] == " ":
                    assert not readVertices
                    words = line.split("   ")
                    assert len(words) == 4
                    self.p.append(Point(float(words[1]), float(words[2]), float(words[3])))
                elif line[0] == "f":
                    if not readVertices:
                        readVertices = True
                        self.trianglesOfPoint = [[] for i in range(len(self.p))]
                    words = line.split(" ")
                    assert len(words) == 4
                    words = words[1:]
                    for word in words:
                        word = word.split("//")
                    self.triangles.append(Triangle([int(words[i][0])-1 for i in range(3)]))
                    for i in range(3):
                            self.trianglesOfPoint[int(words[i][0])-1].append(len(self.triangles) - 1)


    def findAngles(self):
        for i in range(len(self.p)):
            for triangle in self.trianglesOfPoint[i]:
                if i==triangle.a:
                    



s = Shape()
s.read("../lion-poses/lion-reference.obj")
print(len(s.p))
print(s.triangles)