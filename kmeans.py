import main
import random

def kmeans(points, cluster_number, rand_rep, eps, max_rep, max_coordinates):
    print("Computing k_means")
    if not points:
        raise ValueError('No points')

    best_compactness = float('inf')
    best_labels = None
    best_centroids = None
    best_cluster_sizes = None

    for r in range(rand_rep):
        print("Random try number " + str(r))
        centroids = [main.Point([float(random.randint(0, max_coordinates[j])) for j in range(points[0].dim)]) for i in range(cluster_number)]
        labels = [0]*len(points)
        cluster_sizes = [0]*cluster_number
        assign_labels(points, labels, centroids)
        centroids, cluster_sizes = compute_centroids(points, labels, centroids, cluster_sizes)
        compactness = compute_compactness(points, labels, centroids)
        print("Initial labels")
        print(labels)
        rep = 1
        while compactness > eps and rep < max_rep:
            assign_labels(points, labels, centroids)
            centroids, cluster_sizes = compute_centroids(points, labels, centroids, cluster_sizes)
            compactness = compute_compactness(points, labels, centroids)
            rep +=1
        if compactness < best_compactness:
            best_compactness = compactness
            best_centroids = centroids
            best_cluster_sizes = cluster_sizes
            best_labels = labels
        print("Final labels :")
        print(labels)
        print("With compactness :")
        print(compactness)
    return best_compactness, best_centroids, best_labels, best_cluster_sizes


def assign_labels(points, labels, centroids):
    for i, p in enumerate(points):
        min_ = main.Point.dist(p, centroids[0])
        label = 0
        for j, c in enumerate(centroids):
            cur_dist = main.Point.dist(p,c)
            if min_ > cur_dist:
                min_ = cur_dist
                label = j
        labels[i] = label


def compute_compactness(points, labels, centroids):
    res = 0.0
    for i in range(len(points)):
        res += main.Point.dist(points[i], centroids[labels[i]])
    return res


def compute_centroids(points, labels, centroids, cluster_sizes):
    new_centroids = [main.Point([0.0]*centroids[0].dim)]*len(centroids)
    new_cluster_sizes = [0]*len(cluster_sizes)
    for i in range(len(points)):
        new_cluster_sizes[labels[i]] += 1
        new_centroids[labels[i]] += points[i]
    for i in range(len(new_centroids)):
        if new_cluster_sizes[i]:
            new_centroids[i] /= new_cluster_sizes[i]
    return new_centroids, new_cluster_sizes
