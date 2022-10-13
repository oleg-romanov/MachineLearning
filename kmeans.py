import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def random_points(n):
    points = []
    for i in range(n):
        point = {'x': random.randint(0, 100), 'y': random.randint(0, 100)}
        points.append(point)
    return points

def dist_calc(a, b):
    return np.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2)

def centroids(points, k):
    x_sum, y_sum = 0, 0
    for i in range(len(points)):
        x_sum += points[i]['x']
        y_sum += points[i]['y']

    x_sum /= len(points)
    y_sum /= len(points)

    center_point = {'x': x_sum, 'y': y_sum}
    r = 0

    for i in range(len(points)):
        r = max(r, dist_calc(center_point, points[i]))

    centroids = []

    for i in range(k):
        centroid = {'x': x_sum + r * np.cos(2 * np.pi * i / k), 'y': y_sum + r * np.sin(2 * np.pi * i / k)}
        centroids.append(centroid)
    return centroids


def kmeans_with_draw(points, centroids):
    diff = 1
    cluster = np.zeros(len(points))
    while diff:
        for i, point in enumerate(points):
            mn_dist = float('inf')
            for j, centroid in enumerate(centroids):
                d = dist_calc(centroid, point)
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = j
        new_centroids = pd.DataFrame(points).groupby(by=cluster).mean().to_dict('records')

        for i, p in enumerate(points):
            plt.scatter(p['x'], p['y'], color=colors[int(cluster[i])])
        plt.scatter([p['x'] for p in new_centroids], [p['y'] for p in new_centroids], color='r')
        plt.show()

        if centroids == new_centroids:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster


def kmeans(points, centroids):
    diff = 1
    cluster = np.zeros(len(points))
    while diff:
        for i, point in enumerate(points):
            mn_dist = float('inf')
            for j, centroid in enumerate(centroids):
                d = dist_calc(centroid, point)
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = j
        new_centroids = pd.DataFrame(points).groupby(by=cluster).mean().to_dict('records')

        if centroids == new_centroids:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster


def cost_calc(points, centroids, cluster):
    sum = 0
    for i, point in enumerate(points):
        sum += dist_calc(point, centroids[int(cluster[i])])
    return sum


if __name__ == "__main__":
    n = 200
    points = random_points(n)

    cost_list = []
    for k in range(1, 10):
        centroids_list = centroids(points, k)
        centroids_list, cluster = kmeans(points, centroids_list)
        cost = cost_calc(points, centroids_list, cluster)
        cost_list.append(cost)

    k_min = 1
    min_val = float('inf')

    for i in range(1, 4):
        val = abs(cost_list[i + 1] - cost_list[i]) / abs(cost_list[i - 1] - cost_list[i])
        if val < min_val:
            k_min = i
            min_val = val

    centroids_list = centroids(points, k_min)
    colors = ["#008941",
              "#006FA6",
              "#A30000",
              "#7604c7",
              "#f5680a",
              "#001ca6",
              "#1ed6b7",
              "#eb0cd1"]
    centroids_list, cluster = kmeans_with_draw(points, centroids_list)

    for i, p in enumerate(points):
        plt.scatter(p['x'], p['y'], color=colors[int(cluster[i])])
    plt.scatter([p['x'] for p in centroids_list], [p['y'] for p in centroids_list], color='r')
    plt.show()