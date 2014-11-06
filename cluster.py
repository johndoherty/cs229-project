import numpy as np
import gpxpy.gpx
from scipy.cluster.vq import kmeans2

data_file = "data.npy"
out_file = "centroids.gpx"


def output_clusters_to_file(centroids, f_name):
    # Output clusters
    gpx = gpxpy.gpx.GPX()

    for i, centroid in enumerate(centroids):
        waypoint = gpxpy.gpx.GPXWaypoint(centroid[0], centroid[1], name=str(i), symbol=str(i))
        gpx.waypoints.append(waypoint)

    with open(f_name, 'w') as out:
        out.write(gpx.to_xml())

matrix = np.load(data_file)
print matrix.shape
latlng = np.array(matrix[:, 0:2])
latlng = latlng.astype(float)



centroids, labels = kmeans2(latlng, k=30, iter=1, minit='points')
f_name = "k_means_iter/initial.gpx"
output_clusters_to_file(centroids, f_name)

for i in range(100):
    centroids, labels = kmeans2(latlng, k=centroids, iter=1, minit='matrix')
    f_name = "k_means_iter/{0}.gpx".format(i)
    output_clusters_to_file(centroids, f_name)


centroid_scores = []
indexes_to_remove = []
for i, centroid in enumerate(centroids):
    indexes = np.where(labels == i)[0]
    if len(indexes) == 0:
        #indexes_to_remove += [index for index in indexes]
        continue
    score = 0
    for index in indexes:
        dist = np.linalg.norm(latlng[index, :] - centroid)
        score += dist

    score /= len(indexes)
    centroid_scores.append((i, score, len(indexes)))

sorted_scores = sorted(centroid_scores, key=lambda a: a[2])
print centroid_scores

