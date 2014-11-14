import numpy as np
import gpxpy.gpx
from scipy.cluster.vq import kmeans2

def output_clusters_to_file(centroids, f_name):
    # Output clusters
    gpx = gpxpy.gpx.GPX()

    for i, centroid in enumerate(centroids):
        waypoint = gpxpy.gpx.GPXWaypoint(centroid[0], centroid[1], name=str(i), symbol=str(i))
        gpx.waypoints.append(waypoint)

    with open(f_name, 'w') as out:
        out.write(gpx.to_xml())

def cluster_location_data(latlng_data):
    """latlng_data: array of tuples with (lat, lng)"""
    matrix = np.matrix(latlng_data)
    print matrix.shape
    latlng = np.array(matrix)
    latlng = latlng.astype(float)


    centroids, labels = kmeans2(latlng, k=30, iter=1, minit='points')
    f_name = "k_means_iter/initial.gpx"
    output_clusters_to_file(centroids, f_name)

    for i in range(100):
        centroids, labels = kmeans2(latlng, k=centroids, iter=1, minit='matrix')
        f_name = "k_means_iter/{0}.gpx".format(i)
        output_clusters_to_file(centroids, f_name)

    return centroids, labels

