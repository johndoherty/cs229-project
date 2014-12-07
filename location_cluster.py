import numpy as np
import gpxpy.gpx
import os
import os.path
import shutil
from scipy.cluster.vq import kmeans2
from geopy.distance import great_circle

DISTANCE_THRESHOLD_METERS = 20

def build_maps(centroids, centroid_counts, labels, dir_name):
    # output centroids to gpx files
    gpx_centroids = gpxpy.gpx.GPX()
    for label, centroid in enumerate(centroids):
        name = "Cluster: {0} ({1} points)".format(label, centroid_counts[label])
        waypoint = gpxpy.gpx.GPXWaypoint(centroid[0], centroid[1], name=name, symbol=name)
        gpx_centroids.waypoints.append(waypoint)

    filename = os.path.join(dir_name, "centroids.gpx")
    with open(filename, 'w') as out:
        out.write(gpx_centroids.to_xml())


def build_locations(latlng_data):
    # build unique locations. two points within distance_trheshold_meters should be in considered the same location
    locations = []
    location_counts = []
    labels = [0] * len(latlng_data)
    # keeps track of which latlngs have already been added to a location
    used = [False,] * len(latlng_data)
    for i in range(len(latlng_data)):
        if used[i]:
            continue
        label = len(locations)
        labels[i] = label
        count = 1
        new_location = set([latlng_data[i]])
        for j in range(i+1, len(latlng_data)):
            if used[j]:
                continue
            if great_circle(latlng_data[i], latlng_data[j]).meters < DISTANCE_THRESHOLD_METERS:
                new_location.add(latlng_data[j])
                labels[j] = label
                count += 1
                used[j] = True
        locations.append(new_location)
        location_counts.append(count)
        used[i] = True

    centroids = []
    for location in locations:
        average_lat = sum([latlng[0] for latlng in location]) / len(location)
        average_lng = sum([latlng[1] for latlng in location]) / len(location)
        centroid = (average_lat, average_lng)
        centroids.append(centroid)
 
    return centroids, labels, location_counts

def cluster_location_data(latlng_data):
    """latlng_data: array of tuples with (lat, lng)"""
 
    gpx_files_dir = "maps"

    try:
        shutil.rmtree(gpx_files_dir)
    except:
        pass
    os.mkdir(gpx_files_dir)

    centroids, labels, centroid_counts = build_locations(latlng_data)
    build_maps(centroids, centroid_counts, labels, gpx_files_dir)

    centroids = np.array(centroids)
    labels = np.array(labels)

    return centroids, labels, centroid_counts

