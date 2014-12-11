import argparse
import gpxpy
import re
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from location_cluster import cluster_location_data


def parse_data_from_dir(data_dir):
    data_files = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and re.match("storyline.*.gpx", f)]

    # Format lat, long, start_time, end_time
    data = []
    latlng_data = []
    arrival_time_string = ""

    print "Parsing data..."
    for data_file in data_files:
        with open(data_file, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
        for track in gpx.tracks:
            if len(track.segments) > 0 and len(track.segments[0].points) > 0:
                departure_point = track.segments[0].points[0]
                location = {
                    'lat': departure_point.latitude,
                    'lng': departure_point.longitude,
                    'arrival_time': arrival_time_string,
                    'departure_time': departure_point.time.isoformat(),
                }
                data.append(location)
                latlng_data.append((departure_point.latitude,departure_point.longitude))
                arrival_time_string = track.segments[-1].points[-1].time.isoformat()

    # Cluster lat, lng and add that cluster info to the data dict
    print "Clustering..."
    centroids, labels, counts = cluster_location_data(latlng_data)
    for i in range(len(data)):
        if counts[labels[i]] > 4:
            data[i]['cluster'] = labels[i]
        else:
            data[i]['cluster'] = -1

    return data, centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate data files into a more usable format')
    parser.add_argument('-d', '--data', default="data")
    parser.add_argument('-o', '--output', default="data.json")
    parser.add_argument('-c', '--centroids', default="centroids.npy")
    args = parser.parse_args()

    location_dicts,centroids = parse_data_from_dir(args.data)

    out_filename = args.output
    with open(out_filename, 'w') as outfile:
        json.dump(location_dicts, outfile)
        print "Wrote data to {0}".format(out_filename)

    out_filename = args.centroids
    with open(out_filename, 'w') as outfile:
        np.save(outfile, centroids)
        print "Wrote centroids to {0}".format(out_filename)
