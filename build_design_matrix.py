import argparse
import dateutil.parser
import gpxpy.gpx
import json
import numpy as np
import copy
from datetime import datetime
from datetime import timedelta

out_filename = "design_matrix.npy"
data_filename = "data.npy"
cluster_to_index = {}

def gpx_from_design_matrix(design_matrix, labels, centroids, f_name):
    # Output clusters
    gpx = gpxpy.gpx.GPX()

    for row in range(design_matrix.shape[0]):
        start_cluster = design_matrix[row, 0]
        start_centroid = centroids[start_cluster, :]

        end_cluster = labels[row]
        end_centroid = centroids[end_cluster, :]

        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)

        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(start_centroid[0], start_centroid[1], name=str(start_cluster), symbol=str(start_cluster)))
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(end_centroid[0], end_centroid[1], name=str(end_cluster), symbol=str(end_cluster)))

    with open(f_name, 'w') as out:
        out.write(gpx.to_xml())

def build_cluster_to_index_map(data):
    clusters = set()
    for point in data:
        clusters.add(point['cluster'])
    clusters = sorted(list(clusters))

    index = 0
    for point in data:
        if not point['cluster'] in cluster_to_index:
            cluster_to_index[point['cluster']] = index
            index += 1

def extract_features(curr_point, previous_point):
    """
    points are given as dicts with keys (arrival_time, departure_time, lat, lng, cluster)
    """
    current_cluster = [0] * len(cluster_to_index)
    previous_cluster = [0] * len(cluster_to_index)
    current_cluster[cluster_to_index[curr_point['cluster']]] = 1
    previous_cluster[cluster_to_index[previous_point['cluster']]] = 1

    datetime = dateutil.parser.parse(curr_point['departure_time'])
    day_of_week = [0] * 7
    day_of_week[datetime.weekday()] = 1

    hour_bin_size = 6
    hour_bin = [0] * (24 / hour_bin_size)
    hour_bin[datetime.time().hour / hour_bin_size] = 1

    is_weekend = 1 if (day_of_week == 5 or day_of_week == 6) else 0
    ispm = 1 if datetime.time().hour >= 12 else 0
    mwf = 1 if (day_of_week == 0 or day_of_week == 2 or day_of_week == 4) else 0

    features = day_of_week + hour_bin + [is_weekend, ispm]
    features += current_cluster
    #features += previous_cluster
    return features

def build_design_matrix(data):
    X = []
    Y = []
    for i in range(1, len(data) - 1):
        if data[i]['cluster'] == data[i+1]['cluster'] or data[i]['cluster'] == -1 or data[i+1]['cluster'] == -1:
            continue
        # Only instances when I am leaving home
        #if data[i]['cluster'] == 2:
        X.append(extract_features(data[i], data[i-1]))
        Y.append(data[i+1]['cluster'])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def generate_schedule(n_locs=5, n_days=20):
    home_1 = {
        'lat': 0.0,
        'lng': 0.0,
        'arrival_time': datetime(2014, 12, 8, 0, 0, 0),
        'departure_time': datetime(2014, 12, 8, 10, 0, 0),
        'cluster': 0
    }

    class_1 = {
        'lat': 0.0,
        'lng': 0.0,
        'arrival_time': datetime(2014, 12, 8, 10, 0, 0),
        'departure_time': datetime(2014, 12, 8, 11, 0, 0),
        'cluster': 1
    }

    class_2 = {
        'lat': 0.0,
        'lng': 0.0,
        'arrival_time': datetime(2014, 12, 8, 11, 0, 0),
        'departure_time': datetime(2014, 12, 8, 12, 0, 0),
        'cluster': 2
    }

    class_3 = {
        'lat': 0.0,
        'lng': 0.0,
        'arrival_time': datetime(2014, 12, 8, 13, 0, 0),
        'departure_time': datetime(2014, 12, 8, 15, 0, 0),
        'cluster': 3
    }

    home_2 = {
        'lat': 0.0,
        'lng': 0.0,
        'arrival_time': datetime(2014, 12, 8, 15, 0, 0),
        'departure_time': datetime(2014, 12, 8, 23, 59, 0),
        'cluster': 0
    }
    day1 = [home_1, class_1, class_2, class_3, home_2]
    day2 = copy.deepcopy(day1)
    day2[1]['cluster'] = 4
    day2[2]['cluster'] = 5
    day2[3]['cluster'] = 6
    for d in range(n_days):
        time_d = timedelta(days=d)
        new_day = []
        day_of_week = d % 7
        day = day1 if day_of_week in (1, 3, 5) else day2
        for c in day:
            new_c = copy.deepcopy(c)
            new_c['arrival_time'] = (new_c['arrival_time'] + time_d).isoformat()
            new_c['departure_time'] = (new_c['departure_time'] + time_d).isoformat()
            new_day.append(new_c)
        data.extend(new_day)
    return data




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate data files into a more usable format')
    parser.add_argument('-d', '--data', default="data.json")
    parser.add_argument('-x', '--design', default="design.npy")
    parser.add_argument('-y', '--labels', default="labels.npy")
    parser.add_argument('-c', '--centroids', default="centroids.npy")
    parser.add_argument('-g', '--gpx', default="design.gpx")
    parser.add_argument('-f', '--fake', action='store_true')
    args = parser.parse_args()

    print "Building design matrix..."
    data = []
    if args.fake:
        data = generate_schedule(5)
        print data
    else:
        with open(args.data, 'r') as data_file:
            data = json.load(data_file)

    build_cluster_to_index_map(data)
    design_matrix, labels = build_design_matrix(data)

    out_filename = args.design
    print "Design matrix shape: {0} x {1}".format(design_matrix.shape[0], design_matrix.shape[1])
    with open(out_filename, 'w') as outfile:
        np.save(outfile, design_matrix)
        print "Wrote design matrix to {0}".format(out_filename)

    out_filename = args.labels
    with open(out_filename, 'w') as outfile:
        np.save(outfile, labels)
        print "Wrote labels to {0}".format(out_filename)

    centroids = np.load(args.centroids)
    gpx_from_design_matrix(design_matrix, labels, centroids, args.gpx)
