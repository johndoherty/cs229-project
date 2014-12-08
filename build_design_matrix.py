import argparse
import dateutil.parser
import gpxpy.gpx
import json
import numpy as np

out_filename = "design_matrix.npy"
data_filename = "data.npy"

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

def extract_features(curr_point, previous_point):
    """
    points are given as dicts with keys (arrival_time, departure_time, lat, lng, cluster)
    """
    current_cluster = curr_point['cluster']
    datetime = dateutil.parser.parse(curr_point['departure_time'])
    day_of_week = datetime.weekday()
    is_weekend = (day_of_week == 5 or day_of_week == 6)
    hour_3 = datetime.time().hour / 3
    ispm = datetime.time().hour >= 12
    mwf = (day_of_week == 0 or day_of_week == 2 or day_of_week == 4)

    previous_cluster = previous_point['cluster']

    return [current_cluster, previous_cluster, day_of_week, is_weekend, hour_3, ispm]
    #return [current_cluster, day_of_week, is_weekend, hour_3, ispm]
    #return [current_cluster, day_of_week, hour_3]

def build_design_matrix(data):
    X = []
    Y = []
    for i in range(1, len(data) - 1):
        if data[i]['cluster'] == data[i+1]['cluster'] or data[i]['cluster'] == -1 or data[i+1]['cluster'] == -1:
            continue
        X.append(extract_features(data[i], data[i-1]))
        Y.append(data[i+1]['cluster'])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate data files into a more usable format')
    parser.add_argument('-d', '--data', default="data.json")
    parser.add_argument('-x', '--design', default="design.npy")
    parser.add_argument('-y', '--labels', default="labels.npy")
    parser.add_argument('-c', '--centroids', default="centroids.npy")
    parser.add_argument('-g', '--gpx', default="design.gpx")
    args = parser.parse_args()

    print "Building design matrix..."
    data = []
    with open(args.data, 'r') as data_file:
        data = json.load(data_file)

    design_matrix, labels = build_design_matrix(data)

    out_filename = args.design
    with open(out_filename, 'w') as outfile:
        np.save(outfile, design_matrix)
        print "Wrote design matrix to {0}".format(out_filename)

    out_filename = args.labels
    with open(out_filename, 'w') as outfile:
        np.save(outfile, labels)
        print "Wrote labels to {0}".format(out_filename)

    centroids = np.load(args.centroids)
    gpx_from_design_matrix(design_matrix, labels, centroids, args.gpx)
