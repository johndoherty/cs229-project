import argparse
import json
import numpy as np

out_filename = "design_matrix.npy"
data_filename = "data.npy"

def extract_features(from_point, to_point):
    """
    points are given as dicts with keys (arrival_time, departure_time, lat, lng, cluster)
    """
    return [from_point['cluster']]

def build_design_matrix(data):
    X = []
    Y = []
    for i in range(len(data) - 1):
        X.append(extract_features(data[i], data[i+1]))
        Y.append(data[i+1]['cluster'])

    X = np.array(X)
    Y = np.array(Y)
    print X
    print Y


    return X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate data files into a more usable format')
    parser.add_argument('-d', '--data', default="data.json")
    parser.add_argument('-x', '--design', default="design.npy")
    parser.add_argument('-y', '--labels', default="leabels.npy")
    args = parser.parse_args()

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
        np.save(outfile, design_matrix)
        print "Wrote labels to {0}".format(out_filename)

