import gpxpy
import re
import numpy as np
from os import listdir
from os.path import isfile, join

data_dir = "data"
out_filename = "data.npy"

data_files = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f)) and re.match("storyline.*.gpx", f)]

# Format lat, long, time
data = []

for data_file in data_files:
    print "Parsing: {0}".format(data_file)
    with open(data_file, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    for track in gpx.tracks:
        if len(track.segments) > 0 and len(track.segments[0].points) > 0:
            point = track.segments[0].points[0]
            data.append([point.latitude, point.longitude, point.time])
        #for segment in track.segments:
        #for point in segment.points:
        #    data.append([point.latitude, point.longitude, point.time])

matrix = np.matrix(data)

with open(out_filename, 'w') as outfile:
    np.save(outfile, matrix)
