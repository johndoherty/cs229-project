import argparse
import numpy as np
from sklearn import cross_validation
from sklearn import svm

def train(X, Y):
    clf = svm.SVC()
    clf.fit(X,Y)
    return clf

def test(clf, X, Y):
    print clf.score(X, Y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate data files into a more usable format')
    parser.add_argument('-x', '--design', default="design.npy")
    parser.add_argument('-y', '--labels', default="labels.npy")
    parser.add_argument('-m', '--model')
    parser.add_argument('-c', '--classifier', default="svm")
    args = parser.parse_args()

    with open(args.design, 'r') as design:
        X = np.load(design)

    with open(args.labels, 'r') as labels:
        y = np.load(labels)

    print X
    import ipdb; ipdb.set_trace()
    for row in range(X.shape[0]):
        print "{0}  ->  {1}".format(X[row, :], y[row])

    model = svm.SVC()
    #print cross_validation.cross_val_score(model, X, y, scoring='recall')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3, random_state=0)
    clf = train(X_train, y_train)
    test(clf, X_test, y_test)

