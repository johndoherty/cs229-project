import argparse
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot

MODELS = ["SVM", "LogisticRegression", "RandomForest"]

def train(X, y, model):
    if model == "SVM":
        clf = svm.SVC(probability=True)
        #clf = svm.SVC()
        clf.fit(X,y)
    elif model == "LogisticRegression":
        clf = OneVsRestClassifier(LogisticRegression()).fit(X,y)
    elif model == "RandomForest":
        clf = RandomForestClassifier(50).fit(X,y)
    return clf

def predict(clf, x, classes, use_prob = False):
    if use_prob:
        prob = clf.predict_proba(x)
        prediction = 0
        max_prob = 0
        for col in range(prob.shape[1]):
            #if classes[col] != x[0] and prob[0, col] > max_prob:
            if prob[0, col] > max_prob:
                max_prob = prob[0, col]
                prediction = classes[col]
        return prediction
    else:
        return clf.predict(x)


def confusion_to_image(conf, output_file):
    normalize_conf = conf.astype(float)
    for row in range(conf.shape[0]):
        count = np.sum(conf[row, :])
        if count > 0:
            normalize_conf[row, :] = conf[row, :] / float(count)
    figure = pyplot.figure()
    pyplot.clf()
    ax = figure.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(normalize_conf), cmap=pyplot.cm.jet, interpolation='nearest')
    pyplot.savefig(output_file, format='png')


def test(clf, X, Y, labels):
    correct = 0
    score = clf.score(X, Y)
    for row in range(X.shape[0]):
        prediction = predict(clf, X[row, :], labels, False)
        if prediction == Y[row]:
            correct += 1
            #print "{0} -> {1}     It's a match!!!".format(X[row, :], Y[row])
        #else:
            #print "{0} -> {1}     Wrong: correct match = {2}".format(X[row, :], prediction, Y[row])

    score = float(correct) / len(Y)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate data files into a more usable format')
    parser.add_argument('-x', '--design', default="design.npy")
    parser.add_argument('-y', '--labels', default="labels.npy")
    parser.add_argument('-c', '--classifier', default="svm")
    parser.add_argument('-m', '--model', choices=MODELS, default="SVM")
    args = parser.parse_args()

    with open(args.design, 'r') as design:
        X = np.load(design)

    with open(args.labels, 'r') as labels:
        y = np.load(labels)

    #print cross_validation.cross_val_score(model, X, y, scoring='recall')

    avg_score = 0
    n_folds = 2
    if (True):
        kf = cross_validation.StratifiedKFold(y, n_folds=n_folds)
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = train(X_train, y_train, args.model)
            labels = set(y_train.tolist())
            sorted_labels = sorted(list(labels))
            avg_score += test(clf, X_test, y_test, sorted_labels)
        avg_score /= n_folds
        print avg_score
    else:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3, random_state=0)
        clf = train(X_train, y_train, args.model)
        predicted = clf.predict(X_test)
        confusion = confusion_matrix(y_test, predicted)
        print confusion
        confusion_to_image(confusion, "confusion.png")
        sorted_labels = sorted(y_train.tolist())
        score = test(clf, X_test, y_test, sorted_labels)
        print score
