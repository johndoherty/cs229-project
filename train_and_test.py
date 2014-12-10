import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

MODELS = ["SVM", "LogisticRegression", "RandomForest", "LDA"]

def get_model(model_name):
    if model_name == "SVM":
        return svm.SVC(probability=True, class_weight='auto')
        #return OneVsRestClassifier(svm.LinearSVC(probability=True))
    elif model_name == "LogisticRegression":
        return OneVsRestClassifier(LogisticRegression())
    elif model_name == "RandomForest":
        return RandomForestClassifier(50)
    elif model_name == "LDA":
        return LDA()

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
    figure = plt.figure()
    plt.clf()
    ax = figure.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(normalize_conf), cmap=plt.cm.jet, interpolation='nearest')
    plt.savefig(output_file, format='png')


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

def precision_recall_from_confusion(confusion_matrix):
    true_count = np.sum(confusion_matrix, 1)
    predicted_count = np.sum(confusion_matrix, 0)
    correct = np.diagonal(confusion_matrix)
    true_count = true_count.astype(float)
    predicted_count = predicted_count.astype(float)
    correct = correct.astype(float)
    print "True count", true_count
    print "Predicted count", predicted_count
    print "Correct", correct
    precision = np.divide(correct, predicted_count)
    recall = np.divide(correct, true_count)
    return precision, recall

def evaluate_online(errors, actual, predicted, locations):
    f1 = f1_score(actual, predicted, average=None)
    confusion = confusion_matrix(actual, predicted)
    precision, recall = precision_recall_from_confusion(confusion)
    window_error = np.convolve(errors.astype(float), (np.ones(8)/8))
    window_location = np.convolve(locations.astype(float), (np.ones(5)/5))
    print "Online confusion", confusion
    print "F1 score", f1
    print "Precision", precision
    print "Recall", recall
    plt.plot(window_location)
    #plt.plot(locations)
    #plt.axis([0, 200, -1, 2])
    plt.show()


def location_in_predicted(model, X, y):
    class_to_index = {c: i for i, c in enumerate(model.classes_)}
    probs = model.predict_proba(X)
    locations = []
    for row in range(X.shape[0]):
        p = probs[row, :]
        sorted_indexes = np.argsort(p)[::-1]
        loc = np.where(sorted_indexes == class_to_index[y[row]])
        locations.append(loc[0][0])
    print "Mean position of correct class", np.mean(np.array(locations))
    print locations
    print zip(locations, y.tolist())
    plt.hist(locations)
    #plt.axis([0, 200, -1, 2])
    plt.show()
    return locations


def errors_by_features(X, y_predicted, y_actual):
    return


def online_learning(model, X, y, min_data):
    classes = set(y)
    class_to_index = {c: i for i, c in enumerate(classes)}
    errors = []
    predicted = []
    actual = []
    locations = []
    for i in range(min_data, X.shape[0]):
        model.fit(X[:i-1, :], y[:i-1])

        probs = model.predict_proba(X[i, :])[0]
        expanded_probs = [0] * len(classes)
        for j, p in enumerate(probs):
            c = model.classes_[j]
            expanded_probs[class_to_index[c]] = p
        sorted_indexes = np.argsort(expanded_probs)[::-1]
        loc = np.where(sorted_indexes == class_to_index[y[i]])
        locations.append(loc[0][0])

        prediction = model.predict(X[i, :])
        actual.append(y[i])
        predicted.append(prediction)
        errors.append(0 if prediction == y[i] else 1)
    return np.array(actual), np.array(predicted), np.array(errors), np.array(locations)
    

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

    model = get_model(args.model)
    actual, predicted, errors, locations = online_learning(model, X, y, 20)
    evaluate_online(errors, actual, predicted, locations)

    #print cross_validation.cross_val_score(model, X, y, scoring='recall')
    avg_score = 0
    n_folds = 2
    if (False):
        kf = cross_validation.StratifiedKFold(y, n_folds=n_folds)
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            labels = set(y_train.tolist())
            sorted_labels = sorted(list(labels))
            avg_score += test(model, X_test, y_test, sorted_labels)
        avg_score /= n_folds
        print avg_score
    else:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3, random_state=0)
        model.fit(X_train, y_train)
        print "Train accuracy", model.score(X,y)
        predicted = model.predict(X_test)
        labels = set(predicted.tolist())
        labels |= set(y_test.tolist())
        sorted_headings = sorted(list(labels))
        confusion = confusion_matrix(y_test, predicted)
        print sorted_headings
        print confusion
        precision, recall = precision_recall_from_confusion(confusion)
        print "Precision", precision
        print "Recall", recall
        #confusion_to_image(confusion, "confusion.png")
        sorted_labels = sorted(y_train.tolist())
        score = test(model, X_test, y_test, sorted_labels)
        location_in_predicted(model, X_test, y_test)
        print score
