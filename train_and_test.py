import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

MODELS = ["SVM", "LogisticRegression", "RandomForest", "LDA", "NaiveBayes"]

def get_model(model_name):
    if model_name == "SVM":
        print "Model: SVM"
        return svm.SVC(probability=True)
        #return OneVsRestClassifier(svm.LinearSVC(probability=True))
    elif model_name == "LogisticRegression":
        print "Model: Logistic Regression"
        return LogisticRegression()
    elif model_name == "RandomForest":
        print "Model: Random Forest"
        return RandomForestClassifier(50)
    elif model_name == "LDA":
        print "Model: LDA"
        return LDA()
    elif model_name == "NaiveBayes":
        print "Model: Naive Bayes"
        return BernoulliNB()


#def confusion_to_image(conf, output_file):
def show_confusion_matrix(conf):
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
    plt.show()
    #plt.savefig(output_file, format='png')


def precision_recall_from_confusion(confusion_matrix):
    true_count = np.sum(confusion_matrix, 1)
    predicted_count = np.sum(confusion_matrix, 0)
    correct = np.diagonal(confusion_matrix)
    true_count = true_count.astype(float)
    predicted_count = predicted_count.astype(float)
    correct = correct.astype(float)
    #print "True count", true_count
    #print "Predicted count", predicted_count
    #print "Correct", correct
    precision = np.divide(correct, predicted_count)
    recall = np.divide(correct, true_count)
    return precision, recall


def location_in_predicted(model, X, y):
    class_to_index = {c: i for i, c in enumerate(model.classes_)}
    print class_to_index
    probs = model.predict_proba(X)
    locations = []
    for row in range(X.shape[0]):
        if y[row] not in class_to_index:
            continue
        p = probs[row, :]
        sorted_indexes = np.argsort(p)[::-1]
        loc = np.where(sorted_indexes == class_to_index[y[row]])
        locations.append(loc[0][0])
    #print locations
    #print zip(locations, y.tolist())
    return locations

def test_online_learning(model, X, y, min_data):
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

        prediction = model.predict(X[i, :])[0]
        actual.append(y[i])
        predicted.append(prediction)
        errors.append(0 if prediction == y[i] else 1)
    evaluate_online(np.array(actual), np.array(predicted), np.array(errors), np.array(locations))
 

def evaluate_online(actual, predicted, errors, locations):
    confusion = confusion_matrix(actual, predicted)
    precision, recall = precision_recall_from_confusion(confusion)
    window_error = np.convolve(errors.astype(float), (np.ones(12)/12))
    window_location = np.convolve(locations.astype(float), (np.ones(10)/10))
    print "Online confusion"
    print confusion
    print "F1 score", f1_score(actual, predicted)
    print "Precision", precision
    print "Recall", recall
    print "Test accuracy", len(np.where(actual == predicted)[0].astype(float)) / float(len(actual))
    print "Mean position of correct class", np.mean(np.array(locations))
    show_confusion_matrix(confusion)
    #plt.plot(window_location)
    plt.plot(window_error)
    #plt.plot(locations)
    #plt.axis([0, 200, -1, 2])
    plt.show()


def graph_class_dist(y_train, y_test):
    classes = sorted(list(set(y_test) | set(y_train)))
    class_to_index = {c: i for i, c in enumerate(classes)}
    counts_test = np.zeros(len(classes))
    counts_train = np.zeros(len(classes))

    for c in y_test:
        counts_test[class_to_index[c]] += 1

    for c in y_train:
        counts_train[class_to_index[c]] += 1

    train_dist = counts_train / float(len(y_train))
    test_dist = counts_test / float(len(y_test))

    ind = np.arange(len(classes))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, train_dist, width, color='r')
    rects1 = ax.bar(ind+width, test_dist, width, color='b')
    ax.set_xticklabels(classes)
    plt.show()


def test_random_split(model, X, y, test_size=.3):
    for train_index, test_index in cross_validation.StratifiedShuffleSplit(y, 1, test_size=test_size, random_state=0):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

    graph_class_dist(y_train, y_test)

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    confusion = confusion_matrix(y_test, predicted)
    precision, recall = precision_recall_from_confusion(confusion)
    print "Train accuracy", model.score(X_train,y_train)
    print "Test accuracy", model.score(X_test, y_test)
    print "Precision", precision
    print "Recall", recall
    print "F1", f1_score(y_test, model.predict(X_test))
    print confusion
    show_confusion_matrix(confusion)
    locations = location_in_predicted(model, X_test, y_test)
    print "Mean position of correct class", np.mean(np.array(locations))


def test_k_fold(model, X, y, n_folds = 2):
    avg_score = 0
    y_test_predicted = np.array([])
    y_train_predicted = np.array([])
    y_test_true = np.array([])
    y_train_true = np.array([])
    kf = cross_validation.StratifiedKFold(y, n_folds=n_folds)
    locations = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        labels = set(y_train.tolist())
        sorted_labels = sorted(list(labels))
        avg_score += model.score(X_test, y_test)
        y_test_predicted = np.append(y_test_predicted, model.predict(X_test))
        y_train_predicted = np.append(y_train_predicted, model.predict(X_train))
        y_test_true = np.append(y_test_true, y_test)
        y_train_true = np.append(y_train_true, y_train)
        locations.extend(location_in_predicted(model, X_test, y_test))

    avg_score /= n_folds
    confusion = confusion_matrix(y_test_true, y_test_predicted)
    print "K-fold accuracy", avg_score
    print "Train accuracy", len(np.where(y_train_predicted == y_train_true)[0].astype(float)) / float(len(y_train_predicted))
    print "Test accuracy", len(np.where(y_test_predicted == y_test_true)[0].astype(float)) / float(len(y_test_predicted))
    print "F1", f1_score(y_test_true, y_test_predicted)
    print "Confusion"
    print  confusion
    print "Mean position of correct class", np.mean(np.array(locations))
    plt.hist(locations)
    plt.show()
    show_confusion_matrix(confusion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate data files into a more usable format')
    parser.add_argument('-x', '--design', default="design.npy")
    parser.add_argument('-y', '--labels', default="labels.npy")
    parser.add_argument('-c', '--classifier', default="svm")
    parser.add_argument('-m', '--model', choices=MODELS, default="SVM")
    parser.add_argument('-o', '--output', default="eval_images")
    args = parser.parse_args()

    with open(args.design, 'r') as design:
        X = np.load(design)

    with open(args.labels, 'r') as labels:
        y = np.load(labels)

    print "\n\n=====  TESTING  ====="
    model = get_model(args.model)
    print "\n== Testing Online Learning ==\n"
    test_online_learning(model, X, y, 20)
    print "\n\n== Testing K-Fold ==\n"
    test_k_fold(model, X, y, 6)
    print "\n\n== Testing random split ==\n"
    test_random_split(model, X, y, 0.3)
    
