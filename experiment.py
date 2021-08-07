import pandas
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import os


def createXLSX(scores, columns, expName, fileName):
    try:
        os.makedirs("expResult/" + expName)
    except FileExistsError:
        pass
    data = {}
    length = len(columns)
    for i in range(length):
        data[columns[i]] = scores[i]
    df = pandas.DataFrame(data)
    outputFileName = 'expResult/' + expName + '/' + fileName + '.xlsx'
    df.to_excel(outputFileName, index=False)


def classify(classifier, kf, X, y):
    split_num = 5
    score = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = score + metrics.accuracy_score(y_test, y_pred)
    score = score / split_num
    return score


def expKNN(expName, X, y, kf):
    x_graph = []
    y_graph = []
    z_graph = []
    for i in [3, 5, 7, 11, 21, 31, 41]:
        x_graph.append(i)
        knn = KNeighborsClassifier(n_neighbors=i)
        score = classify(knn, kf, X, y)
        y_graph.append(score)
        # print("i= ", i, "and score: ", score)

        knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
        score = classify(knn, kf, X, y)
        z_graph.append(score)
        # print("weights= distance,  i= ", i, "and score: ", score)

    columns = ['K', 'uniform', 'distance']
    createXLSX([x_graph, y_graph, z_graph], columns, expName, 'KNN')

    plt.plot(x_graph, y_graph, 'o-', c='blue')
    plt.plot(x_graph, z_graph, 'o-', c='red')
    plt.title("KNN")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.legend(["uniform", "distance"])
    plt.grid()
    plt.savefig('images/' + expName + '/KNN.png')
    plt.close()
    # plt.show()


def expSVM(expName, X, y, kf):
    x_graph = []
    y_graph = []
    for i in [0.25, 0.5, 1, 5]:
        x_graph.append(i)
        svm = LinearSVC(C=i, random_state=0, max_iter=2000)
        score = classify(svm, kf, X, y)
        y_graph.append(score)
        # print("C= ", i, "and score: ", score)

    columns = ['C', 'Accuracy']
    createXLSX([x_graph, y_graph], columns, expName, 'SVM')

    plt.plot(x_graph, y_graph, 'o-', c='blue')
    plt.title("SVM")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('images/' + expName + '/SVM.png')
    plt.close()
    # plt.show()


def expDT(expName, X, y, kf):
    for s in ['best', 'random']:
        x_graph = []
        y_graph = []
        # z_graph = []
        w_graph = []
        v_graph = []
        for i in [2, 5, 10, 30]:
            x_graph.append(i)
            dt = DecisionTreeClassifier(criterion="entropy", splitter=s, min_samples_split=i)
            score = classify(dt, kf, X, y)
            y_graph.append(score)
            # print("max_features= None,", "min_samples_split= ", i, "and score= ", score)  # max_features='None'

            dt = DecisionTreeClassifier(criterion="entropy", splitter=s, min_samples_split=i, max_features='sqrt')
            score = classify(dt, kf, X, y)
            w_graph.append(score)
            # print("max_features= sqrt,", "min_samples_split= ", i, "and score= ", score)

            dt = DecisionTreeClassifier(criterion="entropy", splitter=s, min_samples_split=i, max_features='log2')
            score = classify(dt, kf, X, y)
            v_graph.append(score)
            # print("max_features= log2,", "min_samples_split= ", i, "and score= ", score)

        columns = ['min_samples_split', 'None', 'sqrt', 'log2']
        createXLSX([x_graph, y_graph, w_graph, v_graph], columns, expName, 'DT - splitter=' + s)

        plt.plot(x_graph, y_graph, 'o-', c='blue')
        plt.plot(x_graph, w_graph, 'o-', c='yellow')
        plt.plot(x_graph, v_graph, 'o-', c='green')
        plt.legend(["None", "sqrt", "log2"])
        title = "DecisionTree \n splitter = " + s
        plt.title(title)
        plt.xlabel("min_samples_split")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.savefig('images/' + expName + '/DT - splitter=' + s + '.png')
        plt.close()
        # plt.show()


def expRF(expName, X, y, kf):
    x_graph = []
    y_graph = []
    z_graph = []
    w_graph = []
    v_graph = []
    for i in [1, 5, 11, 21, 31, 51, 91, 100]:
        x_graph.append(i)
        rf = RandomForestClassifier(criterion='entropy', n_estimators=i, min_samples_split=2)
        score = classify(rf, kf, X, y)
        y_graph.append(score)
        # print("min_samples_split= 2,", "n_estimators= ", i, "and score= ", score)

        rf = RandomForestClassifier(criterion='entropy', n_estimators=i, min_samples_split=5)
        score = classify(rf, kf, X, y)
        z_graph.append(score)
        # print("min_samples_split= 5,", "n_estimators= ", i, "and score= ", score)

        rf = RandomForestClassifier(criterion='entropy', n_estimators=i, min_samples_split=10)
        score = classify(rf, kf, X, y)
        w_graph.append(score)
        # print("min_samples_split= 10,", "n_estimators= ", i, "and score= ", score)

        rf = RandomForestClassifier(criterion='entropy', n_estimators=i, min_samples_split=30)
        score = classify(rf, kf, X, y)
        v_graph.append(score)
        # print("min_samples_split= 30,", "n_estimators= ", i, "and score= ", score)

    columns = ['n_estimators', 'min_samples_split=2', 'min_samples_split=5', 'min_samples_split=10', 'min_samples_split=30']
    createXLSX([x_graph, y_graph, z_graph, w_graph, v_graph], columns, expName, 'RF')

    plt.plot(x_graph, y_graph, 'o-', c='blue')
    plt.plot(x_graph, z_graph, 'o-', c='red')
    plt.plot(x_graph, w_graph, 'o-', c='yellow')
    plt.plot(x_graph, v_graph, 'o-', c='green')
    plt.legend(["min_samples_split=2", "min_samples_split=5", "min_samples_split=10", "min_samples_split=30"])

    plt.title("Random Forest")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('images/' + expName + '/RF.png')
    plt.close()
    # plt.show()


def expET(expName, X, y, kf):
    x_graph = []
    y_graph = []
    z_graph = []
    w_graph = []
    v_graph = []
    for i in [1, 5, 11, 21, 31, 51, 91, 100]:
        x_graph.append(i)
        rf = ExtraTreesClassifier(criterion='entropy', n_estimators=i, min_samples_split=2)
        score = classify(rf, kf, X, y)
        y_graph.append(score)
        # print("min_samples_split= 2,", "n_estimators= ", i, "and score= ", score)

        rf = ExtraTreesClassifier(criterion='entropy', n_estimators=i, min_samples_split=5)
        score = classify(rf, kf, X, y)
        z_graph.append(score)
        # print("min_samples_split= 5,", "n_estimators= ", i, "and score= ", score)

        rf = ExtraTreesClassifier(criterion='entropy', n_estimators=i, min_samples_split=10)
        score = classify(rf, kf, X, y)
        w_graph.append(score)
        # print("min_samples_split= 10,", "n_estimators= ", i, "and score= ", score)

        rf = ExtraTreesClassifier(criterion='entropy', n_estimators=i, min_samples_split=30)
        score = classify(rf, kf, X, y)
        v_graph.append(score)
        # print("min_samples_split= 30,", "n_estimators= ", i, "and score= ", score)

    columns = ['n_estimators', 'min_samples_split=2', 'min_samples_split=5', 'min_samples_split=10', 'min_samples_split=30']
    createXLSX([x_graph, y_graph, z_graph, w_graph, v_graph], columns, expName, 'ET')

    plt.plot(x_graph, y_graph, 'o-', c='blue')
    plt.plot(x_graph, z_graph, 'o-', c='red')
    plt.plot(x_graph, w_graph, 'o-', c='yellow')
    plt.plot(x_graph, v_graph, 'o-', c='green')
    plt.legend(["min_samples_split=2", "min_samples_split=5", "min_samples_split=10", "min_samples_split=30"])

    plt.title("Extra Trees")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('images/' + expName + '/ET.png')
    plt.close()
    # plt.show()


def expGB(expName, X, y, kf):
    x_graph = []
    y_graph = []
    for i in [5, 11, 15, 19, 27, 41, 51, 81]:
        x_graph.append(i)
        gb = GradientBoostingClassifier(n_estimators=i)
        score = classify(gb, kf, X, y)
        y_graph.append(score)
        # print("n_estimators= ", i, "and score= ", score)

    columns = ['n_estimators', 'Accuracy']
    createXLSX([x_graph, y_graph], columns, expName, 'GB')

    plt.plot(x_graph, y_graph, 'o-', c='blue')
    plt.title("Gradient Boosting")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('images/' + expName + '/GB.png')
    plt.close()
    # plt.show()


def expSGD(expName, X, y, kf):
    for l in ['squared_hinge', 'perceptron']:
        x_graph = []
        y_graph = []
        z_graph = []
        w_graph = []
        for i in [0.0001, 0.001, 0.01, 0.1, 1]:
            x_graph.append(i)
            sgd = SGDClassifier(loss=l, eta0=i, penalty='l2')
            score = classify(sgd, kf, X, y)
            y_graph.append(score)
            # print("eta0= ", i, "penalty=l2 and score= ", score)

            sgd = SGDClassifier(loss=l, eta0=i, penalty='l1')
            score = classify(sgd, kf, X, y)
            z_graph.append(score)
            # print("eta0= ", i, "penalty=l1 and score= ", score)

            sgd = SGDClassifier(loss=l, eta0=i, penalty='elasticnet')
            score = classify(sgd, kf, X, y)
            w_graph.append(score)
            # print("eta0= ", i, "penalty=elasticnet and score= ", score)

        columns = ['eta0', 'l2', 'l1', 'elasticnet']
        createXLSX([x_graph, y_graph, z_graph, w_graph], columns, expName, 'SGD - loss=' + l)

        plt.plot(x_graph, y_graph, 'o-', c='blue')
        plt.plot(x_graph, z_graph, 'o-', c='red')
        plt.plot(x_graph, w_graph, 'o-', c='yellow')
        plt.xlabel("eta0")
        plt.ylabel("Accuracy")
        plt.legend(["l2", "l1", "elasticnet"])
        title = "SGD \n loss = " + l
        plt.title(title)
        plt.grid()
        plt.savefig('images/' + expName + '/SGD - loss=' + l + '.png')
        plt.close()
        # plt.show()


def expMLP(expName, X, y, kf):
    for s in ['adam', 'lbfgs']:
        x_graph = []
        y_graph = []
        z_graph = []
        w_graph = []
        v_graph = []
        for i in [32, 64, 128, 256]:
            x_graph.append(i)
            mlp = MLPClassifier(hidden_layer_sizes=i, activation='identity', solver=s)
            score = classify(mlp, kf, X, y)
            y_graph.append(score)
            # print("hidden_layer_sizes= ", i, "activation= identity, solver= ", s, "score = ", score)

            mlp = MLPClassifier(hidden_layer_sizes=i, activation='logistic', solver=s)
            score = classify(mlp, kf, X, y)
            z_graph.append(score)
            # print("hidden_layer_sizes= ", i, "activation= logistic, solver= ", s, "score = ", score)

            mlp = MLPClassifier(hidden_layer_sizes=i, activation='tanh', solver=s)
            score = classify(mlp, kf, X, y)
            w_graph.append(score)
            # print("hidden_layer_sizes= ", i, "activation= tanh, solver= ", s, "score = ", score)

            mlp = MLPClassifier(hidden_layer_sizes=i, activation='relu', solver=s)
            score = classify(mlp, kf, X, y)
            v_graph.append(score)
            # print("hidden_layer_sizes= ", i, "activation= relu, solver= ", s, "score = ", score)

        columns = ['hidden_layer_sizes', 'identity', 'logistic', 'tanh', 'relu']
        createXLSX([x_graph, y_graph, z_graph, w_graph, v_graph], columns, expName, 'MLP - solver = ' + s)

        plt.plot(x_graph, y_graph, 'o-', c='blue')
        plt.plot(x_graph, z_graph, 'o-', c='red')
        plt.plot(x_graph, w_graph, 'o-', c='yellow')
        plt.plot(x_graph, v_graph, 'o-', c='green')
        plt.xlabel("hidden layers")
        plt.ylabel("Accuracy")
        plt.legend(["identity", "logistic", "tanh", "relu"])
        title = "MLP \n solver = " + s
        plt.title(title)
        plt.grid()
        plt.savefig('images/' + expName + '/MLP - solver=' + s + '.png')
        plt.close()
        # plt.show()


def expAB(expName, X, y, kf):
    x_graph = []
    y_graph = []
    for i in [10, 50, 100, 200, 250]:
        x_graph.append(i)
        ab = AdaBoostClassifier(n_estimators=i)
        score = classify(ab, kf, X, y)
        y_graph.append(score)
        # print("n_estimators= ", i, "and score= ", score)

    columns = ['n_estimators', 'Accuracy']
    createXLSX([x_graph, y_graph], columns, expName, 'AB')

    plt.plot(x_graph, y_graph, 'o-', c='blue')
    plt.title("Ada Boost")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('images/' + expName + '/AB.png')
    plt.close()
    # plt.show()


def expNB(expName, X, y, kf):
    x_graph = []
    y_graph = []
    for i in [0.0001, 0.001, 0.01, 0.1, 1]:
        x_graph.append(i)
        nb = MultinomialNB(alpha=i)
        score = classify(nb, kf, X, y)
        y_graph.append(score)
        # print("alpha= ", i, "and score= ", score)

    columns = ['alpha', 'Accuracy']
    createXLSX([x_graph, y_graph], columns, expName, 'NB')

    plt.plot(x_graph, y_graph, 'o-', c='blue')
    plt.title("Naive Bayes")
    plt.xlabel("alpha")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('images/' + expName + '/NB.png')
    plt.close()
    # plt.show()


def runExp(path):
    try:
        os.makedirs("expResult")
    except FileExistsError:
        pass
    try:
        os.makedirs("images")
    except FileExistsError:
        pass
    try:
        os.makedirs("images/" + path[0])
    except FileExistsError:
        pass

    split_num = 5
    data = pandas.read_csv(path[1]).dropna()
    df = sklearn.utils.shuffle(data)
    X = df.drop("label", axis=1).values
    y = df['label'].values
    kf = KFold(n_splits=split_num)

    expKNN(path[0], X, y, kf)
    expSVM(path[0], X, y, kf)
    expDT(path[0], X, y, kf)
    expRF(path[0], X, y, kf)
    expET(path[0], X, y, kf)
    expGB(path[0], X, y, kf)
    expSGD(path[0], X, y, kf)
    expMLP(path[0], X, y, kf)
    expAB(path[0], X, y, kf)
    expNB(path[0], X, y, kf)


if __name__ == "__main__":
    paths = [('URL - Manual Features', 'data/urls_features.csv'),
             ('Email - Bag Of Words', 'data/bag_of_words_2500_features.csv')]
    for path in paths:
        runExp(path)

    print("Done!")
