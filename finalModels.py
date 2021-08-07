import pandas
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
import sklearn
import os


def main():
    try:
        os.makedirs("final")
    except FileExistsError:
        pass

    data = pandas.read_csv('data/urls_features.csv')
    df = sklearn.utils.shuffle(data)
    X = df.drop("label", axis=1).values
    y = df['label'].values
    et = ExtraTreesClassifier(n_estimators=51, min_samples_split=30)
    et.fit(X, y)
    pickle.dump(et, open('final/final-url.sav', 'wb'))

    data2 = pandas.read_csv('data/bag_of_words_2500_features.csv')
    df2 = sklearn.utils.shuffle(data2)
    X2 = df2.drop("label", axis=1).values
    y2 = df2['label'].values
    mlp = MLPClassifier(solver= 'adam', hidden_layer_sizes= 64, activation= 'tanh')
    mlp.fit(X2, y2)
    pickle.dump(mlp, open('final/final-eml.sav', 'wb'))


if __name__ == "__main__":
    main()
    print("Done!")
