import pandas
import urlFeatureExtaction


def createData():
    data = pandas.read_csv('data/url/urls.csv')
    ulrs = data[data.columns[0]].head(15000)
    labels = data[data.columns[1]].head(15000)
    return (ulrs, labels)


def makeFeaturesMatrix():
    urls, labels = createData()
    data = []
    for i in range(len(urls)):
        features = urlFeatureExtaction.extractFeatures(urls[i])
        features["label"] = labels[i]
        data.append(features)

    columns = []
    for featureName in features.keys():
        columns.append(featureName)
    df = pandas.DataFrame(data=data, columns=columns)
    featuresNames = columns[:-1]
    df.to_csv(r'data/urls_features.csv', index=False)
    return df, featuresNames


if __name__ == "__main__":
    makeFeaturesMatrix()
    print("Done!")
