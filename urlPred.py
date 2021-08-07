import urlFeatureExtaction
import pickle


def predict(address):

    et = pickle.load(open("final/final-url.sav", 'rb'))

    address_features = []
    features = urlFeatureExtaction.extractFeatures(address)
    for f in features.keys():
        address_features.append(features[f])

    address_features2 = [address_features]
    prediction = et.predict(address_features2)
    return prediction
