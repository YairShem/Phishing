import emlExtractFeatures
import pickle
import email
import emlBagOfWordsCsv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy


def predict(file_path):
    mlp = pickle.load(open("final/final-eml.sav", 'rb'))

    mail = open(file_path, "rb").read()
    msg = email.message_from_bytes(mail)
    body = emlExtractFeatures.extractBody(msg)
    body = emlBagOfWordsCsv.makeGood(body)

    # Replace email address with 'emailaddress'
    body = body.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')

    # Replace urls with 'webaddress'
    body = body.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

    # Replace money symbol with 'money-symbol'
    body = body.replace(r'£|\$', 'money-symbol')

    # Replace 10 digit phone number with 'phone-number'
    body = body.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phone-number')

    # Replace normal number with 'number'
    body = body.replace(r'\d+(\.\d+)?', 'number')

    # remove punctuation
    body = body.replace(r'[^\w\d\s]', ' ')

    # remove whitespace between terms with single space
    body = body.replace(r'\s+', ' ')

    # remove leading and trailing whitespace
    body = body.replace(r'^\s+|\s*?$', ' ')


    voc = pickle.load(open("final/final-tfidf-features.sav", 'rb'))

    tf_idf_vec_smooth = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, stop_words='english',
                                        vocabulary= voc)

    tf_new = tf_idf_vec_smooth.fit_transform([body]).toarray()


    features = emlExtractFeatures.extractFeatures(msg)

    for f in features.keys():
        if f == 'body':
            continue
        tf_new = numpy.append(tf_new, features[f])

    eml_features = [tf_new]
    prediction = mlp.predict(eml_features)
    return prediction

