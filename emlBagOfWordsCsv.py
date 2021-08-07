import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import glob
import os
import email
import emlExtractFeatures
import pickle


def makeGood(body):
    tmp_body = ' '.join(body.split())

    # Remove ,
    tmp_body = tmp_body.replace(',', '')

    # Remove "
    tmp_body = tmp_body.replace('"', '')

    # Remove =
    tmp_body = tmp_body.replace('=', '')

    # Remove -
    tmp_body = tmp_body.replace('-', '')

    # Remove _
    tmp_body = tmp_body.replace('_', '')

    # Replace email address with 'emailaddress'
    tmp_body = tmp_body.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')

    # Replace urls with 'webaddress'
    tmp_body = tmp_body.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

    # Replace money symbol with 'money-symbol'
    tmp_body = tmp_body.replace(r'£|\$', 'money-symbol')

    # Replace 10 digit phone number with 'phone-number'
    tmp_body = tmp_body.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phone-number')

    # Replace normal number with 'number'
    tmp_body = tmp_body.replace(r'\d+(\.\d+)?', 'number')

    # remove punctuation
    tmp_body = tmp_body.replace(r'[^\w\d\s]', ' ')

    # remove whitespace between terms with single space
    tmp_body = tmp_body.replace(r'\s+', ' ')

    # remove leading and trailing whitespace
    tmp_body = tmp_body.replace(r'^\s+|\s*?$', ' ')

    # change words to lower case
    tmp_body = tmp_body.lower()

    return tmp_body


def createData():
    data = []

    folder_path = 'data/phishing'
    for filename in glob.glob(os.path.join(folder_path, '*.eml')):
        with open(filename, "rb") as f:
            mail = f.read()
            msg = email.message_from_bytes(mail)
            features = emlExtractFeatures.extractFeatures(msg)
            # features = {}
            features["body"] = emlExtractFeatures.extractBody(msg)
            tmp = makeGood(features["body"])
            features["body"] = tmp
            features["label"] = 1
            data.append(features)

    folder_path = 'data/not_phishing'
    for filename in glob.glob(os.path.join(folder_path, '*.*')):
        with open(filename, "rb") as f:
            mail = f.read()
            msg = email.message_from_bytes(mail)
            features = emlExtractFeatures.extractFeatures(msg)
            # features = {}
            features["body"] = emlExtractFeatures.extractBody(msg)
            tmp = makeGood(features["body"])
            features["body"] = tmp
            features["label"] = 0
            data.append(features)

    columns = []
    for featureName in features.keys():
        columns.append(featureName)
    df = pandas.DataFrame(data=data, columns=columns)
    return df
    # df.to_csv(r'data/emlBagOfWords.csv', index=False)


def createCsv(df):
    # df = pandas.read_csv('data/emlBagOfWords.csv').dropna()

    # Replace email address with 'emailaddress'
    df['body'] = df['body'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')

    # Replace urls with 'webaddress'
    df['body'] = df['body'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

    # Replace money symbol with 'money-symbol'
    df['body'] = df['body'].str.replace(r'£|\$', 'money-symbol')

    # Replace 10 digit phone number with 'phone-number'
    df['body'] = df['body'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phone-number')

    # Replace normal number with 'number'
    df['body'] = df['body'].str.replace(r'\d+(\.\d+)?', 'number')

    # remove punctuation
    df['body'] = df['body'].str.replace(r'[^\w\d\s]', ' ')

    # remove whitespace between terms with single space
    df['body'] = df['body'].str.replace(r'\s+', ' ')

    # remove leading and trailing whitespace
    df['body'] = df['body'].str.replace(r'^\s+|\s*?$', ' ')

    # change words to lower case
    # df['body'] = df['body'].str.lower()

    X = df["body"]
    y = df["label"]

    tf_idf_vec_smooth = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, stop_words='english')
    tf_idf_data_smooth = tf_idf_vec_smooth.fit_transform(X)
    tf_idf_dataframe_smooth = pandas.DataFrame(tf_idf_data_smooth.toarray(),
                                               columns=tf_idf_vec_smooth.get_feature_names())
    selector = SelectKBest(chi2, k=2500)
    selector.fit(tf_idf_data_smooth, y)
    mask = selector.get_support()
    selector_dataframe = tf_idf_dataframe_smooth.iloc[:, mask]

    try:
        os.makedirs("final")
    except FileExistsError:
        pass	
	
    pickle.dump(selector_dataframe.columns, open('final/final-tfidf-features.sav', 'wb'))

    res = pandas.concat([selector_dataframe, df.drop(['body'], axis=1)], axis=1)
    res.to_csv(r'data/bag_of_words_2500_features.csv', index=False)


if __name__ == "__main__":
    df = createData()
    createCsv(df)
    print("Done!")
