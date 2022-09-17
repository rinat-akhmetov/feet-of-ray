import re

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('artifacts/train.csv')
test_df = pd.read_csv('artifacts/test.csv')
cols_target = ['obscene', 'insult', 'toxic', 'severe_toxic', 'identity_hate', 'threat']


def preprocess_data(train_df, test_df):
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')
        return text

    train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))
    test_df['char_length'] = test_df['comment_text'].apply(lambda x: len(str(x)))

    # clean the comment_text in train_df [Thanks to Pulkit Jha for the useful pointer.]
    train_df['comment_text'] = train_df['comment_text'].map(lambda com: clean_text(com))
    test_df['comment_text'] = test_df['comment_text'].map(lambda com: clean_text(com))
    train_df = train_df.drop('char_length', axis=1)

    X = train_df.comment_text
    test_X = test_df.comment_text
    return X, test_X


def train(X, test_X):
    # create a function to add features
    def add_feature(X, feature_to_add):
        '''
        Returns sparse feature matrix with added feature.
        feature_to_add can also be a list of features.
        '''
        return hstack([X, csr_matrix(feature_to_add).T], 'csr')

    vect = TfidfVectorizer(max_features=5000, stop_words='english')
    X_dtm = vect.fit_transform(X)
    test_X_dtm = vect.transform(test_X)
    logreg = LogisticRegression(C=12.0)
    submission_binary = pd.read_csv('artifacts/sample_submission.csv')

    for label in cols_target:
        print('... Processing {}'.format(label))
        y = train_df[label]
        # train the model using X_dtm & y
        logreg.fit(X_dtm, y)
        # compute the training accuracy
        y_pred_X = logreg.predict(X_dtm)
        print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
        # compute the predicted probabilities for X_test_dtm
        test_y_prob = logreg.predict_proba(test_X_dtm)[:, 1]
        submission_binary[label] = test_y_prob
        # generate submission file
        submission_binary.to_csv('artifacts/submission_binary.csv', index=False)

        # create submission file
        submission_chains = pd.read_csv('artifacts/sample_submission.csv')

    for label in cols_target:
        print('... Processing {}'.format(label))
        y = train_df[label]
        # train the model using X_dtm & y
        logreg.fit(X_dtm, y)
        # compute the training accuracy
        y_pred_X = logreg.predict(X_dtm)
        print('Training Accuracy is {}'.format(accuracy_score(y, y_pred_X)))
        # make predictions from test_X
        test_y = logreg.predict(test_X_dtm)
        test_y_prob = logreg.predict_proba(test_X_dtm)[:, 1]
        submission_chains[label] = test_y_prob
        # chain current label to X_dtm
        X_dtm = add_feature(X_dtm, y)
        print('Shape of X_dtm is now {}'.format(X_dtm.shape))
        # chain current label predictions to test_X_dtm
        test_X_dtm = add_feature(test_X_dtm, test_y)
        print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))

    # generate submission file
    submission_chains.to_csv('artifacts/submission_chains.csv', index=False)

    # create submission file
    submission_combined = pd.read_csv('artifacts/sample_submission.csv')

    # corr_targets = ['obscene','insult','toxic']
    for label in cols_target:
        submission_combined[label] = 0.5 * (submission_chains[label] + submission_binary[label])

    # generate submission file
    submission_combined.to_csv('artifacts/submission_combined.csv', index=False)
    return submission_combined


X, test_X = preprocess_data(train_df, test_df)
train(X, test_X)
