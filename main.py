import pandas as pd  # data analysis library
import numpy as np  # library linear algebra, Fourier transform and random numbers

# sklearn - а set of python modules for machine learning and data mining
from sklearn.ensemble import RandomForestRegressor  # using the Random Forest Regressor
from sklearn.feature_extraction.text import TfidfVectorizer  # for convert a collection of raw documents to a matrix of TF-IDF features
from sklearn.linear_model import Ridge, LinearRegression  # Ridge - Linear least squares with l2 regularization, Linear Regression - ordinary least squares
from sklearn.pipeline import Pipeline, FeatureUnion  # module implements utilities to build a composite estimator, as a chain of transforms and estimators
from sklearn.base import TransformerMixin, BaseEstimator # TransformerMixin - Mixin class for all transformers in scikit-learn.


import re  # module for working with regular expressions
import scipy  # library is built to work with NumPy arrays, and provides efficient numerical routines such as routines for numerical integration and optimization
from scipy import sparse  # SciPy 2-D sparse matrix package for numeric data
import gc # Garbage Collector - module provides the ability to disable the collector, tune the collection frequency, and set debugging options
from IPython.display import display, HTML  # Jupyter kernel to work with Python code in Jupyter notebooks and other interactive frontends
from pprint import pprint  # module provides a capability to “pretty-print” arbitrary Python data structures in a form which can be used as input to the interpreter
import warnings  # Warning messages are typically issued in situations where it is useful to alert the user of some condition in a program

warnings.filterwarnings("ignore")  # This is the base class of all warning category classes. It is a subclass of Exception.
# The warnings filter controls whether warnings are ignored, displayed, or turned into errors (raising an exception)
# "ignore" - never print matching warnings

pd.options.display.max_colwidth=300  # The maximum width in characters of a column in the repr of a pandas data structure.
#Wen the column overflows, a “…” placeholder is embedded in the output
# this block is needed only for understanding what data we are working with
#  We use data from the 2017 competition "The problem of classification of toxic comments"
df = pd.read_csv("./input/jigsaw-toxic-comment-classification-challenge/train.csv")  # read the data for training and put it in the date frame 'df'
#print(df.shape)

for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:  # we iterate over each taxation column in the table
    #print(f'****** {col} *******')  # display the name of the processed column
    display(df.loc[df[col]==1,['comment_text',col]].sample(10))
    # we will display 10 examples (rows) of the table each in which the column of the value of the given taxation category is equal to one

# Give more weight to severe toxic
df['severe_toxic'] = df.severe_toxic * 2  # multiply the highly toxic value of the column by 2. While the remaining toxicity columns remain at one.
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
# Let's add one more column "y" to our dataframe - equal to the sum of all toxicity values.
# Since we have 6 degrees of toxicity with maximum values in the column:
# 'toxic' = 1
# 'severe_toxic' = 2
# 'obscene' = 1
# 'threat' = 1
# 'insult' = 1
# 'identity_hate' = 1
# the most toxic comment will collect all levels of toxicity 1 + 2 + 1 + 1 + 1 + 1 = 7

df['y'] = df['y']/df['y'].max()  # Let's normalize the values, not from 0 to 7, but from 0 to 1.
# Where 0 is a non-toxic comment, 1 - corresponds to the presence of all signs of toxicity

df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})  # rename column 'comment_text' in 'text'
#text y
#@#!# [0,1]
df.sample(5)  # we will display 5 examples (rows)
df['y'].value_counts()
#print(df['y'])
#for item in df:
#    print(item)
# Divide the resulting dataframe into 7 and save each in a separate csv output file.
# It should be noted that the division into 7 folders is not linear, so we minimize the skew in the number of values, although it will not play a special role here.
n_folds = 7  # number of folders

frac_1 = 0.7
frac_1_factor = 1.5

for fld in range(n_folds):  # iterate over each of the 7 folders in turn
    #print(f'Fold: {fld}')  # display the name of the currently formed folder
    tmp_df = pd.concat([df[df.y>0].sample(frac=frac_1, random_state = 10*(fld+1)) ,
                        df[df.y==0].sample(n=int(len(df[df.y>0])*frac_1*frac_1_factor) ,
                                            random_state = 10*(fld+1))], axis=0).sample(frac=1, random_state = 10*(fld+1))
    # use handling of joining pandas objects along a specific axis with optional setup logic

    tmp_df.to_csv(f'./working/df_fld:{fld}.csv', index=False)  # save the resulting folder dataframe to a csv file and mark it in a folder '/kaggle/working/'
    #print(tmp_df.shape)  # display statistics for in this file, how many comments correspond to one of the 8 degrees of toxicity
    #print(tmp_df['y'].value_counts())  # display statistics in this file. As we can see, all files will contain the same number of lines.


def clean(data, col):  # Replace each occurrence of pattern/regex in the Series/Index

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)', r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}', r'\1\1\1')
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)', r' \1 ')
    # patterns with repeating characters
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b', r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()

    return data  # the function returns the processed value
# Test clean function
test_clean_df = pd.DataFrame({"text":
                              ["heyy\n\nkkdsfj",
                               "hi   how/are/you ???",
                               "hey?????",
                               "noooo!!!!!!!!!   comeone !! ",
                              "cooooooooool     brooooooooooo  coool brooo",
                              "naaaahhhhhhh"]})
display(test_clean_df)  # display the test function before transformation
clean(test_clean_df,'text')  # display the test function after transformation
df = clean(df,'text')  # clear the whole date frame

# # Divide the resulting cleared dataframe by 7 and save each in a separate output csv file.
# It should be noted that, as before, the separation rule is respected. In this way, we created 7 cleared and not cleared data files.
n_folds = 7  # number of folders

frac_1 = 0.7
frac_1_factor = 1.5

for fld in range(n_folds):  # iterate over each of the 7 folders in turn
    #print(f'Fold: {fld}')  # display the name of the currently formed folder
    tmp_df = pd.concat([df[df.y > 0].sample(frac=frac_1, random_state=10 * (fld + 1)),
                        df[df.y == 0].sample(n=int(len(df[df.y > 0]) * frac_1 * frac_1_factor),
                                             random_state=10 * (fld + 1))], axis=0).sample(frac=1,
                                                                                           random_state=10 * (fld + 1))
    # use handling of joining pandas objects along a specific axis with optional setup logic

    tmp_df.to_csv(f'./working/df_clean_fld:{fld}.csv',
                  index=False)  # save the resulting folder dataframe to a csv file and mark it in a folder '/kaggle/working/'
    #print(tmp_df.shape)  # display statistics for in this file, how many comments correspond to one of the 8 degrees of toxicity
    #print(tmp_df['y'].value_counts())  # display statistics in this file. As we can see, all files will contain the same number of lines.


del df,tmp_df  # remove the applied date frames
gc.collect()  # With no arguments, run a full collection,

#Ruddit data
df_ = pd.read_csv("./input/ruddit/ruddit_with_text.csv")  # create a dateframe based on a file

#print(df_.shape)  # display its size

df_ = df_[['txt', 'offensiveness_score']].rename(columns={'txt': 'text',
                                                                'offensiveness_score':'y'})  # change columns

df_['y'] = (df_['y'] - df_.y.min()) / (df_.y.max() - df_.y.min())  # converting all toxicity values from 0 to 1
#df_.y.hist()  # display all values on the histogram

# Divide the resulting cleared dataframe by 7 and save each in a separate output csv file.
n_folds = 7  # number of folders

frac_1 = 0.7  # for all categories we take 70% of the original amount

for fld in range(n_folds):  # iterate over each of the 7 folders in turn
    print(f'Fold: {fld}RUddit')  # display the name of the currently formed folder
    tmp_df = df_.sample(frac=frac_1, random_state = 10*(fld+1))  # use handling of joining pandas objects along a specific axis with optional setup logic
    tmp_df.to_csv(f'./working/df2_fld:{fld}.csv', index=False)  # save the resulting folder dataframe to a csv file and mark it in a folder '/kaggle/working/'
    #print(tmp_df.shape)  # display statistics for in this file
    #print(tmp_df['y'].value_counts())  # display statistics in this file. As we can see, all files will contain the same number of lines.
del tmp_df, df_;  # remove the applied date frames
gc.collect()  # With no arguments, run a full collection

# Validation data

df_val = pd.read_csv("./input/jigsaw-toxic-severity-rating/validation_data.csv")  # create a variable dataframe containing data from the original competition data file
#print(df_val.shape)  # display statistics for in this file
#print(df_val.head())  # display the first 5 rows of the dataframe table

# Test data

df_sub = pd.read_csv("./input/jigsaw-toxic-severity-rating/comments_to_score.csv")  # create a variable dataframe containing data from the original competition data file
#print(df_sub.shape)  # display statistics for in this file
#print(df_sub.head())
# Test data

#build machine learning model
# NOT USED
# class LengthTransformer(BaseEstimator, TransformerMixin):

#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         return sparse.csr_matrix([[(len(x)-360)/550] for x in X])
#     def get_feature_names(self):
#         return ["lngth"]

class LengthUpperTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return sparse.csr_matrix([[sum([1 for y in x if y.isupper()])/len(x)] for x in X])
    def get_feature_names(self):
        return ["lngth_uppercase"]

df_val['upper_1'] = np.array(LengthUpperTransformer().transform(df_val['less_toxic']).todense()).reshape(-1,1)
df_val['upper_2'] = np.array(LengthUpperTransformer().transform(df_val['more_toxic']).todense()).reshape(-1,1)

#print(df_val['upper_1'].mean(), df_val['upper_1'].std())
#print(df_val['upper_2'].mean(), df_val['upper_2'].std())

#df_val['upper_1'].hist(bins=100)
#df_val['upper_2'].hist(bins=100)

val_preds_arr1 = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2 = np.zeros((df_val.shape[0], n_folds))
test_preds_arr = np.zeros((df_sub.shape[0], n_folds))
#train on toxic data
for fld in range(n_folds):
    #print("\n\n")
    #print(f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv('./working/df_fld{fld}.csv')
    #print(df.shape)

    features = FeatureUnion([
        # ('vect1', LengthTransformer()),
        # ('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df=3, max_df=0.5, analyzer='char_wb', ngram_range=(3, 5))),
        # ("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),

    ])
    pipeline = Pipeline(
        [
            ("features", features),
            # ("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            # ("clf",LinearRegression())
        ]
    )
    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'], df['y'])

    # What are the important features for toxicity

    print('\nTotal number of features:', len(pipeline['features'].get_feature_names()))

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(),
                                  np.round(pipeline['clf'].coef_, 2))),
                         key=lambda x: x[1],
                         reverse=True)

    pprint(feature_wts[:30])

    print("\npredict validation data ")
    val_preds_arr1[:, fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2[:, fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arr[:, fld] = pipeline.predict(df_sub['text'])
#train on Toxic clean data
val_preds_arr1c = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2c = np.zeros((df_val.shape[0], n_folds))
test_preds_arrc = np.zeros((df_sub.shape[0], n_folds))

for fld in range(n_folds):
    print("\n\n")
    print(f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv(f'./working/df_clean_fld{fld}.csv')
    print(df.shape)

    features = FeatureUnion([
        # ('vect1', LengthTransformer()),
        # ('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df=3, max_df=0.5, analyzer='char_wb', ngram_range=(3, 5))),
        # ("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),

    ])
    pipeline = Pipeline(
        [
            ("features", features),
            # ("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            # ("clf",LinearRegression())
        ]
    )
    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'], df['y'])

    # What are the important features for toxicity

    print('\nTotal number of features:', len(pipeline['features'].get_feature_names()))

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(),
                                  np.round(pipeline['clf'].coef_, 2))),
                         key=lambda x: x[1],
                         reverse=True)

    pprint(feature_wts[:30])

    print("\npredict validation data ")
    val_preds_arr1c[:, fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2c[:, fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arrc[:, fld] = pipeline.predict(df_sub['text'])


#train on Ruddit data pipeline
val_preds_arr1_ = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2_ = np.zeros((df_val.shape[0], n_folds))
test_preds_arr_ = np.zeros((df_sub.shape[0], n_folds))

for fld in range(n_folds):
    print("\n\n")
    print(f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv('./working/df2_fld{fld}.csv')
    print(df.shape)

    features = FeatureUnion([
        # ('vect1', LengthTransformer()),
        # ('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df=3, max_df=0.5, analyzer='char_wb', ngram_range=(3, 5))),
        # ("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),

    ])
    pipeline = Pipeline(
        [
            ("features", features),
            # ("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            # ("clf",LinearRegression())
        ]
    )
    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'], df['y'])

    # What are the important features for toxicity

    print('\nTotal number of features:', len(pipeline['features'].get_feature_names()))

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(),
                                  np.round(pipeline['clf'].coef_, 2))),
                         key=lambda x: x[1],
                         reverse=True)

    pprint(feature_wts[:30])

    print("\npredict validation data ")
    val_preds_arr1_[:, fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2_[:, fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arr_[:, fld] = pipeline.predict(df_sub['text'])
del df, pipeline, feature_wts
gc.collect()

print(" Toxic data ")
p1 = val_preds_arr1.mean(axis=1)
p2 = val_preds_arr2.mean(axis=1)

print(f'Validation Accuracy is { np.round((p1 < p2).mean() * 100,2)}')

print(" Ruddit data ")
p3 = val_preds_arr1_.mean(axis=1)
p4 = val_preds_arr2_.mean(axis=1)

print(f'Validation Accuracy is { np.round((p3 < p4).mean() * 100,2)}')

print(" Toxic CLEAN data ")
p5 = val_preds_arr1c.mean(axis=1)
p6 = val_preds_arr2c.mean(axis=1)

print(f'Validation Accuracy is { np.round((p5 < p6).mean() * 100,2)}')

print("Find right weight")
#Validate the pipeline
wts_acc = []
for i in range(30,70,1):
    for j in range(0,20,1):
        w1 = i/100
        w2 = (100 - i - j)/100
        w3 = (1 - w1 - w2 )
        p1_wt = w1*p1 + w2*p3 + w3*p5
        p2_wt = w1*p2 + w2*p4 + w3*p6
        wts_acc.append( (w1,w2,w3,
                         np.round((p1_wt < p2_wt).mean() * 100,2))
                      )
sorted(wts_acc, key=lambda x:x[3], reverse=True)[:5]

w1,w2,w3,_ = sorted(wts_acc, key=lambda x:x[2], reverse=True)[0]
#print(best_wts)

p1_wt = w1*p1 + w2*p3 + w3*p5
p2_wt = w1*p2 + w2*p4 + w3*p6

w1,w2,w3,_ = sorted(wts_acc, key=lambda x:x[2], reverse=True)[0]
#print(best_wts)

p1_wt = w1*p1 + w2*p3 + w3*p5
p2_wt = w1*p2 + w2*p4 + w3*p6
df_val['p1'] = p1_wt
df_val['p2'] = p2_wt
df_val['diff'] = np.abs(p2_wt - p1_wt)

df_val['correct'] = (p1_wt < p2_wt).astype('int')

### Incorrect predictions with similar scores

df_val[df_val.correct == 0].sort_values('diff', ascending=True).head(20)
### Incorrect predictions with dis-similar scores


df_val[df_val.correct == 0].sort_values('diff', ascending=False).head(20)
# Predict using pipeline

df_sub['score'] = w1*test_preds_arr.mean(axis=1) + w2*test_preds_arr_.mean(axis=1) + w3*test_preds_arrc.mean(axis=1)
#test_preds_arr
print(df_sub)
# Cases with duplicates scores

df_sub['score'].count() - df_sub['score'].nunique()
same_score = df_sub['score'].value_counts().reset_index()[:10]
same_score
df_sub[df_sub['score'].isin(same_score['index'].tolist())]
# # Rank the predictions

df_sub['score']  = scipy.stats.rankdata(df_sub['score'], method='ordinal')

print(df_sub['score'].rank().nunique())
df_sub[['comment_id', 'score']].to_csv("submission.csv", index=False)  # we form an output file for evaluation in the competition


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
