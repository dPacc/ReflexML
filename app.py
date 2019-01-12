# Importing the Flask packages
from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, IMAGES, DATA, ALL
from werkzeug import secure_filename
from flask_sqlalchemy import SQLAlchemy

import os
import datetime
import time

# Exploratory Data Analysis Packages
import pandas as pd
import numpy as np

# # Machine Learning packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Machine Learning Packages for Vectorization of Text for Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)


# Configuration for file UploadSet
files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app, files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'

db = SQLAlchemy(app)

# Saving the data to the Database Storage
# class FileContents(db.Model):
# 	id = db.Column(db.Integer,primary_key=True)
# 	name = db.Column(db.String(300))
# 	modeldata = db.Column(db.String(300))
# 	data = db.Column(db.LargeBinary)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/datauploads', methods=['GET','POST'])
def datauploads():
    if request.method == 'POST' and 'csv_data' in request.files:
        file = request.files['csv_data']
        filename = secure_filename(file.filename)
        # os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
        file.save(os.path.join('static/uploadsDB', filename))
        fullfile = os.path.join('static/uploadsDB',filename)

        # Date
        date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

        # EDA Function
        data = pd.read_csv(os.path.join('static/uploadsDB', filename))
        data_size = data.size
        data_shape = data.shape
        data_columns = list(data.columns)
        data_targetname = data[data.columns[-1]].name
        data_features = data_columns[0: -1]

        data_xfeatures = data.iloc[:, 0:-1]
        data_ylabels = data[data.columns[-1]]

        # Create a Table
        data_table = data
        X = data_xfeatures
        y = data_ylabels

        # Models
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))


        results = []
        names = []
        allmodels = []
        scoring = 'accuracy'
        best_scores = list([])

        for name, model in models:
            kfold = model_selection.KFold(n_splits=10)
            cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f: (%f)" % (name, cv_results.mean(), cv_results.std())
            allmodels.append(msg)
            model_results = results
            model_names = names

    return render_template('details.html', filename=filename,
                            date=date,
                            data_table=data,
                            data_shape=data_shape,
                            data_size=data_size,
                            data_columns=data_columns,
                            data_targetname=data_targetname,
                            model_results=allmodels,
                            model_names=model_names,
                            fullfile=fullfile)


if __name__ == "__main__":
    app.run(debug=True)
