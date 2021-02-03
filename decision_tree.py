#-*- coding: utf-8 -*-
# author: B. X. Weinstein Xiao
# contact: rchopin@outlook.com
import pandas as pd
import os
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from IPython.core.pylabtools import figsize
import codecs
import _pickle as cPickle
from sklearn.model_selection import GridSearchCV

try:
    from pydotplus import graph_from_dot_data
    import graphviz
except ImportError:
    print('Please check if you have installed packages: graphviz, pydotplus.')
from sklearn.externals.six import StringIO

figsize(12.5, 6)

########################################################################################################################
# Please check that you have correctly installed 'Graphviz 2.38'.
# If not, go to https://graphviz.gitlab.io/_pages/Download/Download_windows.html, download and install the '.msi' file.
# Remember to add the env variable path as follows:
# C:\\Program Files (x86)\\Graphviz2.38
# C:\\Program Files (x86)\\Graphviz2.38\\bin
########################################################################################################################

########################################################################################################################
# Example:
# inputfile = 'sales_data.xls'
# data = pd.read_excel(inputfile, index_col = u'序号')

# data[data == u'好'] = 1
# data[data == u'是'] = 1
# data[data == u'高'] = 1
# data[data != 1] = -1
# x = data.iloc[:,:3].astype('int')
# y = data.iloc[:,3].astype('int')

# viz = TreeViz(x, y, regressor='classification')
# mod = viz.train_model()
# viz.viz(mod)
########################################################################################################################


os.environ["path"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38"
os.environ["path"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin"

class TreeViz(object):

    def __init__(self, X, Y, regressor='classification', grid_search=False):
        self.X = X
        self.Y = Y
        self.task_type = regressor
        self.grid_search = grid_search

    def train_model(self):
        if self.grid_search:
            m, n = self.X.shape
            parameters = {'min_samples_split': range(2, int(m/3)),
                          'max_depth': range(2, int(n/2)),
                          'min_samples_leaf': range(1, int(m/14))}
            if self.task_type == 'classification':
                model = DecisionTreeClassifier(criterion='entropy')
                model_search = GridSearchCV(model, parameters)
                model_search.fit(self.X, self.Y)
                best_params = model_search.best_params_
                model_ = DecisionTreeClassifier(criterion='entropy',
                                                max_depth=best_params['max_depth'],
                                                min_samples_leaf=best_params['min_samples_leaf'],
                                                min_samples_split=best_params['min_samples_split'])
                model_.fit(self.X, self.Y)
            elif self.task_type == 'regression':
                model = DecisionTreeRegressor(criterion='entropy')
                model_search = GridSearchCV(model, parameters)
                model_search.fit(self.X, self.Y)
                best_params = model_search.best_params_
                model_ = DecisionTreeClassifier(criterion='entropy',
                                                max_depth=best_params['max_depth'],
                                                min_samples_leaf=best_params['min_samples_leaf'],
                                                min_samples_split=best_params['min_samples_split'])
                model_.fit(self.X, self.Y)
            else:
                model_ = None
                print('Wrong task type. (regression or classification)')

            return model_

        else:
            if self.task_type == 'classification':
                model = DecisionTreeClassifier(criterion='entropy', max_depth=120, min_samples_leaf=1300)
                model.fit(self.X, self.Y)
            elif self.task_type == 'regression':
                model = DecisionTreeRegressor(criterion='entropy')
                model.fit(self.X, self.Y)
            else:
                model = None
                print('Wrong task type. (regression or classification)')
            return model

    def viz(self, model, txt_dir='dot_data.txt', txt_dir_utf8='dot_data_utf8.txt', filename='test.png'):
        dot_data = StringIO()
        export_graphviz(model,
                        out_file=dot_data,
                        feature_names=list(self.X.columns),
                        filled=True,
                        rounded=True)

        with open('dot_data.txt', 'w', encoding='utf-8') as f:
            f.writelines(dot_data.getvalue())

        with codecs.open(txt_dir, 'r', encoding='utf-8') as f, \
            codecs.open(txt_dir_utf8, 'w', encoding='utf-8') as wf:
            for line in f:
                lines = line.strip().split('\t')
                if 'label' in lines[0]:
                    newline = lines[0].replace('\n', '').replace(' ', '')
                else:
                    newline = lines[0].replace('\n', '').replace('helvetica', ' "Microsoft YaHei" ')
                wf.write(newline + '\t')

        with open(txt_dir_utf8, encoding='utf-8') as f:
            dot_graph = f.read()

        graph = graph_from_dot_data(dot_graph)
        graph.write_png(filename)


files = os.listdir('D:\\tree regularization project\\Beat')
dats_tst = []
dats_trn = []
x_train = []
y_train = []
y_test = []
x_test = []
for i in files:
    d = np.array(pd.read_csv('./Beat/'+i, engine='python', encoding='utf_8_sig'))
    dats_trn.append(d[:5000, :])
    dats_tst.append(d[5000:, ])

for i, j in enumerate(dats_trn):
    dum = [0, 0, 0, 0]
    dum[i] = 1
    y_train += [dum for _ in range(len(j))]
    x_train += j.tolist()
train_zip = list(zip(y_train, x_train))
random.shuffle(train_zip)
for i, j in enumerate(dats_tst):
    dum = [0, 0, 0, 0]
    dum[i] = 1
    y_test += [dum for _ in range(len(j))]
    x_test += j.tolist()
test_zip = list(zip(y_test, x_test))
random.shuffle(test_zip)
x_train = [x for _, x in train_zip]
y_train = [y for y, _ in train_zip]
x_test = [x for _, x in test_zip]
y_test = [y for y, _ in test_zip]
del train_zip
del test_zip
X = np.array(x_train)
X = pd.DataFrame(X)
Y = np.argmax(np.array(y_train), axis=1)
Y = np.reshape(Y, (20000, -1))


dats = []
for i in files:
    dats.append(np.array(pd.read_csv('./Beat/'+i, engine='python', encoding='utf_8_sig')))
x_train = []
y_train = []
for i, j in enumerate(dats):
    n = j.shape[0]
    y_train += [i for _ in range(n)]
    x_train += j.tolist()
y_train = np.array(y_train)
x_train = np.array(x_train)
X, XT, Y, YT = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train)
X = pd.DataFrame(X)
Y = np.reshape(Y, (-1, 1))
viz = TreeViz(X, Y)
mod = viz.train_model()
y_pred = mod.predict(XT)
confusion_matrix(Y, y_pred)
viz.viz(mod)