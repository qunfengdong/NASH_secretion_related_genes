#!/usr/bin/env python3

# Classification for a list of DEGs
# Author: Yue Xing

import pandas as pd
from optparse import OptionParser
import numpy as np
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

parser=OptionParser(description='Machine Learning for comparing two groups in RNA-seq, pairwise')
parser.add_option('-c', '--count_t', help="Gene count table")
parser.add_option('-t', '--treatment_t', help="Treatment/Group table")
parser.add_option('-g', '--genes', help="Genes to choose")
parser.add_option('--group1', help="Group 1")
parser.add_option('--group2', help="Group 2")
parser.add_option('-o', '--out_pref', help="Prefix for output")
parser.add_option('-m', '--method', help="Machine learning method")

(opts, args)=parser.parse_args()

gr1=opts.group1
gr2=opts.group2
pref=opts.out_pref
tp=opts.method

out=".".join([pref,gr1,gr2])
out_t=".".join([out,tp,"out.txt"])
out_tp=".".join([out,tp,"out.png"])
out_c=".".join([out,tp,"out.csv"])

with open(out_t, 'w') as f:
	df = pd.read_csv(opts.count_t,index_col=0)
	df=df.transpose()
	print(df.head(), file=f) 

	trt = pd.read_csv(opts.treatment_t,index_col=2)
	print(trt.head(), file=f) 

	de = pd.read_csv(opts.genes)
	print(de.head(), file=f)
	deg = list(de.iloc[:,0])
	df = df[deg]

	trti=trt.loc[trt['Group'].isin([gr1,gr2])]
	idx=trti.index
	df=df.reindex(index=idx)

	ss=list(df.index==trti.index)
	print(ss.count(False), file=f)
	print(df.head(), file=f)
	print(df.shape, file=f)

	y=trti['Group'].to_numpy()
	X=df.to_numpy()

	yb = label_binarize(y, classes=[gr1,gr2])
	yb = np.ravel(yb)

	data = pd.DataFrame(list(df.columns),columns=["ID"],index=list(df.columns))

	### Leave-one-out
	cv = LeaveOneOut()

	if tp=="RF":
		from sklearn.ensemble import RandomForestClassifier
	elif tp=="XGB":
		from xgboost import XGBClassifier
	elif tp=="ADB":
		from sklearn.ensemble import AdaBoostClassifier
	elif tp=="GDB":
		from sklearn.ensemble import GradientBoostingClassifier
	elif tp=="LG":
		from sklearn.linear_model import LogisticRegression
	elif tp=="LGB":
		from lightgbm import LGBMClassifier

	all_yt = []
	all_yp=[]
	all_ypv=[]
	scores=[]
	for train, test in cv.split(X, yb):
		if tp=="RF":
			model = RandomForestClassifier(random_state=1)
		elif tp=="XGB":
			model = XGBClassifier()
		elif tp=="ADB":
			model = AdaBoostClassifier()
		elif tp=="GDB":
			model = GradientBoostingClassifier()
		elif tp=="LG":
			model = LogisticRegression(solver='liblinear', random_state=0)
		elif tp=="LGB":
			model = LGBMClassifier()

		model.fit(X[train], yb[train])
		all_yt.append(yb[test])
		all_yp.append(model.predict_proba(X[test])[:,1])
		all_ypv.append(model.predict(X[test]))
		scores.append(accuracy_score(yb[test], model.predict(X[test])))

		feature_imp = pd.Series(model.feature_importances_,index=df.columns).to_frame()
		#data=data.merge(feature_imp,left_index=True,right_index=True,how="outer")
		#print("===================", file=f)
		#print(feature_imp, file=f)

	all_ms=mean(scores)
	all_sds=std(scores)
	all_f1s=f1_score(all_yt,all_ypv)

	all_yt = np.array(all_yt)
	all_yp = np.array(all_yp)

	fpr, tpr, thrs = roc_curve(all_yt,all_yp)
	roc_auc = auc(fpr, tpr)

	# plot the roc curve for the model
	pyplot.plot(fpr, tpr, marker='.')
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	pyplot.show()

	pyplot.savefig(out_tp)
	pyplot.close()

	print(all_ms, file=f)
	print(all_sds, file=f)
	print(all_f1s, file=f)
	print(roc_auc, file=f)

	data.to_csv(out_c)
	feature_imp.to_csv(out_c + "ft_i")

f.close()

