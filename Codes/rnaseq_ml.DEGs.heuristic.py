#!/usr/bin/env python3

# Heuristic strategy to identify subsets of DEGs for best classification performance
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
from sklearn.metrics import mean_absolute_error

parser=OptionParser(description='Machine Learning for comparing two groups in RNA-seq, pairwise')
parser.add_option('-c', '--count_t', help="Gene count table")
parser.add_option('-t', '--treatment_t', help="Treatment/Group table")
parser.add_option('-d', '--de_t', help="DE table")
parser.add_option('--group1', help="Group 1")
parser.add_option('--group2', help="Group 2")
parser.add_option('-o', '--out_pref', help="Prefix for output")
parser.add_option('-m', '--method', help="Machine learning method")
parser.add_option('-l', '--limit', help="Number of genes to limit to")
parser.add_option('-r', '--order_method', help="Method used to order genes")

(opts, args)=parser.parse_args()

gr1=opts.group1
gr2=opts.group2
pref=opts.out_pref

count_t=opts.count_t
treatment_t=opts.treatment_t
de_t=opts.de_t
lmt=int(opts.limit)
om=opts.order_method
tp=opts.method

out=".".join([pref,om,"lmt",str(lmt),gr1,gr2,tp])
out_t=".".join([out,"out.txt"])
out_tp=".".join([out,"out.png"])

#def reg_sco(y_pred, y_true):
#	error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5
#	sco = 1 - error
#	return sco

with open(out_t, 'w') as f:
	df = pd.read_csv(count_t,index_col=0)
	df=df.transpose()
	print(df.head(), file=f) 

	trt = pd.read_csv(treatment_t,index_col=2)
	print(trt.head(), file=f) 

	de = pd.read_csv(de_t,index_col=0)
	print(de.head(), file=f) 

	# pre-filter
	# make sure all genes are significant
	de = de.loc[de['padj']<0.05]
	#de = de.loc[abs(de['log2FoldChange'])>float(logfc)]

	if om=="adjp":
		de=de.sort_values(by=['padj'])
	elif om=="lfc":
		de['abs_lfc']=abs(de['log2FoldChange'])
		de=de.sort_values(by=['abs_lfc'],ascending=False)
	elif om=="pi":
		de['pi']=de['padj']**abs(de['log2FoldChange'])
		de=de.sort_values(by=['pi'])

	n0=de.shape[0]
	if n0<lmt:
		lmt=n0

	de=de.iloc[0:lmt,]
	print(de.shape, file=f)
	print(de, file=f) 
	deg = list(de.index)
	if len(deg)==0:
		print("No DE genes avaliable", file=f) 
		f.close()
		exit(0)

	df = df[deg]

	trti=trt.loc[trt['Group'].isin([gr1,gr2])]
	idx=trti.index
	df=df.reindex(index=idx)

	ss=list(df.index==trti.index)
	if ss.count(False) != 0:
		print("Error whole data", file=f)

	a=list(range(len(deg)))
	b = [x+1 for x in a]

	x0=np.array(b)

	all_ms = []
	all_sds = []
	all_auc = []
	all_f1s = []
	all_scores = []

	for i in b:
		#print(i, file=f)
		outp=".".join([out,str(i),"png"])

		de_l = deg[0:i]
		dfi = df[de_l]

		ss=list(dfi.index==trti.index)
		if ss.count(False) != 0:
			print(" ".join(["Error",str(i)]), file=f)

		y=trti['Group'].to_numpy()
		X=dfi.to_numpy()
		yb = label_binarize(y, classes=[gr1,gr2])
		yb = np.ravel(yb)

		### Leave-one-out
		#cvi = LeaveOneOut()
		cv = LeaveOneOut()

		if tp=="RF":
			from sklearn.ensemble import RandomForestClassifier
			#modeli = RandomForestClassifier(random_state=1)
		elif tp=="XGB":
			from xgboost import XGBClassifier
			#modeli = XGBClassifier()
		elif tp=="ADB":
			from sklearn.ensemble import AdaBoostClassifier
			#modeli = AdaBoostClassifier()
		elif tp=="GDB":
			from sklearn.ensemble import GradientBoostingClassifier
			#modeli = GradientBoostingClassifier()
		elif tp=="LS":
			from sklearn.linear_model import Lasso
		elif tp=="EN":
			from sklearn.linear_model import ElasticNet
		elif tp=="LGB":
			from lightgbm import LGBMClassifier

		#scores = cross_val_score(modeli, X, y, scoring='accuracy', cv=cvi, n_jobs=-1)
		#all_ms.append(mean(scores))
		#all_sds.append(std(scores))

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
			elif tp=="LS":
				model = Lasso(alpha=1.0,normalize=True)
			elif tp=="EN":
				model = ElasticNet(alpha=1.0, l1_ratio=0.5)
			elif tp=="LGB":
				model = LGBMClassifier()
			
			model.fit(X[train], yb[train])
			all_yt.append(yb[test])
			all_ypv.append(model.predict(X[test]))

			if tp not in ['LS',"EN"]:
				scores.append(accuracy_score(yb[test], model.predict(X[test])))
				all_yp.append(model.predict_proba(X[test])[:,1])

		if tp not in ['LS',"EN"]:
			all_ms.append(mean(scores))
			all_sds.append(std(scores))
			all_f1s.append(f1_score(all_yt,all_ypv))
			#all_scores.append(accuracy_score(all_yt,all_ypv))

			all_yt = np.array(all_yt)
			all_yp = np.array(all_yp)
			fpr, tpr, thrs = roc_curve(all_yt,all_yp)
			roc_auc = auc(fpr, tpr)
			all_auc.append(roc_auc)
			# plot the roc curve for the model
			#pyplot.plot(fpr, tpr, marker='.')
			#pyplot.xlabel('False Positive Rate')
			#pyplot.ylabel('True Positive Rate')
			#pyplot.show()

			#pyplot.savefig(outp)
			#pyplot.close()
		else:
			all_scores.append(mean_absolute_error(all_yt, all_ypv))

	if tp not in ['LS',"EN"]:
		print(all_auc, file=f)
		print(all_f1s, file=f)
		#print(all_scores, file=f)
		print(all_ms, file=f)
		print(all_sds, file=f)

		all_ms=np.array(all_ms)
		all_sds=np.array(all_sds)
		all_auc=np.array(all_auc)
		all_f1s=np.array(all_f1s)

		fig, (ax0, ax1, ax2) = pyplot.subplots(nrows=3, sharex=True)
		ax0.errorbar(x0, all_ms, yerr=all_sds, fmt='-o')
		ax0.set_title('Accuracy score and AUC')
		ax1.plot(x0, all_auc, '-o')
		ax2.plot(x0, all_f1s, '-o')
		pyplot.show()
		pyplot.savefig(out_tp)
		pyplot.close()
	else:
		print(all_scores, file=f)
		fig, (ax0) = pyplot.subplots(nrows=1, sharex=True)
		ax0.plot(x0, all_scores, '-o')
		ax0.set_title('Accuracy score')
		pyplot.show()
		pyplot.savefig(out_tp)
		pyplot.close()

f.close()


