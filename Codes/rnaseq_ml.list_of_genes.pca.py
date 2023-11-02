#!/usr/bin/env python3

# PCA plot for a list of DEGs
# Author: Yue Xing

import pandas as pd
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
out_tp=".".join([out,tp,"pca.png"])

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

	y=trti['Group']

	X = StandardScaler().fit_transform(df)
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(X)
	principalDf = pd.DataFrame(data = principalComponents, 
	    columns = ['principal component 1', 'principal component 2'])

	yl=y.reset_index()
	finalDf = pd.concat([principalDf, yl], axis = 1)
	#finalDf.head()

	yn=y.to_numpy()

	targets = [gr1, gr2]
	colors = ['blue', 'red']
	for target, color in zip(targets,colors):
	    indicesToKeep = yn == target
	    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], 
	        finalDf.loc[indicesToKeep, 'principal component 2'], 
	        c = color, s = 50)
	    plt.xlabel('Principal Component 1', fontsize = 15)
	    plt.ylabel('Principal Component 2', fontsize = 15)

	plt.legend(targets)
	plt.savefig(out_tp, dpi=300)

f.close()

