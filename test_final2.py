import pandas as pd
import numpy as np
reader=pd.read_csv('breast cancer.csv',header=None)
#print reader
pd.options.mode.chained_assignment = None
#for row in range(0,len(reader)):
#	print reader[row]
l=[]

#for i in range(0,len(reader)):
#	print len(reader[i])
#	for j in range(0,10):
#		print type(reader[i][j])
#		#l.append(reader[i][j])
#print l
for i in range(0,len(reader[0])):
	if(reader[0][i]=='recurrence-events'):
		reader[0][i]=1
	else:
		reader[0][i]=2
for i in range(0,len(reader[1])):
	if(reader[1][i]=='10-19'):
		reader[1][i]=1
	elif(reader[1][i]=='20-29'):
		reader[1][i]=2
	elif(reader[1][i]=='30-39'):
		reader[1][i]=3
	elif(reader[1][i]=='40-49'):
		reader[1][i]=4
	elif(reader[1][i]=='50-59'):
		reader[1][i]=5
	elif(reader[1][i]=='60-69'):
		reader[1][i]=6
	elif(reader[1][i]=='70-79'):
		reader[1][i]=7
	elif(reader[1][i]=='80-89'):
		reader[1][i]=8
	elif(reader[1][i]=='90-99'):
		reader[1][i]=9
for i in range(0,len(reader[2])):
	if(reader[2][i]=='premeno'):
		reader[2][i]=1
	elif(reader[2][i]=='ge40'):
		reader[2][i]=2
	elif(reader[2][i]=='lt40'):
		reader[2][i]=3
for i in range(0,len(reader[3])):

	if(reader[3][i]=='0-4'):
		reader[3][i]=1
	elif(reader[3][i]=='5-9'):
		reader[3][i]=2
	elif(reader[3][i]=='10-14'):
		reader[3][i]=3
	elif(reader[3][i]=='15-19'):
		reader[3][i]=4
	elif(reader[3][i]=='20-24'):
		reader[3][i]=5
	elif(reader[3][i]=='25-29'):
		reader[3][i]=6
	elif(reader[3][i]=='30-34'):
		reader[3][i]=7
	elif(reader[3][i]=='35-39'):
		reader[3][i]=8
	elif(reader[3][i]=='40-44'):
		reader[3][i]=9
	elif(reader[3][i]=='45-49'):
		reader[3][i]=10
	elif(reader[3][i]=='50-54'):
		reader[3][i]=11
	elif(reader[3][i]=='55-59'):
		reader[3][i]=12
for i in range(0,len(reader[4])):

	if(reader[4][i]=='0-2'):
		reader[4][i]=1
	elif(reader[4][i]=='3-5'):
		reader[4][i]=2
	elif(reader[4][i]=='6-8'):
		reader[4][i]=3
	elif(reader[4][i]=='9-11'):
		reader[4][i]=4
	elif(reader[4][i]=='12-14'):
		reader[4][i]=5
	elif(reader[4][i]=='15-17'):
		reader[4][i]=6
	elif(reader[4][i]=='18-20'):
		reader[4][i]=7
	elif(reader[4][i]=='21-23'):
		reader[4][i]=8
	elif(reader[4][i]=='24-26'):
		reader[4][i]=9
	elif(reader[4][i]=='27-29'):
		reader[4][i]=10
	elif(reader[4][i]=='30-32'):
		reader[4][i]=11
	elif(reader[4][i]=='33-35'):
		reader[4][i]=12
	elif(reader[4][i]=='36-38'):
		reader[4][i]=13
for i in range(0,len(reader[5])):
	if(reader[5][i]=='yes'):
		reader[5][i]=1
	else:
		reader[5][i]=2

for i in range(0,len(reader[6])):

	reader[6][i]=reader[6][i]

for i in range(0,len(reader[7])):
	if(reader[7][i]=='left'):
		reader[7][i]=1
	else:
		reader[7][i]=2

for i in  range(0,len(reader[8])):
	if(reader[8][i]=='left_up'):
		reader[8][i]=1
	if(reader[8][i]=='left_low'):
		reader[8][i]=2
	if(reader[8][i]=='right_up'):
		reader[8][i]=3
	if(reader[8][i]=='right_low'):
		reader[8][i]=4
	if(reader[8][i]=='central'):
		reader[8][i]=5
for i in range(0,len(reader[9])):
	if(reader[9][i]=='yes'):
		reader[9][i]=1
	else:
		reader[9][i]=2
#print reader[1]
reader.to_csv('preprocessed.csv', sep='\t')
reader=reader.values
print reader.shape
count=0
for i in range(0,286):
	for j in range(0,10):
		if(reader[i][j]=='?' or reader[i][j]=='unknown'):
			reader[i][j]=1
for i in range(0,286):
	for j in range(0,10):
		reader[i][j]=float(reader[i][j])
#for i in range(0,286):
#	for j in range(0,10):
#		print type(reader[i][j])
X, y = reader[:, :-1], reader[:, -1]
for i in range(0,286):
	y[i]=str(y[i])
print X.shape,y.shape
X.reshape(286,-1)
y.reshape(286,1)
print X.shape,y.shape
for i in range(0,286):
	print type(y[i])
np.asarray(y, dtype=float, order=None)
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn import linear_model, datasets
logreg = linear_model.LogisticRegression(C=1e5)

clf = LinearSVC()
x_test=[]
y_test=[]
x_train=[]
y_train=[]
for i in range(0,160):
	x_train.append(X[i])
for i in range(0,160):
	y_train.append(y[i])
for i in range(160,180):
	x_test.append(X[i])
for i in range(160,180):
	y_test.append(y[i])
for i in range(180,280):
	x_train.append(X[i])
for i in range(180,280):
	y_train.append(y[i])
for i in range(280,286):
	x_test.append(X[i])
for i in range(280,286):
	y_test.append(y[i])
logreg.fit(x_train, y_train)
#print x_train
#print y_train
#print x_test
print y_test
print "Training complete"
#print x_test[0]
for i in range(0,len(x_test)):
	c=np.array(x_test[i])
	d=np.array([c])
	
	#d=d.reshape(1,-1)
#	print d.shape
	y_predict=logreg.predict(d)
	 #y_predict[0]
	if y_predict==y_test[i]:
		print "matched"
	else:
		print "not matched"
#print X_plot.shape
#print count


#for x in np.nditer(reader,flags=["REFS_OK"]):
#	print x


