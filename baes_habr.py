from __future__ import division
from collections import defaultdict
from math import log
import csv

def train(samples):
    classes, freq = defaultdict(lambda:0), defaultdict(lambda:0)
    for feats, label in samples:
        classes[label] += 1                 # count classes frequencies
        for feat in feats:
            freq[label, feat] += 1          # count features frequencies

    for label, feat in freq:                # normalize features frequencies
        freq[label, feat] /= classes[label]
    for c in classes:                       # normalize classes frequencies
        classes[c] /= len(samples)

    return classes, freq                    # return P(C) and P(O|C)

def classify(classifier, feats):
    classes, prob = classifier
    return min(classes.keys(),              # calculate argmin(-log(C|O))
        key = lambda cl: -log(classes[cl]) + \
            sum(-log(prob.get((cl,feat), 10**(-7))) for feat in feats))

def getData():
	#Было: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	mf = open('train.csv','r')
	rdr = csv.reader(mf, delimiter=',',quotechar='"')
	data = []
	print (rdr.next())
	for row in rdr:
		if row[5]=='':
			row[5]='-1'
		sib = int(row[6])
		parch = int(row[7])
		family = 'notfamily'
		if (sib !=0 or parch!=0):
			family = 'family'
		data.append(('class'+row[2],row[4]+'_'+family, float(row[5]), int(row[1])))
	return data

def getTestData():
	fl = open('test.csv','rb')
	reader = csv.reader(fl, delimiter=',', quotechar='"')
	reader.next()
	result = []
	#Формируем по-другому массив
	#Было: PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	for row in reader:
		if row[4]=='':
			row[4] = '-1'
		sib = int(row[5])
		parch = int(row[6])
		family = 'notfamily'
		if (sib !=0 or parch!=0):
			family = 'family'
		result.append(('class'+row[1],row[3]+'_'+family, float(row[4]), row[0]))
	return result
def getFeatures(sample): return (sample[0],sample[1],sample[2]) # get last letter

data = getData()
features = [(getFeatures(sample), sample[-1]) for sample in data]

classifier = train(features)
resCsv = open("result.csv","w")
resCsv.write("PassengerId,Survived\n")
for testsample in getTestData():
	resCsv.write(testsample[-1])
	resCsv.write(',')
	clsfd = classify(classifier,testsample[:-1])
	print clsfd
	resCsv.write(str(clsfd))
	resCsv.write('\n')
resCsv.close()
	
#print 'gender: ', classify(classifier, get_features(u'Аглафья'))