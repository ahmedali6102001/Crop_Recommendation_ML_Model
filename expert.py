
# Import Liberaries
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/ahmedali6102001/Crop_Recommendation_DS/main/Crop_recommendation.csv"
dataset = read_csv(url)

#Data Description...
print(dataset.shape)
print(dataset.head(10))
print(dataset.describe())
print(dataset.groupby('label').size())
print(dataset['N'].std())
dataset.hist(figsize=(10,10))

# Split-out validation dataset
cols=dataset.shape[1]
x = dataset.values[:,:cols-1]
y = dataset.values[:,cols-1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)

# Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results,names=[],[]
for n,m in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
    res=cross_val_score(m,x,y,cv=kfold,scoring='accuracy')
    results.append(res)
    names.append(n)
    print('%s: %f (%f)'%(n,res.mean(),res.std()))


# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')

#Making Prediction
model=GaussianNB()
model.fit(xtrain,ytrain)
prediction=model.predict(xtest)

#Evaluating Prediction
print(accuracy_score(ytest,prediction))
print(confusion_matrix(ytest,prediction))
print(classification_report(ytest,prediction))

#User Prediction
N=float(input('Enter the value of N: '))
P=float(input('Enter the value of P: '))
K=float(input('Enter the value of K: '))
temperature=float(input('Enter the temperature: '))
humidity=float(input('Enter the humidity: '))
ph=float(input('Enter the ph: '))
rainfall=float(input('Enter the rainfall: '))
values=[[N,P,K,temperature,humidity,ph,rainfall]]
print('the predicted value of entered data is',model.predict(values))


  