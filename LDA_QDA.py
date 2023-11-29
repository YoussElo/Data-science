import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#####LDA#####
smkt = pd.read_csv('smarket.csv')

def DirectionToInt(t):
    if t == "Down":
        return 0
    else:
        return 1

smkt['Direction'] = smkt['Direction'].apply(DirectionToInt)

X_train = smkt[smkt['Year'] < 2005][['Lag1','Lag2']]
y_train = smkt[smkt['Year'] < 2005]['Direction']

X_test = smkt[smkt['Year'] >= 2005][['Lag1', 'Lag2']]
y_test = smkt[smkt['Year'] >= 2005]['Direction']

LDA = LinearDiscriminantAnalysis()
lda_mdl = LDA.fit(X_train, y_train)

print('\n','LDA Prior probabilities: \n', lda_mdl.priors_,'\n')
print('LDA Group means : \n', lda_mdl.means_[0],'\n', lda_mdl.means_[1], '\n')
print('LDA Group Coefficients:\n', lda_mdl.coef_, '\n')

lda_pred = lda_mdl.predict(X_test)

m = confusion_matrix(y_test, lda_pred).T
print('LDA Confusion Matrix')
print(m[0], '\n',m[1])
report = classification_report(y_test, lda_pred)
print(report)

#####QDA#####
QDA = QuadraticDiscriminantAnalysis()
qda_mdl = QDA.fit(X_train, y_train)

print('\nQDA Prior probabilities:', qda_mdl.priors_,'\n')
print('QDA Group means :', qda_mdl.means_[0],'\n' ,lda_mdl.means_[1], '\n')


qda_pred = qda_mdl.predict(X_test)
m2 = confusion_matrix(y_test, qda_pred).T
print('QDA Confusion Matrix')
print(m2[0],'\n',m2[1], '\n')


report2 = classification_report(y_test, qda_pred)
print(report2)
