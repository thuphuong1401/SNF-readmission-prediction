import numpy as np
import sklearn
import preprocessing
import icd_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Put all preprocessed columns into a big input matrix
X = np.stack((preprocessing.gender, preprocessing.length_of_stay, preprocessing.ethnic,
    preprocessing.white, preprocessing.black, preprocessing.other_race, preprocessing.lives_alone,
    preprocessing.asthma, preprocessing.afib, preprocessing.cad, preprocessing.chf,
    preprocessing.copd, preprocessing.diabetes, preprocessing.diabetes_by_dashbd, preprocessing.htn,
    preprocessing.obesity, preprocessing.ckd, preprocessing.cld, preprocessing.dprsn,
    preprocessing.ostpor, preprocessing.cl, preprocessing.lipid, preprocessing.left, preprocessing.right,
    preprocessing.both, preprocessing.unknown, preprocessing.hip, preprocessing.knee, preprocessing.anterior,
    preprocessing.age, preprocessing.provider_1, preprocessing.provider_2, preprocessing.provider_3,
    preprocessing.provider_4, preprocessing.provider_5, preprocessing.provider_6, preprocessing.provider_7,
    preprocessing.provider_other, preprocessing.reg_fsc_1, preprocessing.reg_fsc_2,
    preprocessing.reg_fsc_3, preprocessing.reg_fsc_4, preprocessing.reg_fsc_5, preprocessing.reg_fsc_other,
    preprocessing.height, preprocessing.weight, preprocessing.prev_snf_admission, icd_preprocessing.icd10_one,
    icd_preprocessing.icd10_two, icd_preprocessing.icd10_three, icd_preprocessing.icd10_four,
    icd_preprocessing.icd10_other), axis = 1)

#print(X.shape) # => (4310, 52)

X = X.astype(np.float64)

# Y is the output matrix
Y = preprocessing.snf
Y = Y.astype(np.float64)

# Split the data into train/validation/test set, with ratio 70/15/15
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
X_test1, X_val, Y_test1, Y_val = train_test_split(X_test, Y_test, test_size = 0.5, random_state = 1)
# print(X_val.shape)


# Logistic Regression
logistic_regression = LogisticRegression(solver = 'liblinear')
logistic_regression.fit(X_train, Y_train)
logistic_prediction = logistic_regression.predict(X_test1)

print("This is the performance result of the Logistic Regression model: ")
print("This is the confusion matrix")
print(confusion_matrix(Y_test1, logistic_prediction))
print(classification_report(Y_test1, logistic_prediction))


# SVM
svm = SVC(gamma = 'auto')
svm.fit(X_train, Y_train)
svm_prediction = svm.predict(X_test1)

print("This is the performance result of the SVM model:")
print("This is the confusion matrix")
print(confusion_matrix(Y_test1, svm_prediction))
print(classification_report(Y_test1, svm_prediction))


# Random Forest
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, Y_train)
rf_prediction = random_forest.predict(X_test1)

print("This is the result of the Random Forest model: ")
print("This is the confusion matrix")
print(confusion_matrix(Y_test1, rf_prediction))
print(classification_report(Y_test1, rf_prediction))
