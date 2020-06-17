# -*- coding: utf-8 -*-
from datetime import datetime
from csv import DictWriter
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import export_text
import os


#counterr = number of days in generated data
def Dat_Merger(count, monCount, dataString):
    print("-------RUNNING DATA MERGER------")
    month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
             'November', 'December']

    newFile = []
    times_missed = 0
    # arg1
    counterr = 0
    # arg2
    #monCount = 4
    # arg3
    #dataString = 'Summer_Data_Weekdays'
    dayCount = 0
    for y in range(1, 5):
        if (counterr > count):
            break
        for z in range(1, 30):
            if (counterr > count):
                break
            link1 = 'D:\\FYP\\Weather_Data\\2018\\' + str(monCount + 1) + '\\' + str(z) + ' ' + month[
                monCount] + ' 2018.csv'
            link2 = 'D:\\FYP\\Generated_Data\\' + dataString + '\\Day_' + str(dayCount) + '.csv'

            with open(link1) as f:
                a = [{k: v for k, v in row.items()}
                     for row in csv.DictReader(f, skipinitialspace=True)]

            with open(link2) as f:
                b = [{k: v for k, v in row.items()}
                     for row in csv.DictReader(f, skipinitialspace=True)]

            a.pop(0)

            # b is generated device state data
            # a is scraped weather data
            for i in range(0, len(b)):
                check = False
                time1 = datetime.strptime(b[i]['Time'], "%H:%M:%S")
                for j in range(0, len(a)):
                    time2 = datetime.strptime(a[j]['Time'], "%H:%M")
                    diff = (time1 - time2).total_seconds() / 60
                    if (abs(diff) < 60):
                        t2 = datetime(1900, 1, 1)
                        tt = (time1 - t2).total_seconds() / 60.0
                        check = True
                        # tt = b[i]['Time']
                        dd = b[i]['Device']
                        ss = b[i]['State']
                        tp = a[i]['Temp']
                        tp = tp.replace("°C", "")
                        ww = a[i]['Weather']
                        ww = ww.replace(".", "")
                        if "Rain" in ww or "rain" in ww:
                            ww = "Rain"
                        elif "Thunder" in ww:
                            ww = "Thunderstorms"
                        elif "cloud" in ww or "Cloud" in ww:
                            ww = "Clouds"
                        wi = a[i]['Wind']
                        wi = wi.replace(" km/h", "")
                        if "No wind" in wi or wi == "N/A":
                            wi = 0
                        bb = a[i]['Barometer']
                        bb = bb.replace("%", "")
                        if bb == "N/A":
                            bb = 65
                        vv = a[i]['Visibility']
                        vv = vv.replace(" mbar", "")
                        if vv == "N/A":
                            vv = 1006
                        d = {'Time': tt, 'Device': dd, 'State': ss, 'Temp': tp, 'Weather': ww, 'Wind': wi,
                             'Barometer': bb,
                             'Visibility': vv}
                        newFile.append(d)
                        break
                if check is False:
                    times_missed = times_missed + 1
                    print('Missed in ' + str(counterr) + ' ' + str(monCount) + ' ' + str(z))
                    t2 = datetime(1900, 1, 1)
                    tt = (time1 - t2).total_seconds() / 60.0
                    # tt = b[i]['Time']
                    dd = b[i]['Device']
                    ss = b[i]['State']
                    d = {'Time': tt, 'Device': dd, 'State': ss, 'Temp': tp, 'Weather': ww, 'Wind': wi, 'Barometer': bb,
                         'Visibility': vv}
                    newFile.append(d)

            # print(newFile)
            dayCount = dayCount + 1
            counterr = counterr + 1
        monCount = monCount + 1

    print('Times missed: ' + str(times_missed))
    #this contains merged data for a single weather and weekday/weekend type
    return newFile


def Dat_Separator(theDat):
    print("-------RUNNING DATA SEPARATOR------")

    nightLight = []
    light = []
    AC = []
    fan = []

    for i in range(0, len(theDat)):
        if (theDat[i]['Device'] == 'nightLight'):
            nightLight.append(theDat[i])

        elif (theDat[i]['Device'] == 'AC'):
            AC.append(theDat[i])

        elif (theDat[i]['Device'] == 'Fan'):
            fan.append(theDat[i])

        elif (theDat[i]['Device'] == 'Light'):
            light.append(theDat[i])
        else:
            print('Error!')

    return light, nightLight, fan, AC


###################

def Dat_Interpolator(theDat):
    newDat = []
    for i in range(0, len(theDat) - 1):
        newDat.append(theDat[i])
        tt = float(theDat[i]['Time'])
        dd = theDat[i]['Device']
        ss = theDat[i]['State']
        tp = theDat[i]['Temp']
        ww = theDat[i]['Weather']
        wi = theDat[i]['Wind']
        bb = theDat[i]['Barometer']
        vv = theDat[i]['Visibility']

        myBool = False
        diff = float(theDat[i + 1]['Time']) - float(theDat[i]['Time'])
        if (diff > 10):
            quo = diff / 10
        elif (diff < 0):
            quo = (1439 - float(theDat[i]['Time'])) + (float(theDat[i + 1]['Time']))
            quo = abs(quo) / 10

        for k in range(0, 9):
            if (tt + quo < 1440 and myBool is False):
                tt = tt + quo
            else:
                if (myBool is False):
                    quo1 = 1439 - tt
                    quo1 = quo - quo1
                    tt = quo1
                    myBool = True
                else:
                    tt = tt + quo

            d = {'Time': tt, 'Device': dd, 'State': ss, 'Temp': tp, 'Weather': ww, 'Wind': wi, 'Barometer': bb,
                 'Visibility': vv}
            newDat.append(d)

    return newDat



####################
def process_data(theDat):
    print("-------RUNNING PROCESS DATA------")
    print(theDat)
    theData = pd.DataFrame(theDat)
    # Printing the dataset shape
    print("Dataset Length: ", len(theData))
    print("Dataset Shape: ", theData.shape)

    #create dummies for categorical data
    dummy1 = pd.get_dummies(theData, columns=['State', 'Weather', 'Device'])
    for col in dummy1.columns:
        print(col)
    print(dummy1.shape)
    return dummy1


def splitData(theData):
    print("-------RUNNING SPLIT DATA------")

    #index 5,6 are ON/OFF columns
    #X = theData.values[:, [0,1,2, 3, 4, 7,8,9,10,11,12,13,14,15,16,17,18]]
    #Y = theData.values[:, 5]
    Y = theData[['State_ON']]
    del theData['State_ON']
    del theData['State_OFF']
    X = theData
    Y = Y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(Xtrain, Xtest, ytrain):
    print("-------RUNNING GINI TRAINER------")

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(Xtrain, ytrain)
    return clf_gini


# Function to perform training with entropy.
def tarin_using_entropy(Xtrain, Xtest, ytrain):
    print("-------RUNNING ENTROPY TRAINER------")

    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(Xtrain, ytrain)
    return clf_entropy


# Function to make predictions
def prediction(Xtest, clf_object):
    print("-------RUNNING PREDITCOR------")

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(Xtest)
    print("Predicted values:")
    print(y_pred)
    return y_pred


def cal_accuracy(ytest, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(ytest, y_pred))

    print("Accuracy : ",
          accuracy_score(ytest, y_pred) * 100)

    print("Report : ",
          classification_report(ytest, y_pred))


###################

def Dec_Tree(dat):
    if(dat == []):
        print("Not sufficient data")
    else:
        data = process_data(dat)
        # Building Phase
        X, Y, X_train, X_test, y_train, y_test = splitData(data)
        clf_gini = train_using_gini(X_train, X_test, y_train)
        #print("*************")
       # print(clf_gini)
        clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

        # Operational Phase
        print("Results Using Gini Index:")

        # Prediction using gini
        y_pred_gini = prediction(X_test, clf_gini)
        cal_accuracy(y_test, y_pred_gini)

        print("Results Using Entropy:")
        # Prediction using entropy
        y_pred_entropy = prediction(X_test, clf_entropy)
        cal_accuracy(y_test, y_pred_entropy)

        print("*******************************")
        r = export_text(clf_gini,feature_names=(list(X.columns)))
        print(r)


# Driver code
def main():
    s = 'Summer_Data_Weekdays'
    count = 78
    #1 less than actual month number
    mon = 5
    d = Dat_Merger(count,mon,s)
    l,n,f,a = Dat_Separator(d)
    feed = Dat_Interpolator(a)
    Dec_Tree(feed)

# Calling main function
if __name__ == "__main__":
    main()

