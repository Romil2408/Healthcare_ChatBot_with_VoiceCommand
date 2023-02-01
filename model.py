import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from joblib import dump



data = pd.read_csv("dataset.csv")
data_sevrity = pd.read_csv("Symptom-severity.csv")

def remove_space_between_word(dataset):
    for col in dataset.columns:
        for i in range(len(dataset[col])):
            if (type(dataset[col][i]) == str ):
                dataset[col][i] = dataset[col][i].strip()
                dataset[col][i] = dataset[col][i].replace(" ", "_")
    return data

new_df = remove_space_between_word(data)

def enc(dataset):
    for ind in data_sevrity.index:
        dataset = dataset.replace(data_sevrity["Symptom"][ind] , data_sevrity["weight"][ind])
    dataset = dataset.fillna(0) # put empty cell to 0
    dataset = dataset.astype({"Symptom_6":"int64","Symptom_7":"int64","Symptom_8":"int64","Symptom_9":"int64","Symptom_10":"int64","Symptom_11":"int64","Symptom_12":"int64","Symptom_13":"int64","Symptom_14":"int64","Symptom_15":"int64","Symptom_16":"int64","Symptom_17":"int64"})
    dataset = dataset.replace("foul_smell_of_urine" , 5)
    dataset = dataset.replace("dischromic__patches" , 6)
    dataset = dataset.replace("spotting__urination" , 6)
    return dataset

df = enc(new_df)

df_data = df.drop('Disease' , axis =1)
label = data["Disease"]

x_train, x_test, y_train, y_test = train_test_split(df_data, label, shuffle=True, train_size = 0.70)
randomFC = RandomForestClassifier()
randomFC.fit(x_train, y_train)
result = randomFC.predict(x_test)
print(randomFC)
print(classification_report(y_true=y_test, y_pred=result))
print('F1-score% =', f1_score(y_test, result, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, result)*100)

dump(randomFC,"model.pkl")