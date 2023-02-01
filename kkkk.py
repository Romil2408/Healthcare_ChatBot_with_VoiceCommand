import pandas
import pandas as pd

df = pd.read_csv('doctors_dataset.csv')
def remove_space_between_word(dataset):
        for i in range(len(dataset['Prognosis'])):
            if (type(dataset['Prognosis'][i]) == str ):
                dataset['Prognosis'][i] = dataset['Prognosis'][i].strip()
                dataset['Prognosis'][i] = dataset['Prognosis'][i].replace(" ", "_")
        return dataset

dd = remove_space_between_word(df)
print(dd)
dd.to_csv("Doctor.csv")