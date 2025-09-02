'''
# using a dataframe

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

data = pd.DataFrame({
   'A': [1, 1, 1],   # constant which means no variance
    'B': [1, 2, 3],   # increasing values which means it has varinace
    'C': [4, 4, 4],   # constant 
    'D': [5, 6, 7],   # increasing values
    'E': [0, 0, 0]    # constant 
})

print("Before threshold:\n",data)
selected=VarianceThreshold(threshold=0.1) #get values that fit the threshold
selected.fit(data)

selected_columns=data.columns[selected.get_support()]
after_dataset=data[selected_columns]
print("After threshold:\n",after_dataset)
'''
# using a dataset

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Load dataset
dataset = pd.read_csv(r"C:\Users\HASAN ENTERPRISES\Desktop\Coding\Pandas programming\Titanic-Dataset.csv.csv")

print("Before Threshold:\n", dataset)

# Select only numeric columns
numeric = dataset.select_dtypes(include='number')
print("\nNumeric Columns Preview:\n", numeric.head())

# Apply Variance Threshold
selector = VarianceThreshold(threshold=0.1)
selector.fit(numeric)

# Get column names that passed the threshold
selected_columns = numeric.columns[selector.get_support()]

# Create filtered dataset
after_dataset = numeric[selected_columns]
print("\nAfter Threshold:\n", after_dataset.head())
