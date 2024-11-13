import pandas as pd
import numpy as np
# id=10000
# dataset = pd.read_csv("/Users/abdullahhesham/Documents/PWC/idp/credit_risk_dataset_processed.csv")


# print(', '.join(dataset.columns))


# print(dataset)
# This function explains a certain ID Answer
def Explain(id,dataset):
    dataset['time_series'] = pd.to_datetime(dataset['time_series'], format='%Y-%m-%d')
    filtered_dataset_Targeted_id = dataset[dataset['id'] == id]
    seq_size=len(filtered_dataset_Targeted_id)
    # print("this is my filtered id")
    # print(filtered_dataset_Targeted_id)

    columns_to_exclude = ['id', 'time_series']  # Replace these with the actual column names you want to exclude

    # Create a list of column names excluding the specified ones
    filtered_columns = [col for col in dataset.columns if col not in columns_to_exclude]
    unique_ids_counts = dataset['id'].value_counts()
    ids_to_extract = unique_ids_counts[(unique_ids_counts == seq_size) & (unique_ids_counts.index != id)].index
    filtered_dataset = dataset[dataset['id'].isin(ids_to_extract)]
    filtered_dataset = filtered_dataset.sort_values(by=['id', 'time_series'])
    # print(filtered_dataset)

    unique_ids_list = filtered_dataset[filtered_dataset['Default Flag'] == 1]['id'].unique()
    accl=[]
    l=[]
    for selected_id in unique_ids_list:
        rows_for_id = filtered_dataset[filtered_dataset['id'] == selected_id]
        acc,ll=compare_two_sequences(filtered_dataset_Targeted_id,rows_for_id)
        accl.append(acc)
        l.append(ll)

        # print(results)
    # print("Explanation")
    # print(first_max_index(accl))
    # print(l[first_max_index(accl)])
    return InterpretFunction(l[first_max_index(accl)], filtered_columns)




#     This function compares two sequences
from sklearn.metrics import mean_squared_error

def first_max_index(lst):
    if not lst:
        # Return None for an empty list
        return None

    max_val = max(lst)
    first_index = lst.index(max_val)
    return first_index
def compare_two_sequences(seq1, seq2, threshold=1):
    # Extract columns for comparison, excluding 'id' and 'time_series'
    cols_to_compare = seq1.columns.difference(['id', 'time_series'])

    # Initialize a list to store RMSE for each column
    rmse_list = []

    # Loop over the selected columns
    for col in cols_to_compare:
        # Calculate RMSE for the current column
        rmse = np.sqrt(mean_squared_error(seq1[col], seq2[col]))
        rmse_list.append(rmse)

    # Create a binary list based on the threshold
    binary_list = [0 if rmse < threshold else 1 for rmse in rmse_list]

    # Calculate accuracy (1 - mean of binary list)
    accuracy = 1 - np.mean(binary_list)
    accuracy=accuracy*100

    return accuracy, binary_list



# this function collects sequences from the dataset with same length
def Collect_Sequences():
    sequences=[]
    return sequences

# this function will interpret the sequences differences
def InterpretFunction(l, column_names):
    res="These columns affected the evaluation : \n"
    for index, value in enumerate(l):
        if value == 1:
            res+=f" Column name : {column_names[index]} \n"
    return res



# print(Explain(id,dataset))
