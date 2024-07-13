import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('.')) # to import src package
import src.data_preprocessing.load_data as load_data
import src.utils.constants as constants

# Attività celebrale classificabile in:
#preictal state  : periodo di tempo prima della crisi. Durata: 1 ora
#interictal state: periodo di tempo fra le crisi       Durata: almeno 4 ore prima dello stato preictale


# In order to overcome the problem of the imbalanced dataset, we selected the number of interictal segments to be equal to the available number of preictal segments during the training process. The interictal segments were selected at random from the overall interictal samples.

def merge_rows(n, dataframe):
    return pd.DataFrame(dataframe.values.reshape(1, -1), columns=sum([list(dataframe.columns) for _ in range(len(dataframe))], []))

def create_dataset(patient_ids):
    for id in patient_ids:
        patient = load_data.load_summary_from_file(id)
        return patient.make_dataset()


def split_group(group, n=1280):
    num_complete_groups = len(group) // n
    return [group.iloc[i*n:(i+1)*n] for i in range(num_complete_groups)]


def balance_dataset(dataset):
    dataset = dataset.get('arr_0')
    df = pd.DataFrame(dataset)
    grouped = df.groupby(df.columns[-1])
    split_groups = {cls: split_group(group) for cls, group in grouped}

    a = np.array(split_groups.get(0))
    interictal = pd.DataFrame(a)
    print(interictal.info)

    # for cls, groups in split_groups.items():
    #     print(f"Class {cls}:")
    #     print(len(groups))

    #n_min = min(len(split_groups[0]), len(split_groups[1]))

    #print(len(split_groups.get(0)))
    # list = split_groups.get(0)
    # print(list)
    # interictal = pd.DataFrame(list)
    # interictal = interictal.sample(n=n_min)
    #print("ok")

    # preictal = pd.DataFrame(split_groups.get(1))
    # frames = [interictal, preictal]

    #return pd.concat(frames)


    # print(type(split_groups))
    # print(type(split_groups.get(0)))

    # interictal_group = 

    # g = dataset.groupby(dataset.columns[-1])
    # for x, y in g:
    #     print(f'{x}\n{y}\n')

    # interictal = dataset[dataset[dataset.columns[-1]] == 0]
    # preictal = dataset[dataset[dataset.columns[-1]] == 1]

    # print(interictal, "\n")
    # print(preictal)

    #return

def main():
    create_dataset(['chb01'])
    dataset = np.load(f'{constants.DATASET_FOLDER}/chb01.npz')

    for k in dataset.keys():
        print(dataset.get(k).shape)

    # a = np.array(
    #     [[1, 2, 3],
    #      [4, 5, 6],
    #      [7, 8, 9],
    #      [10, 11, 12]]
    # )

    # print(a, "\n")

    # rng = np.random.default_rng()
    # print(rng.choice(a, size=2,replace=False))

    return

if __name__ == '__main__':
    main()