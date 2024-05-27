import read_data
import make_dataset as mk
import os
#from pathlib import Path
#import mne
#import pandas as pd

def main():
    patients = read_data.read_data()
    for p in patients:
        mk.make_dataset(p)
    return True

if __name__ == '__main__':
    main()