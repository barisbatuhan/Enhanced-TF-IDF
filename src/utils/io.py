from typing import List, Union

import numpy as np 
import pandas as pd

class IO:
    """
    Description: A collection of basic data reading and writing operations.
    """
    def read_txt_corpus(filepath :str):
        """
        Description: Reads the corpus '.txt' file that includes a 
                     document at each line.
        
        Inputs:
            filepath (string) : Path of the corpus .txt
        
        Outputs:
            lines (List[str]) : List of documents that are in string format.
        """
        assert ".txt" in filepath

        with open(filepath, "r") as f:
            lines = f.readlines()
        return lines

    
    def save_to_csv(
        data     :np.ndarray, 
        filepath :str, 
        rownames :List[Union[str, int, float]]=None, 
        colnames :List[Union[str, int, float]]=None):
    
        """
        Description: Saves the 2D np.array data to a csv file.

        Inputs:
            data (np.array, 2D) : data to save
            filepath (str)      : file path to save the data
            rownames (List[Union[str, int, float]]) : name of the rows to save in CSV
            colnames (List[Union[str, int, float]]) : name of the cols to save in CSV
        """

        assert rownames is None or data.shape[0] == len(rownames)
        assert colnames is None or data.shape[1] == len(colnames)
        assert len(data.shape) == 2, "Given data should be a 2D numpy array"

        if rownames is None:
            rownames = np.arange(data.shape[0])
        if colnames is None:
            colnames = np.arange(data.shape[1])

        df = pd.DataFrame(data=data, index=rownames, columns=colnames)
        df.to_csv(filepath)