import os
import json
from typing import List, Union, Any

import numpy as np 
import pandas as pd

class IO:
    """
    Description: A collection of basic data reading and writing operations.
    """
    
    def create_corpus_txt_from_json_list_file(inp_path :str, out_path :str, key :Any):
        """
        Description: Reads the corpus '.json' or '.jsonl' file that includes a list of 
                     dictionaries. Iterates through the dictionaries and selects the 
                     items that has the 'key'.
        
        Inputs:
            inp_path (string) : Path of the corpus .json or .jsonl file
            out_path (string) : Path to write the extracted corpus .txt
            key (Union[int, string, float]) : key to select from each json dict.
        """
        assert ".json" in inp_path or ".jsonl" in inp_path
        assert ".txt" in out_path

        try:
            with open(inp_path, "r") as f:
                lines = f.readlines()
            
            f_out = open(out_path, "w")
            for line in lines:
                d = json.loads(line)
                f_out.write(d[key].replace("\n", "").strip() + "\n")
            f_out.close()
        except:
            raise UnicodeDecodeError("File to read or write is not correct !") 
    
    
    def create_corpus_txt_from_json_file(inp_path :str, out_path :str, key :Union[int, str, float]):
        """
        Description: Reads the corpus '.json' file that includes a series of keys and values. Iterates through
                     the values that are dicts and selects the items that has the 'key'.

        Inputs:
            inp_path (string) : Path of the corpus .json file
            out_path (string) : Path to write the extracted corpus .txt
            key (Union[int, string, float]) : key to select from the json in the 2nd depth.
        """
        assert ".json" in inp_path
        assert ".txt" in out_path

        try:
            with open(inp_path, "r") as f:
                d = json.load(f)

            f_out = open(out_path, "w")

            for k in d.keys():
                for el in d[k]:
                    txt = el[k].replace("\n", "").strip()
                    f_out.write(txt + "\n")
            f_out.close()
        
        except:
            raise UnicodeDecodeError("File to read or write is not correct !") 

    
    def read_txt_corpus(filepath :str):
        """
        Description: Reads the corpus '.txt' file that includes a document at each line.
        
        Inputs:
            filepath (string) : Path of the corpus .txt
        
        Outputs:
            lines (List[str]) : List of documents that are in string format.
        """
        assert ".txt" in filepath

        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
            return lines
        except:
            raise UnicodeDecodeError("File to read is not correct !")
        

    
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
        assert len(filepath) > 4 and filepath[-4:] == ".csv", \
            "Filepath should have '.csv' extension !"
        assert rownames is None or data.shape[0] == len(rownames)
        assert colnames is None or data.shape[1] == len(colnames)
        assert len(data.shape) == 2, "Given data should be a 2D numpy array"

        if rownames is None:
            rownames = np.arange(data.shape[0])
        if colnames is None:
            colnames = np.arange(data.shape[1])

        df = pd.DataFrame(data=data, index=rownames, columns=colnames)
        df.to_csv(filepath)