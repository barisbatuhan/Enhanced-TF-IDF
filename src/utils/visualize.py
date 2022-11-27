import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """
    Description: Collection of basic visualization operations.
    """
    def vis_heatmap(data :np.ndarray, filepath :str, color :str="coolwarm"):
        """
        Description: Visualizes a heatmap from the given data.

        Inputs:
            data (np.ndarray) : 2D numpy array to visualize.
            filepath (string) : Path to save the figure.
            color (string)    : Color class of heatmap.
        """
        assert ".png" in filepath or ".jpg" in filepath, \
            "File path to visualize should be an image file !"
        assert len(data.shape) == 2, \
            "Given data should be a 2D numpy array"

        plt.imshow(data, cmap=color, aspect='auto')
        plt.savefig(filepath)
    
    def vis_closeness(data :np.ndarray, filepath :str, labels:list=None, axis :int=1, color :str="coolwarm"):
        """
        Description: Visualizes the similarities of feature words given the data.

        Inputs:
            data (np.ndarray) : 2D numpy array to visualize.
            filepath (string) : Path to save the figure.
            axis (int)        : Visualize either by the axis 0 or 1.
            color (string)    : Color class of heatmap.
        """
        assert axis in [0, 1], "Axis should be either 0 or 1 for 2D array !"
        assert ".png" in filepath or ".jpg" in filepath, \
            "File path to visualize should be an image file !"
        assert len(data.shape) == 2, \
            "Given data should be a 2D numpy array"

        if axis == 0:
            vecs = data
        else:
            vecs = data.T

        result = cosine_similarity(vecs, vecs)
        if labels is not None:
            result = pd.DataFrame(data=result, index=labels, columns=labels)
        sns.heatmap(result)
        plt.savefig(filepath)