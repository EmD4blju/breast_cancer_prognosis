import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as spp
import sklearn.manifold as smf


"""
data_preparator.py contains methods prepared for preprocessing wpbc dataset to my own need.
"""


def load(path:str, delimiter:str) -> pd.DataFrame:
    """Loads a .csv file to a dataframe

    Args:
        path (str): path of a file to load
        delimiter (str): .csv delimiter/separator

    Returns:
        pd.DataFrame: loaded dataframe
    """
    wpbc_dataframe = pd.read_csv(
        path,
        delimiter=delimiter,
        header=0
    )
    return wpbc_dataframe

def save_as_csv(dataframe:pd.DataFrame, file_path:str, sep:str) -> None:
    """Saves a dataframe to a .csv file

    Args:
        dataframe (pd.DataFrame): dataframe to save
        file_path (str): path to save the dataframe to
        sep (str): separator/delimiter for the .csv format
    """
    dataframe.to_csv(
        path_or_buf=file_path,
        sep=sep,
        columns=dataframe.columns,
        header=True,
        index=False
    )

def encode(dataframe:pd.DataFrame, columns_to_encode:list) -> pd.DataFrame:
    """Encodes the dataframe's specifed columns with a LabelEncoder

    Args:
        dataframe (pd.DataFrame): dataframe to encode
        columns_to_encode (list): columns of the dataframe to encode

    Returns:
        pd.DataFrame: dataframe with encoded columns
    """
    encoder=spp.LabelEncoder()
    encoded_dataframe=dataframe
    for column in columns_to_encode:
        encoded_dataframe[column]=encoder.fit_transform(
            y=encoded_dataframe[column]
        )
    return encoded_dataframe

def replace_nan(dataframe:pd.DataFrame, to_replace:str, value) -> pd.DataFrame:
    """Replaces values in the given dataframe

    Args:
        dataframe (pd.DataFrame): dataframe to replace values in
        to_replace (str): value to replace
        value (_type_): replacement value

    Returns:
        pd.DataFrame: modified dataframe with replacements
    """
    replaced_dataframe = dataframe
    replaced_dataframe = replaced_dataframe.replace(
        to_replace=to_replace,
        value=value
    )
    return replaced_dataframe

def separate(dataframe:pd.DataFrame, id_label:str, outcome_label:str, feature_start:int, feature_end:int) -> tuple:
    """Separates different columns  in a dataset and returns them separatly

    Args:
        dataframe (pd.DataFrame): dataframe to separate
        id_label (str): label of id column
        outcome_label (str): label of outcome categorical column
        feature_start (int): an index of a column where feature values start
        feature_end (int): an index of a column where feature values end

    Returns:
        tuple: tuple of two Series (ids, outcomes) and one feature dataframe
    """
    return (
        dataframe.loc[:,id_label],
        dataframe.loc[:,outcome_label],
        dataframe.iloc[:,feature_start:feature_end]
    )

def reduce(features:pd.DataFrame, n_components:int) -> pd.DataFrame:
    """Reduces the dimension in the dataframe according to specified number of components. Uses TSNE reduction.

    Args:
        features (pd.DataFrame): features dataframe to reduce
        n_components (int): number of dimension for the dataframe to be reduced to

    Returns:
        pd.DataFrame: reduced dataframe
    """
    tsne = smf.TSNE(
        n_components=n_components,
        method='exact'
    )
    reduced_features = tsne.fit_transform(
        X=features
    )
    return pd.DataFrame(reduced_features)

def merge_dataframes(dataframe_1:pd.DataFrame, dataframe_2:pd.DataFrame) -> pd.DataFrame:
    """Merges two specified dataframes

    Args:
        dataframe_1 (pd.DataFrame): dataframe to merge
        dataframe_2 (pd.DataFrame): dataframe to merge

    Returns:
        pd.DataFrame: merged dataframe of
    """
    return pd.merge(
        left=dataframe_1, 
        right=dataframe_2,
        left_index=True,
        right_index=True
    )

def normalize(features:pd.DataFrame) -> pd.DataFrame:
    """Normalizes the specified features with MinMaxScaler.

    Args:
        features (pd.DataFrame): features to normalize

    Returns:
        pd.DataFrame: normalized features
    """
    scaler = spp.MinMaxScaler(feature_range=(0,1))
    scaled_dataframe = scaler.fit_transform(
        X=features
    )
    return pd.DataFrame(scaled_dataframe)


