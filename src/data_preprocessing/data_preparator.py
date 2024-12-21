import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as spp
import sklearn.manifold as smf


def load(path:str, delimiter:str) -> pd.DataFrame:
    wpbc_dataframe = pd.read_csv(
        path,
        delimiter=delimiter,
        header=0
    )
    return wpbc_dataframe

def save_as_csv(dataframe:pd.DataFrame, file_path:str, sep:str) -> None:
    dataframe.to_csv(
        path_or_buf=file_path,
        sep=sep,
        columns=dataframe.columns,
        header=True,
        index=False
    )

def encode(dataframe:pd.DataFrame, columns_to_encode:list) -> pd.DataFrame:
    encoder=spp.LabelEncoder()
    encoded_dataframe=dataframe
    for column in columns_to_encode:
        encoded_dataframe[column]=encoder.fit_transform(
            y=encoded_dataframe[column]
        )
    return encoded_dataframe

def replace_nan(dataframe:pd.DataFrame, to_replace:str, value) -> pd.DataFrame:
    replaced_dataframe = dataframe
    replaced_dataframe = replaced_dataframe.replace(
        to_replace=to_replace,
        value=value
    )
    return replaced_dataframe

def separate(dataframe:pd.DataFrame, id_label:str, outcome_label:str, feature_start:int, feature_end:int) -> tuple:
    return (
        dataframe.loc[:,id_label],
        dataframe.loc[:,outcome_label],
        dataframe.iloc[:,feature_start:feature_end]
    )

def reduce(features:pd.DataFrame, n_components:int) -> pd.DataFrame:
    tsne = smf.TSNE(
        n_components=n_components,
        method='exact'
    )
    reduced_features = tsne.fit_transform(
        X=features
    )
    return pd.DataFrame(reduced_features)

def merge_dataframes(dataframe_1:pd.DataFrame, dataframe_2:pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        left=dataframe_1, 
        right=dataframe_2,
        left_index=True,
        right_index=True
    )

def normalize(features:pd.DataFrame) -> pd.DataFrame:
    """use spp.MinMaxScaler to normalize features to a 0-1 format"""
    scaler = spp.MinMaxScaler(feature_range=(0,1))
    scaled_dataframe = scaler.fit_transform(
        X=features
    )
    return pd.DataFrame(scaled_dataframe)


