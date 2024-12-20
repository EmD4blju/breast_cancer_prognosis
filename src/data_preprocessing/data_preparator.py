import pandas as pd

def load(path:str) -> pd.DataFrame:
    pass

def encode(dataframe:pd.DataFrame, columns_to_encode:list) -> pd.DataFrame:
    pass

def replace_nan(dataframe:pd.DataFrame, to_replace:str, value, columns:list) -> pd.DataFrame:
    pass

def separate(dataframe:pd.DataFrame, id_label:str, outcome_label:str, feature_start_label:int, after:bool) -> tuple:
    pass

def prepare_wbpc(dataframe:pd.DataFrame) -> pd.DataFrame:
    pass
