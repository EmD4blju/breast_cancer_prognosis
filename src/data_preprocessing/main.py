import data_preparator as dp
import numpy as np
if __name__ == "__main__":
    wpbc_dataframe = dp.load(
        path='dataset/wpbc.csv',
        delimiter=','
    )
    wpbc_dataframe['TIME'] = wpbc_dataframe['TIME'].astype(dtype=np.float64)
    wpbc_dataframe = dp.encode(
        dataframe=wpbc_dataframe,
        columns_to_encode=['OUTCOME']
    )
    
    wpbc_dataframe = dp.replace_nan(
        dataframe=wpbc_dataframe,
        to_replace='?',
        value=np.nan
    )
    
    wpbc_dataframe['LYMPH_NODE_STATUS'] = wpbc_dataframe['LYMPH_NODE_STATUS'].astype(dtype=np.float64)
    lymph_node_mean=wpbc_dataframe['LYMPH_NODE_STATUS'].mean(
        axis=0, 
        skipna=True
    )
    wpbc_dataframe = dp.replace_nan(
        dataframe=wpbc_dataframe,
        to_replace=np.nan,
        value=lymph_node_mean
    )
    
    separated_wpbc_dataframe = dp.separate(
        dataframe=wpbc_dataframe,
        id_label='ID',
        outcome_label='OUTCOME',
        feature_start=2,
        feature_end=wpbc_dataframe.columns.size
    )
    wpbc_ids = separated_wpbc_dataframe[0]
    wpbc_outcomes = separated_wpbc_dataframe[1]
    wpbc_features = separated_wpbc_dataframe[2]
    
    wpbc_ids_outcomes = dp.merge_dataframes(
        dataframe_1=wpbc_ids,
        dataframe_2=wpbc_outcomes
    )
    reduced_features_list = []
    for num in range(5,20,5):
        reduced_features_list.append(
            dp.reduce(
                features=wpbc_features,
                n_components=num
            )
        )
    reduced_dataframes_list = []
    for reduced_features in reduced_features_list:
        reduced_dataframes_list.append(
            dp.merge_dataframes(
                dataframe_1=wpbc_ids_outcomes,
                dataframe_2=reduced_features
            )
        )
    dataframes_to_save = [dp.merge_dataframes(
        dataframe_1=wpbc_ids_outcomes, 
        dataframe_2=wpbc_features
    )]
    dataframes_to_save.extend(reduced_dataframes_list)
    for dataframe in dataframes_to_save:
        dataframe_features = dataframe.iloc[:,2:]
        normalized_features = dp.normalize(dataframe_features)
        dataframe.iloc[:,2:] = normalized_features
    i = 0
    for dataframe in dataframes_to_save:
        dp.save_as_csv(
            dataframe=dataframe,
            file_path='dataset/wpbc' + str(i) + '.csv',
            sep=','
        )
        i += 1