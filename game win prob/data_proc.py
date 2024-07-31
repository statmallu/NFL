import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def print_na_percentage(play):
    for col in play.columns.tolist():
        na_percentage = play[col].isnull().sum() / len(play[col])
        print(f'{col}: {na_percentage:.2%}')

def drop_na(play):
    play.dropna(inplace=True)
    return play

def scale_columns(play, columns_to_scale):
    scaler = MinMaxScaler()
    play[columns_to_scale] = scaler.fit_transform(play[columns_to_scale])
    return play

def drop_unnecessary_columns(play, columns_to_drop):
    return play.drop(columns=columns_to_drop, axis=1)

def process_play_data(play, categorical_columns, columns_to_drop, columns_to_scale):
    play = convert_game_clock(play)
    play = convert_categorical_columns(play, categorical_columns)
    play = add_home_win_column(play)
    print_na_percentage(play)
    play = drop_na(play)
    play = scale_columns(play, columns_to_scale)
    play_final = drop_unnecessary_columns(play, columns_to_drop)
    return play_final

