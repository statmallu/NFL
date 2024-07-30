import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import seaborn as sns

class GhostingExperiment:
    def __init__(self, csv_path):
        self.inputdata = pd.read_csv(csv_path)
        self.attack_col = ['x_Attack_01','y_Attack_01','s_Attack_01','dbf_Attack_01',
                           'x_Attack_02','y_Attack_02','s_Attack_02','dbf_Attack_02',
                           'x_Attack_03','y_Attack_03','s_Attack_03','dbf_Attack_03',
                           'x_Attack_04','y_Attack_04','s_Attack_04','dbf_Attack_04',
                           'x_Attack_05','y_Attack_05','s_Attack_05','dbf_Attack_05',
                           'x_Attack_06','y_Attack_06','s_Attack_06','dbf_Attack_06', 
                           'x_Attack_07','y_Attack_07','s_Attack_07','dbf_Attack_07',
                           'x_Attack_08','y_Attack_08','s_Attack_08','dbf_Attack_08',
                           'x_Attack_09','y_Attack_09','s_Attack_09','dbf_Attack_09',
                           'x_Attack_10','y_Attack_10','s_Attack_10','dbf_Attack_10',
                           'x_Attack_11','y_Attack_11','s_Attack_11','dbf_Attack_11']
        self.defense_col = ['x_Defense_01','y_Defense_01','s_Defense_01','dbf_Defense_01',
                            'x_Defense_02','y_Defense_02','s_Defense_02','dbf_Defense_02',
                            'x_Defense_03','y_Defense_03','s_Defense_03','dbf_Defense_03',
                            'x_Defense_04','y_Defense_04','s_Defense_04','dbf_Defense_04',
                            'x_Defense_05','y_Defense_05','s_Defense_05','dbf_Defense_05',
                            'x_Defense_06','y_Defense_06','s_Defense_06','dbf_Defense_06', 
                            'x_Defense_07','y_Defense_07','s_Defense_07','dbf_Defense_07',
                            'x_Defense_08','y_Defense_08','s_Defense_08','dbf_Defense_08',
                            'x_Defense_09','y_Defense_09','s_Defense_09','dbf_Defense_09',
                            'x_Defense_10','y_Defense_10','s_Defense_10','dbf_Defense_10',
                            'x_Defense_11','y_Defense_11','s_Defense_11','dbf_Defense_11']
        self.in_len = 2
        self.out_len = 30
        self.callback = EarlyStopping(monitor='loss', patience=3)
        self._prepare_data()

    def _prepare_data(self):
        self.inputdata = self.inputdata[['time','gameId','playId'] + self.attack_col + self.defense_col]
        self.inputdata.dropna(axis=0, inplace=True)

    def data_proc(self, data, seq_ln, forecast_ln):
        X, Y = [], []
        start, end, forecast_end = 0, seq_ln, seq_ln + forecast_ln
        for _ in range(data.shape[0] - (seq_ln + forecast_ln)):
            values = data.values
            X.append(values[start:end].tolist())
            Y.append(values[end:forecast_end].tolist())
            start, end, forecast_end = start + 1, end + 1, forecast_end + 1
        return X, Y

    def generate_sequences(self, cols):
        past, future = [], []
        for gameid in self.inputdata.gameId.unique().tolist():
            for playid in self.inputdata.playId.unique().tolist():
                sequence = self.inputdata[(self.inputdata['gameId'] == gameid) & (self.inputdata['playId'] == playid)][cols]
                x, y = self.data_proc(sequence, self.in_len, self.out_len)
                past.append(x)
                future.append(y)
        return self.flatten(past), self.flatten(future)

    @staticmethod
    def flatten(l):
        return [item for sublist in l for item in sublist]

    def prepare_training_data(self):
        offense_past, offense_next = self.generate_sequences(self.attack_col)
        defense_past, defense_next = self.generate_sequences(self.defense_col)

        offense_past = np.array(offense_past)
        offense_next = np.array(offense_next)
        defense_past = np.array(defense_past)
        defense_next = np.array(defense_next)

        combined_past = np.ones((offense_past.shape[0], offense_past.shape[1], offense_past.shape[2] + defense_past.shape[2]))
        for c in range(offense_past.shape[0]):
            for t in range(offense_past.shape[1]):
                combined_past[c][t] = np.append(offense_past[c][t], defense_past[c][t])

        defence_target = np.ones((defense_next.shape[0], defense_next.shape[1], 22))
        a = pd.Series(np.arange(0, 44, 4)).append(pd.Series(np.arange(1, 44, 4))).tolist()
        a.sort()
        for c in range(defense_next.shape[0]):
            for t in range(defense_next.shape[1]):
                defence_target[c][t] = defense_next[c][t][a]

        return combined_past, defence_target

    def split_data(self, combined_past, defence_target):
        test_len = int(0.9 * combined_past.shape[0])
        x_train_val, x_test = combined_past[:test_len], combined_past[test_len:]
        y_train_val, y_test = defence_target[:test_len], defence_target[test_len:]

        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, random_state=0, train_size=0.8)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def build_model(self, input_shape, output_shape):
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=input_shape))
        model.add(RepeatVector(output_shape[0]))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(output_shape[1])))
        model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))
        model.summary()
        return model

    def train_model(self, model, x_train, y_train, x_val, y_val):
        history = model.fit(x_train, y_train, batch_size=10, epochs=100, validation_data=(x_val, y_val), callbacks=[self.callback])
        return history

    def plot_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'model loss\n{self.in_len}_{self.out_len}')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.show()

    def evaluate_model(self, model, x_test, y_test, x_train, y_train, x_val, y_val):
        predictions = model.predict(x_test)
        print("R-Square value from train: ", r2_score(self.flatten(y_train), self.flatten(model.predict(x_train))))
        print("max deviation train: ", max(np.subtract(self.flatten(y_train), self.flatten(model.predict(x_train)))))
        print("R-Square value from val: ", r2_score(self.flatten(y_val), self.flatten(model.predict(x_val))))
        print("max deviation val: ", max(np.subtract(self.flatten(y_val), self.flatten(model.predict(x_val)))))
        print("R-Square value from test: ", r2_score(self.flatten(y_test), self.flatten(predictions)))
        print("max deviation test: ", max(np.subtract(self.flatten(y_test), self.flatten(predictions))))
        return predictions

    def plot_predictions(self, predictions, y_test):
        n = 1
        fig, ax = plt.subplots(figsize=(10, 6))
        for t in range(30):
            for player in range(11):
                ax.scatter(predictions[n][t][player], predictions[n][t][player + 1], c='y', s=100)
                ax.scatter(y_test[n][t][player], y_test[n][t][player + 1], c='b', s=100)
        plt.show()

# Example of using the class

csv_path = '../data/processed/inputdata.csv'

# Initialize the GhostingExperiment class
experiment = GhostingExperiment(csv_path)

# Prepare training data
combined_past, defence_target = experiment.prepare_training_data()

# Split data into training, validation, and test sets
x_train, x_val, x_test, y_train, y_val, y_test = experiment.split_data(combined_past, defence_target)

# Build the LSTM model
model = experiment.build_model(input_shape=(x_train.shape[1], x_train.shape[2]), 
                               output_shape=(y_train.shape[1], y_train.shape[2]))

# Train the model
history = experiment.train_model(model, x_train, y_train, x_val, y_val)

# Plot the training loss
experiment.plot_loss(history)

# Evaluate the model and make predictions
predictions = experiment.evaluate_model(model, x_test, y_test, x_train, y_train, x_val, y_val)

# Plot the predictions
experiment.plot_predictions(predictions, y_test)
