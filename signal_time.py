import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from collections import deque
import random

# Traffic Signal Controller with LSTM
class TrafficSignalController:
    def __init__(self, model):
        self.model = model
        self.past_vehicle_counts = deque(maxlen=4)  # Store last 4 vehicle counts
        self.green_time = 30  # Initial values
        self.red_time = 20
        self.yellow_time = 5

    def update_signal_timings(self):
        """Predict and update signal timings dynamically using LSTM."""
        if len(self.past_vehicle_counts) < 4:
            return
        
        input_seq = np.array(self.past_vehicle_counts).reshape(1, 4, 1)
        predicted_timings = self.model.predict(input_seq)[0]
        
        self.green_time = max(10, min(int(predicted_timings[0]), 60))
        self.red_time = max(10, min(int(predicted_timings[1]), 40))
        self.yellow_time = max(3, min(int(predicted_timings[2]), 10))

    def print_signal_timings(self):
        print(f"Green Time: {self.green_time} sec, Red Time: {self.red_time} sec, Yellow Time: {self.yellow_time} sec")

# Train LSTM Model
def train_lstm_model():
    """Train an LSTM model to predict best signal timings dynamically."""
    traffic_data = np.array([random.randint(5, 70) for _ in range(200)])
    
    X_train, y_train = [], []
    for i in range(len(traffic_data) - 4):
        X_train.append(traffic_data[i:i+4])
        
        green_time = min(max(traffic_data[i+4] * 0.8, 10), 60)
        red_time = min(max(traffic_data[i+4] * 0.5, 10), 40)
        yellow_time = min(max(traffic_data[i+4] * 0.2, 3), 10)
        
        y_train.append([green_time, red_time, yellow_time])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(4, 1)),
        LSTM(50, activation='relu'),
        Dense(3)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)
    
    return model

# Run Simulation
def run_simulation(model):
    controller = TrafficSignalController(model)
    
    for _ in range(20):
        random = random.randint(5, 70)  
        controller.past_vehicle_counts.append(random)
        
        controller.update_signal_timings()  # Predict and update timings
        print(f"Predicted Vehicle Count: {random}")
        controller.print_signal_timings()

if __name__ == "__main__":
    lstm_model = train_lstm_model()
    run_simulation(lstm_model)