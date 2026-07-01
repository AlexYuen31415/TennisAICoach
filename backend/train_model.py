import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load arrays processed from Step 1
X = np.load("X_sequences.npy")
y = np.load("y_labels.npy")

# Convert labels to categorical matrix formatting (One-Hot Encoding)
y = to_categorical(y, num_classes=4)

# Partition data split (80% training, 20% verification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Architecture Architecture Blueprint
model = Sequential([
    # LSTM Layers process dynamic sequence modifications across time horizons
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 132)), # 30 frames x 132 coordinates
    Dropout(0.2),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax') # Outputs percentage probability distribution across the 4 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Beginning model convergence routines...")
history = model.fit(
    X_train, y_train, 
    epochs=60, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# Export structural weights to server storage asset 
model.save("tennis_stroke_lstm.h5")
print("Model optimized and compiled as 'tennis_stroke_lstm.h5'")