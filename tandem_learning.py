import neuralfit as nf
import matplotlib.pyplot as plt
import numpy as np
import keras

# Define the dataset
x = np.linspace(0, 1, 100).reshape(-1,1)
y = (x > 0.5) * 1

# Create and compile the model
model = nf.Model(inputs=1, outputs=1)

# Evolve the model
for i in range(50):
    model.compile(optimizer='alpha', loss='mse', monitors=['size'])
    model.evolve(x, y, epochs=10)
    model = model.to_keras()
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=10)
    model = nf.from_keras(model)

# Get model predictions
y_hat = model.predict(x)

# Plot results
plt.plot(x, y, label='True', color='k', linestyle='--')
plt.plot(x, y_hat, label='Predicted',color='#52C560', linewidth=2)
plt.show()
