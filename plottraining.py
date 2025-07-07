import pandas as pd
import matplotlib.pyplot as plt

# Load monitor file
df = pd.read_csv("C:/Users/90546/Desktop/customenv/monitor.csv", skiprows=1)

# Plot episode reward vs cumulative timesteps
plt.plot(df['l'].cumsum(), df['r'])
plt.xlabel("Timesteps")
plt.ylabel("Episode reward")
plt.title("Training reward over time")
plt.show()
