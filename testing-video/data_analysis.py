import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where this script resides
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(current_dir, "people_count.csv")

# Verify current directory and file path (for debugging purposes)
print("Current working directory:", os.getcwd())
print("Using CSV file:", file_name)

# Read the CSV file
df = pd.read_csv(file_name)

# Convert the 'timestamp' column to datetime objects
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Compute occupancy based on cumulative sum:
df["computed_current"] = df["entered"].cumsum() - df["exited"].cumsum()

# Calculate net flow (change in occupancy per interval)
df["flow_rate"] = df["current"].diff().fillna(0)

# -------------------------------------------
# Plot 1: Occupancy over Time with Rolling Average
# -------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(
    df["timestamp"], df["current"], marker="o", linestyle="-", label="Current Occupancy"
)
df["rolling_current"] = df["current"].rolling(window=5, min_periods=1).mean()
plt.plot(
    df["timestamp"], df["rolling_current"], linestyle="--", label="Rolling Average"
)
plt.xlabel("Timestamp")
plt.ylabel("Number of People")
plt.title("Room Occupancy Over Time")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save the plot as a PNG file in the same directory
occupancy_plot_path = os.path.join(current_dir, "occupancy_over_time.png")
plt.savefig(occupancy_plot_path)
plt.show()

# -------------------------------------------
# Plot 2: Cumulative Entries and Exits
# -------------------------------------------
plt.figure(figsize=(12, 6))
df["cumulative_entered"] = df["entered"].cumsum()
df["cumulative_exited"] = df["exited"].cumsum()
plt.plot(df["timestamp"], df["cumulative_entered"], label="Cumulative Entered")
plt.plot(df["timestamp"], df["cumulative_exited"], label="Cumulative Exited")
plt.xlabel("Timestamp")
plt.ylabel("Cumulative Count")
plt.title("Cumulative Entries and Exits Over Time")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save the plot as a PNG file in the same directory
cumulative_plot_path = os.path.join(current_dir, "cumulative_entries_exits.png")
plt.savefig(cumulative_plot_path)
plt.show()

# -------------------------------------------
# Plot 3: Net Flow Rate
# -------------------------------------------
plt.figure(figsize=(12, 6))
plt.bar(df["timestamp"], df["flow_rate"], width=0.03, color="skyblue")
plt.xlabel("Timestamp")
plt.ylabel("Net Flow (Change in Occupancy)")
plt.title("Net Flow Rate Over Time")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
# Save the plot as a PNG file in the same directory
netflow_plot_path = os.path.join(current_dir, "net_flow_rate.png")
plt.savefig(netflow_plot_path)
plt.show()
