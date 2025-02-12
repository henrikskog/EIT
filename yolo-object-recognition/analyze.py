import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def load_and_prepare_data():
    # Read the CSV file
    df = pd.read_csv('data.csv')
    
    # Convert datetime string to datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    return df

def plot_people_over_time(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['num_people'], '-o')
    plt.title('Number of People Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of People')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('people_over_time.png')
    plt.close()

def plot_occupancy_heatmap(df):
    # Resample data to 1-minute intervals and get mean occupancy
    df_resampled = df.resample('1T').mean()
    
    # Create hour and minute columns
    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['minute'] = df_resampled.index.minute
    
    # Create pivot table for heatmap
    pivot_table = df_resampled.pivot_table(
        values='num_people', 
        index='hour',
        columns='minute',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_table, cmap='YlOrRd', cbar_kws={'label': 'Average Number of People'})
    plt.title('Occupancy Heatmap by Hour and Minute')
    plt.xlabel('Minute')
    plt.ylabel('Hour')
    plt.tight_layout()
    plt.savefig('occupancy_heatmap.png')
    plt.close()

def calculate_statistics(df):
    stats = {
        'Total Duration': f"{(df.index[-1] - df.index[0]).total_seconds() / 60:.2f} minutes",
        'Average Occupancy': f"{df['num_people'].mean():.2f} people",
        'Maximum Occupancy': f"{df['num_people'].max()} people",
        'Minimum Occupancy': f"{df['num_people'].min()} people",
        'Number of Changes': len(df),
        'Most Common Count': f"{df['num_people'].mode().iloc[0]} people"
    }
    
    # Calculate time spent at each occupancy level
    occupancy_duration = df.groupby('num_people').size()
    total_duration = len(df)
    occupancy_percentage = (occupancy_duration / total_duration * 100).round(2)
    
    return stats, occupancy_percentage

def plot_occupancy_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='num_people', discrete=True)
    plt.title('Distribution of Occupancy Levels')
    plt.xlabel('Number of People')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('occupancy_distribution.png')
    plt.close()

def main():
    try:
        # Load the data
        df = load_and_prepare_data()
        
        # Generate visualizations
        plot_people_over_time(df)
        plot_occupancy_heatmap(df)
        plot_occupancy_distribution(df)
        
        # Calculate statistics
        stats, occupancy_percentage = calculate_statistics(df)
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n=== Time Spent at Each Occupancy Level ===")
        for people, percentage in occupancy_percentage.items():
            print(f"{people} people: {percentage}% of time")
        
        print("\nVisualizations have been saved as:")
        print("- people_over_time.png")
        print("- occupancy_heatmap.png")
        print("- occupancy_distribution.png")
        
    except FileNotFoundError:
        print("Error: data.csv not found. Please run the people counter first to generate data.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 