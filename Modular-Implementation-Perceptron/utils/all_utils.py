import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Function to split DataFrame into features (X) and target (y)
def prepare_data(df, target_col="y"):
    X = df.drop(target_col, axis=1)  # Drop the target column to get features
    y = df[target_col]               # Extract the target column
    return X, y 

# Function to save a plot showing data points and decision regions
def save_plot(df, model, filename="plot.png", plot_dir="plots"):
    # Helper function to create the base scatter plot
    def _create_base_plot(df):
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="coolwarm")  # Scatter plot of data
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)            # Horizontal axis line
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)            # Vertical axis line
        figure = plt.gcf()
        figure.set_size_inches(10, 8)                                           # Set plot size
    
    # Helper function to plot decision regions for a classifier
    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        colors = ("cyan", "lightgreen")                                         # Colors for regions
        cmap = ListedColormap(colors)
        X = X.values # Convert DataFrame to numpy array
        x1 = X[:, 0]
        x2 = X[:, 1]
        # Define grid for plotting
        x1_min, x1_max = x1.min() - 1, x1.max() + 1 
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution)
                              )
        # Predict class for each point in the grid
        y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)
        # Plot decision regions
        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.plot()
        
    X, y = prepare_data(df)              # Prepare features and target
    _create_base_plot(df)                # Plot data points
    _plot_decision_regions(X, y, model)  # Plot decision regions
    os.makedirs(plot_dir, exist_ok=True) # Create directory if it doesn't exist
    plot_path = os.path.join(plot_dir, filename) # Full path for saving plot
    plt.savefig(plot_path)               # Save the plot to file