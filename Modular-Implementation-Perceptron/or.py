from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd


def main(data, modelName, plotName, eta, epochs):
    # Convert input dictionary to pandas DataFrame
    df_OR = pd.DataFrame(data)
    # Split DataFrame into features (X) and target (y)
    X, y = prepare_data(df_OR)

    # Initialize Perceptron model with given learning rate and epochs
    model = Perceptron(eta=eta, epochs=epochs)
    # Train the model on the OR data
    model.fit(X, y)

    # Calculate and print the total loss after training
    _ = model.total_loss()

    # Save the trained model to disk
    model.save(filename=modelName, model_dir="model")
    # Save a plot of the data and decision boundary
    save_plot(df_OR, model, filename=plotName)


if __name__ == "__main__":
    # Define the OR logic gate dataset
    OR = {
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1],
        "y": [0, 1, 1, 1],
    }
    ETA = 0.3  # Learning rate for perceptron
    EPOCHS = 10  # Number of training epochs
    # Run the main function with OR data and parameters
    main(data=OR, modelName="or.model", plotName="or.png", eta=ETA, epochs=EPOCHS)