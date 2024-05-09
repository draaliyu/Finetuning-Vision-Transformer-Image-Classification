# main.py
import pickle
from load_data import load_data
from processing import create_datasets
from finetune_model import build_model, compile_model, train_model
from plot_results import plot_history

DATASET_PATH = 'test'


def main():
    # Load data
    X_train, X_val, y_train, y_val, classes = load_data(DATASET_PATH)

    # Create datasets
    train_dataset, val_dataset = create_datasets(X_train, X_val, y_train, y_val, len(classes))

    # Build and compile the model
    model = build_model(len(classes))
    model = compile_model(model)

    # Train the model
    history = train_model(model, train_dataset, val_dataset)

    # Save history object to a file
    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Plot results
    plot_history(history)


if __name__ == '__main__':
    main()
