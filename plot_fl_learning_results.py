import os
import pandas as pd
import matplotlib.pyplot as plt

from learning_params import NUM_ROUNDS, NUM_CLIENTS, METHODS, CONSTRAINED


def plot_fl_results(rounds, clients, method, constraint=False):
    """
    Plot federated learning results from a CSV file.

    Args:
        csv_path: Path to the CSV file containing federated learning metrics
        output_dir: Directory to save the plots
        loss_fname: Filename for the loss plot
        acc_fname: Filename for the accuracy plot
    """
    base_dir = 'federated_learning_results'
    if constraint:
        base_dir = 'federated_learning_results/constrained'
    result_dir = os.path.join(base_dir, method, f"{rounds}-rounds", f"{clients}-clients")
    csv_path = os.path.join(result_dir, f"fl_run_metrics_num_clients-{clients}.csv")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    loss_fname = f'fl_loss_clients-{clients}.png'
    acc_fname = f'fl_accuracy_clients-{clients}.png'
    # Get the number of clients
    num_clients = df['num_clients'].iloc[0] if 'num_clients' in df.columns else "Unknown"
    # Save plots in the same directory as the CSV file
    plots_dir = result_dir  # Using the same directory for simplicity
    os.makedirs(plots_dir, exist_ok=True)
    # Plot loss
    plt.figure(figsize=(10, 7))
    plt.plot(df['round'], df['loss'], marker='o')
    plt.xlabel('Round', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    loss_path = os.path.join(result_dir, loss_fname)
    plt.savefig(loss_path)
    print(f"Saved loss plot to {loss_path}")
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 7))
    plt.plot(df['round'], df['accuracy'], marker='o', color='green')
    plt.xlabel('Round', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    acc_path = os.path.join(plots_dir, acc_fname)
    plt.savefig(acc_path)
    print(f"Saved accuracy plot to {acc_path}")
    plt.close()


if __name__ == "__main__":
    plain = METHODS[0]
    ckks = METHODS[1]
    ho = METHODS[2]
    rounds = [2,5,10,15,20,25,50,100]
    clients = [3]
    for method in METHODS:
        for round in rounds:
            for num_clients in clients:
                plot_fl_results(rounds=round, clients=num_clients, method=method, constraint=CONSTRAINED)