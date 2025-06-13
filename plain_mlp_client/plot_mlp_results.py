import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning_params import CONSTRAINED


def plot_history(csv_path, output_dir, loss_fname, acc_fname=None):
    df = pd.read_csv(csv_path)
    epochs = df.index + 1
    os.makedirs(output_dir, exist_ok=True)

    # --- Loss ---
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, df['loss'], label='Training Loss')
    if 'val_loss' in df.columns:
        plt.plot(epochs, df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=24)
    plt.grid(True)
    loss_path = os.path.join(output_dir, loss_fname)
    plt.savefig(loss_path)
    print(f"Saved loss plot to {loss_path}")
    plt.close()

    # --- Accuracy ---
    if acc_fname and 'accuracy' in df.columns:
        plt.figure(figsize=(10, 7))
        plt.plot(epochs, df['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in df.columns:
            plt.plot(epochs, df['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch', fontsize=24)
        plt.ylabel('Accuracy', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(fontsize=24)
        plt.grid(True)
        acc_path = os.path.join(output_dir, acc_fname)
        plt.savefig(acc_path)
        print(f"Saved accuracy plot to {acc_path}")
        plt.close()

def main(constrainted_model):
    base_dir = 'plain_mlp_client/plain_mlp_model/'
    if constrainted_model:
        csv_file       = 'constraint_mlp_model_history.csv'
        loss_output    = 'mlp_model_loss_plot_constraint.png'
        acc_output     = 'mlp_model_accuracy_plot_constraint.png'
    else:
        csv_file       = 'mlp_model_history.csv'
        loss_output    = 'mlp_model_loss_plot.png'
        acc_output     = 'mlp_model_accuracy_plot.png'

    csv_path = os.path.join(base_dir, csv_file)
    plot_history(csv_path, base_dir, loss_output, acc_output)

if __name__ == '__main__':
    main(CONSTRAINED)