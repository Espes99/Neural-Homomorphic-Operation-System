import os
import pandas as pd
import re
import glob


def extract_metrics(base_path='.'):
    """
    Extract final accuracy and best accuracy with round number from FL metrics CSV files.
    Searches recursively through directories matching the x-rounds/y-clients pattern.
    """
    # Create a list to store the results
    results = []

    # Find all CSV files that match the pattern
    pattern = r'(\d+)-rounds/(\d+)-clients/fl_run_metrics_num_clients-\d+.csv'
    csv_files = []

    # Walk through directories recursively
    for root, _, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(root, file)
            # Normalize path for consistent matching across OS
            normalized_path = full_path.replace('\\', '/')
            if re.search(pattern, normalized_path):
                csv_files.append(full_path)

    if not csv_files:
        print("No CSV files found. Make sure you're running this script from the correct directory.")
        print("Looking for: x-rounds/y-clients/fl_run_metrics_num_clients-y.csv")
        return

    for csv_path in csv_files:
        try:
            # Normalize path for consistent matching
            normalized_path = csv_path.replace('\\', '/')

            # Extract num_rounds and num_clients from the path
            match = re.search(r'(\d+)-rounds/(\d+)-clients', normalized_path)
            if match:
                num_rounds = int(match.group(1))
                num_clients = int(match.group(2))

                # Read the CSV file
                df = pd.read_csv(csv_path)

                if df.empty:
                    print(f"Warning: {csv_path} is empty.")
                    continue

                # Check if the CSV has the required columns
                required_columns = ['round', 'accuracy']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: {csv_path} does not have the required columns: {required_columns}")
                    continue

                # Get the final accuracy (from the last row)
                final_accuracy = df['accuracy'].iloc[-1]

                # Find the best accuracy and its round number
                best_idx = df['accuracy'].idxmax()
                best_accuracy = df['accuracy'].iloc[best_idx]
                best_round = df['round'].iloc[best_idx]

                # Store the results
                results.append({
                    'num_rounds': num_rounds,
                    'num_clients': num_clients,
                    'final_accuracy': final_accuracy,
                    'best_accuracy(roundNumber)': f"{best_accuracy}({best_round})"
                })

                print(f"Processed {csv_path}")
            else:
                print(f"Warning: Could not extract information from path: {csv_path}")
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    # Create output CSV
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv('extracted_metrics.csv', index=False)
        print(f"Results saved to extracted_metrics.csv")
    else:
        print("No valid results extracted from CSV files.")


if __name__ == "__main__":
    extract_metrics()