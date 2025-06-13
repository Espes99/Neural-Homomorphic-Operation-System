cp learning_params.py learning_params.py.bak

rounds=(2 5 10 15 20 25 50 100)
clients=(2)

for round in "${rounds[@]}"; do
  for client in "${clients[@]}"; do
    echo "Running HO simulation with $round rounds and $client clients"

    cat > learning_params.py << EOF

GLOBAL_SEED = 142
NUM_ROUNDS = $round
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_CLIENTS = $client
CONSTRAINED = False
METHODS = ["PLAIN", "CKKS", "ABHO"]
SCALE = 100_000
EOF

    python ho_fl_pipeline.py

    # If the run was unsuccessful, report and exit, 0 success
    if [ $? -ne 0 ]; then
    echo "Error running ho_fl_pipeline.py with $round rounds and $client clients"
    mv learning_params.py.bak learning_params.py
    exit 1
    fi
  done
done

# Restore the original learning_params.py
mv learning_params.py.bak learning_params.py

echo "All runs completed!"