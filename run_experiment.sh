#!/bin/bash

# Usage: ./run_experiment.sh <attack> <defense> <dataset>
# Example: ./run_experiment.sh backdoor krum cifar10

if [ "$#" -ne 4 ]; then
    echo "❌ Usage: $0 <attack> <defense> <dataset> <flow>"
    echo "Running default: $0 scaling none gtsrb centralized"
fi

ATTACK="${1:-scaling}"  # Default to 'scaling' if not provided
DEFENSE="${2:-none}"  # Default to 'none' if not provided
DATASET="${3:-gtsrb}"  # Default to 'gtsrb' if not provided
FLOW="${4:-centralized}"  # Default to 'centralized' if not provided

TEMPLATE="experiments/configs/base_template.yml"
CONFIG_TMP="experiments/configs/tmp_${ATTACK}_${DEFENSE}_${DATASET}_${FLOW}.yml"

# Replace placeholders in the YAML template
sed "s/{attack}/${ATTACK}/g; s/{defense}/${DEFENSE}/g; s/{dataset}/${DATASET}/g; s/{flow}/${FLOW}/g" "$TEMPLATE" > "$CONFIG_TMP"

# Ensure logs directory exists
mkdir -p logs

echo "🚀 Running experiment: Attack=$ATTACK | Defense=$DEFENSE | Dataset=$DATASET | Flow=$FLOW"
echo "Using config: $CONFIG_TMP"

# Run experiment
python -m experiments.run_parallel --config "$CONFIG_TMP" > "experiments/logs/${ATTACK}_${DEFENSE}_${DATASET}_${FLOW}.log" 2>&1

# Report status
if [ $? -eq 0 ]; then
    echo "✅ Experiment completed successfully."
else
    echo "❌ Experiment failed. Check experiments/logs/${ATTACK}_${DEFENSE}_${DATASET}_${FLOW}.log"
fi

# Optional cleanup
rm -f "$CONFIG_TMP"
