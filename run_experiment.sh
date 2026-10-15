#!/usr/bin/env bash
# Usage: ./run_experiment.sh <attack> <defense> <dataset> <flow>
# Example: ./run_experiment.sh iba saga gtsrb decentralized
#
# Optional environment overrides (used by run_seeds.sh):
#   CONFIG_PATH : config file to use instead of experiments/configs/base_template.yml
#                 (may still contain {attack} {defense} {dataset} {flow} placeholders)
#   LOG_FILE    : where the python output goes
#                 (default experiments/logs/<attack>_<defense>_<dataset>_<flow>.log)

if [ "$#" -ne 4 ]; then
    echo "❌ Usage: $0 <attack> <defense> <dataset> <flow>"
    echo "Running defaults for missing arguments (scaling none gtsrb centralized)"
fi

ATTACK="${1:-scaling}"
DEFENSE="${2:-none}"
DATASET="${3:-gtsrb}"
FLOW="${4:-centralized}"

TEMPLATE="${CONFIG_PATH:-experiments/configs/base_template.yml}"
LOG_FILE="${LOG_FILE:-experiments/logs/${ATTACK}_${DEFENSE}_${DATASET}_${FLOW}.log}"

if [ ! -f "$TEMPLATE" ]; then
    echo "❌ Config template not found: $TEMPLATE"
    exit 2
fi

mkdir -p experiments/configs experiments/logs "$(dirname "$LOG_FILE")"

# Unique temp config per invocation (safe for parallel runs)
CONFIG_TMP="$(mktemp "experiments/configs/tmp_${ATTACK}_${DEFENSE}_${DATASET}_${FLOW}_XXXXXX.yml")"
trap 'rm -f "$CONFIG_TMP"' EXIT

# Fill the placeholders
sed "s/{attack}/${ATTACK}/g; s/{defense}/${DEFENSE}/g; s/{dataset}/${DATASET}/g; s/{flow}/${FLOW}/g" \
    "$TEMPLATE" > "$CONFIG_TMP"

echo "🚀 Running experiment: Attack=$ATTACK | Defense=$DEFENSE | Dataset=$DATASET | Flow=$FLOW"
echo "Template : $TEMPLATE"
echo "Config   : $CONFIG_TMP"
echo "Log file : $LOG_FILE"

PYTHONUNBUFFERED=1 python -m experiments.run_parallel --config "$CONFIG_TMP" > "$LOG_FILE" 2>&1
rc=$?

if [ $rc -eq 0 ]; then
    echo "✅ Experiment completed successfully."
else
    echo "❌ Experiment failed (rc=$rc). Check $LOG_FILE"
fi

exit $rc