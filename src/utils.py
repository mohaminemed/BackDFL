import csv
import os
from datetime import datetime

class ResultsLogger:
    """
    A simple utility to log experiment results to a CSV file.
    Creates one file per experiment run with a clean header and row-by-row results.
    """
    def __init__(self, experiment_name: str, config: dict):
        """
        Initializes the logger and creates the results file with a header.

        Args:
            experiment_name (str): A name for the experiment, used in the filename.
            config (dict): The experiment configuration (used for filename, not content).
        """
        self.log_dir = "experiments/outputs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name_from_config = config.get('experiment_name', 'experiment')
        self.filename = os.path.join(self.log_dir, f"{exp_name_from_config}.csv")
        
        self._write_header()

    def _write_header(self):
        """Writes a simple header row for the results."""
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'main_accuracy', 'min_accuracy', 'backdoor_accuracy', 'attack_active'])

    def log_round(self, round_idx: int, main_accuracy: float, min_accuracy: float, backdoor_accuracy: float, is_attack_active: bool):
        """
        Logs the metrics for a single round by appending a new row to the CSV file.
        """
        row = [round_idx, f"{main_accuracy:.4f}", f"{min_accuracy:.4f}", f"{backdoor_accuracy:.4f}", int(is_attack_active)]
        
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)