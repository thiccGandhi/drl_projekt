import wandb
from datetime import datetime
import os

class Logger:
    """
    Logger class that uses Weights & Biases (wandb).
    """
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        self.config = config

        api = wandb.Api()
        entity = "t_p-universit-t-ulm"
        project = config["project_name"]

        runs = api.runs(f"{entity}/{project}")
        prefix = "test"

        existing_versions = [
            run.name for run in runs
            if run.name and run.name.startswith(prefix)
        ]
        version_number = len(existing_versions) + 1

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{prefix}_v{version_number}_{timestamp}"

        # Initialize wandb
        wandb.init(project=config["project_name"], name=run_name, config=config, dir=self.log_dir)
        self.run = wandb.run


    def log_episode(self, metrics, step):
        """
        Logs training metrics at the end of an episode.
        """
        wandb.log(metrics, step=step)

    def log_eval(self, eval_metrics, step):
        """
        Logs evaluation metrics (e.g., success rate) periodically.
        """
        wandb.log(eval_metrics, step=step)

    def close(self):
        wandb.finish()
