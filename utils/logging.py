import wandb

class Logger:
    """
    Logger class that uses Weights & Biases (wandb).
    """
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        self.config = config

        # Initialize wandb
        wandb.init(project=config["project_name"], name=config["run_name"], config=config)

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
