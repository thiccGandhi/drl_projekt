# drl_projekt/utils/mylogging.py
import os
import tempfile
import pickle
import imageio
import copy
from datetime import datetime
import wandb

class Logger:
    """
    W&B logger that gives each sweep job a unique, informative run name.
    """
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        self.config = config

        # Build a unique run name
        prefix = str(config.get("run_name", "run"))
        parts = [prefix]

        # Include env/agent if present in config (helps grouping)
        if "env_name" in config:
            parts.append(config["env_name"])
        if "agent" in config:
            parts.append(config["agent"])

        # Add SLURM IDs (very helpful during arrays)
        sj = os.getenv("SLURM_JOB_ID")
        sa = os.getenv("SLURM_ARRAY_TASK_ID")
        if sj: parts.append(f"J{sj}")
        if sa: parts.append(f"A{sa}")

        # Include cfg filename (cfg_000 etc.) if available
        cfg_path = os.getenv("CFG_PATH")
        if cfg_path:
            parts.append(os.path.splitext(os.path.basename(cfg_path))[0])

        # Timestamp for final uniqueness
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_name = "__".join(parts)

        # Ensure local dir exists; store each run in its own subdir
        run_dir = os.path.join(self.log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Initialize W&B
        self.run = wandb.init(
            project=config["project_name"],
            entity="t_p-universit-t-ulm",
            name=run_name,
            config=config,
            dir=run_dir,
            resume="never",
        )

    def log_episode(self, metrics, step):
        wandb.log(metrics, step=step)

    def log_eval(self, eval_metrics, step):
        wandb.log(eval_metrics, step=step)

    def save_agent(self, option, agent, episode_data, episode_counter):
        """Save agent (without buffer) + episode_data as a W&B artifact."""
        agent_to_save = copy.copy(agent)
        agent_to_save.replay_buffer = None

        if option == "best":
            filename = f"best_agent_{episode_counter}.pkl"
        elif isinstance(option, int):
            filename = f"agent_{option}k.pkl"
        else:
            raise ValueError("Option must be 'best' or an integer.")

        # Name artifact uniquely per run
        artifact_name = f"agent-{self.run.id}"

        with tempfile.TemporaryDirectory() as tmpdir:
            p_agent = os.path.join(tmpdir, filename)
            p_data  = os.path.join(tmpdir, "episode_data.pkl")
            with open(p_agent, "wb") as f:
                pickle.dump(agent_to_save, f)
            with open(p_data, "wb") as f:
                pickle.dump(episode_data, f)

            art = wandb.Artifact(artifact_name, type="model")
            art.add_file(p_agent, name=filename)
            art.add_file(p_data,  name="episode_data.pkl")
            wandb.log_artifact(art, aliases=[filename])

    def save_animation(self, option, gif_array, episode_counter):
        """Save a GIF to W&B (logged as a Video)."""
        if option == "best":
            filename = f"best_agent_{episode_counter}.gif"
        elif isinstance(option, int):
            filename = f"agent_{option}k.gif"
        else:
            raise ValueError("Option must be 'best' or an integer.")

        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, filename)
            imageio.mimsave(p, gif_array, fps=30)
            wandb.log({f"agent_animations/{filename}": wandb.Video(p, format="gif")})

    def close(self):
        wandb.finish()
