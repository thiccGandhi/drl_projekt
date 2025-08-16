import wandb
from datetime import datetime
import os
import tempfile
import pickle
import imageio
import copy


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
        self.version_number = version_number

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{prefix}_v{version_number}_{timestamp}"

        # Initialize wandb
        wandb.init(project=config["project_name"], name=run_name, config=config, dir=self.log_dir + run_name)
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
        
    
    def save_agent(self, option, agent, episode_data, episode_counter):
        """Saves the agent as a pickle file without the buffer.

        Args:
            option: The option `"best"` to overwrite the best agent or the number of K iterations for freq saving.
            agent: The agent object to save.
            episode_data: The episode data to save.
            episode_counter: The episode from which the agent and data are.
        """
        agent_to_save = copy.copy(agent)
        agent_to_save.replay_buffer = None  # Clear the replay buffer to save memory
            

        # save the agent
        if option == "best":
            filename = f"best_agent_{episode_counter}.pkl"
        elif isinstance(option, int):
            filename = f"agent_{option}k.pkl"
        else:
            raise ValueError("Invalid option for saving agent. Option must be 'best' or an integer.")
        
        filename = str(self.version_number) + "_" + filename

        # Save the agent first as temp locally and then to wandb
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path_agent = os.path.join(temp_dir, filename)
            tmp_path_data = os.path.join(temp_dir, "episode_data.pkl")
            with open(tmp_path_agent, "wb") as f:
                pickle.dump(agent_to_save, f)
            with open(tmp_path_data, "wb") as f:
                pickle.dump(episode_data, f)

            artifact = wandb.Artifact("agent", type="model")
            artifact.add_file(tmp_path_agent, name=filename)
            artifact.add_file(tmp_path_data, name="episode_data.pkl")
            wandb.log_artifact(artifact, aliases=[filename])
            
    
    def save_animation(self, option, gif_array, episode_counter):
        """
        Saves an animation as a video file to wandb.
        
        Args:
            option: The option to specify the saving behavior (e.g., "best" or a specific iteration).
            gif_array: The animation object to save.
            episode_counter: The episode the animation is from.
        """
        if option == "best":
            filename = f"best_agent_{episode_counter}.gif"
        elif isinstance(option, int):
            filename = f"agent_{option}k.gif"
        else:
            raise ValueError("Invalid option for saving animation. Option must be 'best' or an integer.")

        filename = str(self.version_number) + "_" + filename
        
        # save imageio as tmp first and then upload to wandb
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = os.path.join(temp_dir, filename)
            imageio.mimsave(tmp_path, gif_array, fps=30)

            wandb.log({f"agent_animations/{filename}": wandb.Video(tmp_path, format="gif")})


    def close(self):
        wandb.finish()
