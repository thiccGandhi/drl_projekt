import os
import matplotlib.pyplot as plt
import pandas as pd
import json

class ResultsPlotter:
    def __init__(self, trainer, out_base_dir="test_results"):
        self.trainer = trainer
        wandb_run = getattr(trainer.logger, "run", None)
        if wandb_run is not None:
            run_id = wandb_run.id
            run_name = wandb_run.name
        else:
            run_id = "no_wandb"
            run_name = trainer.logger.config.get("run_name", "unnamed_run")
        self.out_dir = os.path.join(out_base_dir, f"run_{run_id}_{run_name}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.run_info = {
            "wandb_run_id": run_id,
            "wandb_run_name": run_name,
            "wandb_dir": getattr(wandb_run, "dir", None)
        }

    def plot_training(self, save=True, show=True):
        history = self.trainer.training_history
        if not history:
            print("No training history found.")
            return

        steps = [h["step"] for h in history]
        actor_losses = [h["actor_loss"] for h in history]
        critic_losses = [h["critic_loss"] for h in history]
        success_rates = [h.get("train/success_rate_100", 0) for h in history]

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        axs[0].plot(steps, actor_losses, label="Actor Loss")
        axs[0].set_ylabel("Actor Loss")
        axs[0].set_title("Actor Loss per Episode")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(steps, critic_losses, label="Critic Loss")
        axs[1].set_ylabel("Critic Loss")
        axs[1].set_title("Critic Loss per Episode")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(steps, success_rates, label="Success Rate (last 100)", color="green")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Success Rate")
        axs[2].set_title("Moving Success Rate (last 100 episodes)")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        if save:
            out_path = os.path.join(self.out_dir, "training_curves.png")
            fig.savefig(out_path)
            print(f"Saved: {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_eval(self, save=True, show=True):
        eval_history = self.trainer.eval_history
        if not eval_history:
            print("No eval history.")
            return
        steps, rates = zip(*eval_history)
        fig = plt.figure(figsize=(8, 5))
        plt.plot(steps, rates, marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Eval Success Rate")
        plt.title("Evaluation Success Rate")
        plt.grid(True)
        if save:
            out_path = os.path.join(self.out_dir, "eval_success.png")
            plt.savefig(out_path)
            print(f"Saved: {out_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_history_csv(self):
        if self.trainer.training_history:
            train_path = os.path.join(self.out_dir, "training_history.csv")
            pd.DataFrame(self.trainer.training_history).to_csv(train_path, index=False)
            print(f"Saved: {train_path}")
        if self.trainer.eval_history:
            eval_path = os.path.join(self.out_dir, "eval_history.csv")
            pd.DataFrame(self.trainer.eval_history, columns=["step", "eval_success"]).to_csv(eval_path, index=False)
            print(f"Saved: {eval_path}")

    def save_config(self):
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(self.trainer.config, f, indent=2)
        print(f"Saved: {os.path.join(self.out_dir, 'config.json')}")
    
    def save_run_info(self):
        with open(os.path.join(self.out_dir, "run_info.json"), "w") as f:
            json.dump(self.run_info, f, indent=2)
        print(f"Saved: {os.path.join(self.out_dir, 'run_info.json')}")

    def plot_all(self, save=True, show=True):
        self.plot_training(save=save, show=show)
        self.plot_eval(save=save, show=show)
        self.save_history_csv()
        self.save_config()
        self.save_run_info()
