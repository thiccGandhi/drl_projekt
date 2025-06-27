import os
import matplotlib.pyplot as plt
import pandas as pd

class ResultsPlotter:
    def __init__(self, trainer, out_dir=None):
        self.trainer = trainer
        # Set up result directory
        run_name = getattr(trainer.logger.config, "run_name", None) \
                   or trainer.logger.config.get("run_name", "unnamed_run")
        self.out_dir = out_dir or f"results_{run_name}"
        os.makedirs(self.out_dir, exist_ok=True)

    def plot_training(self, save=True):
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
            fig.savefig(os.path.join(self.out_dir, "training_curves.png"))
            print(f"Saved: {os.path.join(self.out_dir, 'training_curves.png')}")
        plt.show()

    def plot_eval(self, save=True):
        eval_history = self.trainer.eval_history
        if not eval_history:
            print("No eval history.")
            return
        steps, rates = zip(*eval_history)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, rates, marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Eval Success Rate")
        plt.title("Evaluation Success Rate")
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(self.out_dir, "eval_success.png"))
            print(f"Saved: {os.path.join(self.out_dir, 'eval_success.png')}")
        plt.show()

    def save_history_csv(self):
        if self.trainer.training_history:
            pd.DataFrame(self.trainer.training_history).to_csv(
                os.path.join(self.out_dir, "training_history.csv"), index=False)
            print(f"Saved: {os.path.join(self.out_dir, 'training_history.csv')}")
        if self.trainer.eval_history:
            pd.DataFrame(self.trainer.eval_history, columns=["step", "eval_success"]).to_csv(
                os.path.join(self.out_dir, "eval_history.csv"), index=False)
            print(f"Saved: {os.path.join(self.out_dir, 'eval_history.csv')}")

    def save_config(self):
        # Save a copy of your config
        import json
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(self.trainer.config, f, indent=2)
        print(f"Saved: {os.path.join(self.out_dir, 'config.json')}")

    def plot_all(self, save=True):
        self.plot_training(save=save)
        self.plot_eval(save=save)
        self.save_history_csv()
        self.save_config()
