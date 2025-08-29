import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_log_metrics(logs: list[dict], plot_path: str) -> None:
    palette = sns.color_palette("muted")
    loss_color, acc_color = palette[4], palette[6]
    
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    token_accuracies = []

    for log_entry in logs:
        if "loss" in log_entry and "step" in log_entry:
            train_losses.append(log_entry["loss"])
            train_steps.append(log_entry["step"])

        if "eval_loss" in log_entry and "step" in log_entry:
            eval_losses.append(log_entry["eval_loss"])
            eval_steps.append(log_entry["step"])

        if "eval_mean_token_accuracy" in log_entry:
            token_accuracies.append(log_entry["eval_mean_token_accuracy"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax1, ax2 = axes

    # Plot Losses
    if train_losses or eval_losses:
        # Training Loss
        if train_losses:
            ax1.plot(
                train_steps,
                train_losses,
                "o-",
                linewidth=2,
                label="Training Loss",
                alpha=0.5,
                color=loss_color,
            )
            # Mark best and last training loss
            best_train_idx = int(np.argmin(train_losses))
            last_train_idx = len(train_losses) - 1
            # Best
            ax1.annotate(
                f"Best: {train_losses[best_train_idx]:.4f}",
                (train_steps[best_train_idx], train_losses[best_train_idx]),
                textcoords="offset points",
                xytext=(0, -20),
                ha="center",
                color=loss_color,
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=loss_color, lw=1, alpha=0.7),
                arrowprops=dict(arrowstyle="->", color=loss_color, lw=1),
            )
            # Last (if not same as best)
            if last_train_idx != best_train_idx:
                ax1.annotate(
                    f"Last: {train_losses[last_train_idx]:.4f}",
                    (train_steps[last_train_idx], train_losses[last_train_idx]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color=loss_color,
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=loss_color, lw=1, alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color=loss_color, lw=1),
                )

        # Evaluation Loss
        if eval_losses:
            ax1.plot(
                eval_steps,
                eval_losses,
                "o-",
                linewidth=3,
                label="Evaluation Loss",
                alpha=1.0,
                color=loss_color,
            )
            # Mark best and last eval loss
            best_eval_idx = int(np.argmin(eval_losses))
            last_eval_idx = len(eval_losses) - 1
            # Best
            ax1.annotate(
                f"Best: {eval_losses[best_eval_idx]:.4f}",
                (eval_steps[best_eval_idx], eval_losses[best_eval_idx]),
                textcoords="offset points",
                xytext=(0, -20),
                ha="center",
                color=loss_color,
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=loss_color, lw=1, alpha=0.7),
                arrowprops=dict(arrowstyle="->", color=loss_color, lw=1),
            )
            # Last (if not same as best)
            if last_eval_idx != best_eval_idx:
                ax1.annotate(
                    f"Last: {eval_losses[last_eval_idx]:.4f}",
                    (eval_steps[last_eval_idx], eval_losses[last_eval_idx]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color=loss_color,
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=loss_color, lw=1, alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color=loss_color, lw=1),
                )

        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training vs Evaluation Loss", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Token Accuracy
        if token_accuracies:
            ax2.plot(
                eval_steps,
                token_accuracies,
                "o-",
                linewidth=3,
                label="Token Accuracy",
                alpha=1.0,
                color=acc_color,
            )
            # Mark best and last token accuracy
            best_acc_idx = int(np.argmax(token_accuracies))
            last_acc_idx = len(token_accuracies) - 1
            # Best
            ax2.annotate(
                f"Best: {token_accuracies[best_acc_idx]:.4f} ({token_accuracies[best_acc_idx]:.2%})",
                (eval_steps[best_acc_idx], token_accuracies[best_acc_idx]),
                textcoords="offset points",
                xytext=(0, -20),
                ha="center",
                color=acc_color,
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=acc_color, lw=1, alpha=0.7),
                arrowprops=dict(arrowstyle="->", color=acc_color, lw=1),
            )
            # Last (if not same as best)
            if last_acc_idx != best_acc_idx:
                ax2.annotate(
                    f"Last: {token_accuracies[last_acc_idx]:.4f} ({token_accuracies[last_acc_idx]:.2%})",
                    (eval_steps[last_acc_idx], token_accuracies[last_acc_idx]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color=acc_color,
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=acc_color, lw=1, alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color=acc_color, lw=1),
                )

            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Token Accuracy")
            ax2.set_title("Token Accuracy", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: "{:.1%}".format(y))
            )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, facecolor="white")
    plt.close()

    print(f"Training metrics plot saved to: {plot_path}")
