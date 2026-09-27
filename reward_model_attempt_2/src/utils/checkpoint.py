import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_metric,
    val_loss,
    val_accuracy,

    config,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "global_step" : global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "best_metric": best_metric,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "config": config,
        "seed": config["seed"],
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if (
        scheduler is not None
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint