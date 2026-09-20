import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_accuracy,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    model,
    optimizer,
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

    return checkpoint