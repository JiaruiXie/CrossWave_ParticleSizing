"""
Training utilities for particle sizing models.

This module contains reusable training and validation functions.
"""

from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    verbose=True
):
    """
    Train model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module

    dataloader : torch.utils.data.DataLoader

    criterion : loss function

    optimizer : torch optimizer

    device : torch.device

    Returns
    -------
    float
        Average training loss.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(
        dataloader,
        desc="Training",
        leave=False,
        disable=not verbose
    ):

        images = batch["image"].to(device)

        targets = batch["target"].to(device)

        weights = batch["weight"].to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(
            predictions,
            targets,
            weights
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss


# ============================================================
# Validate One Epoch
# ============================================================

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    verbose=True
):
    """
    Validate model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module

    dataloader : torch.utils.data.DataLoader

    criterion : loss function

    device : torch.device

    Returns
    -------
    float
        Average validation loss.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="Validation",
            leave=False,
            disable=not verbose
        ):

            images = batch["image"].to(device)

            targets = batch["target"].to(device)

            weights = batch["weight"].to(device)

            predictions = model(images)

            loss = criterion(
                predictions,
                targets,
                weights
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            total_samples += batch_size

    average_loss = total_loss / total_samples

    return average_loss


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    filepath
):
    """
    Save model checkpoint.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }

    torch.save(
        checkpoint,
        filepath
    )


# ============================================================
# Full Training Loop
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    output_dir,
    early_stopping_patience=20,
    verbose=True
):
    """
    Train model with validation and early stopping.

    Parameters
    ----------
    model : torch.nn.Module

    train_loader : DataLoader

    val_loader : DataLoader

    criterion : loss function

    optimizer : torch optimizer

    device : torch.device

    num_epochs : int

    output_dir : str or Path

    early_stopping_patience : int

    Returns
    -------
    history : pandas.DataFrame
        Training history.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    model.to(device)

    for epoch in range(1, num_epochs + 1):

        if verbose:
            print()
            print("=" * 60)
            print(f"Epoch {epoch}/{num_epochs}")
            print("=" * 60)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            verbose=verbose
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            verbose=verbose
        )

        if verbose:
            print(f"Train loss : {train_loss:.6f}")
            print(f"Val loss   : {val_loss:.6f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                filepath=output_dir / "best_model.pt"
            )

            if verbose:
                print("Best model saved.")

        else:

            epochs_without_improvement += 1

            if verbose:
                print(
                    f"No improvement for "
                    f"{epochs_without_improvement} epoch(s)."
                )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= early_stopping_patience:

            if verbose:
                print()
                print("Early stopping triggered.")
                print(f"Best epoch    : {best_epoch}")
                print(f"Best val loss : {best_val_loss:.6f}")

            break

    history = pd.DataFrame(history)

    history.to_csv(
        output_dir / "training_history.csv",
        index=False
    )

    return history