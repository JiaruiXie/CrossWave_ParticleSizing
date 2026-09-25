"""
Precompute fusion teacher predictions and embeddings.

Teacher:
    input = [440, 555, 740]
    output = prediction + embedding

Saved files:
    data/processed/teacher_outputs/<teacher_name>/train_teacher_predictions.npy
    data/processed/teacher_outputs/<teacher_name>/train_teacher_embeddings.npy
    data/processed/teacher_outputs/<teacher_name>/val_teacher_predictions.npy
    data/processed/teacher_outputs/<teacher_name>/val_teacher_embeddings.npy
    data/processed/teacher_outputs/<teacher_name>/test_teacher_predictions.npy
    data/processed/teacher_outputs/<teacher_name>/test_teacher_embeddings.npy
"""

import sys
from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Precompute teacher predictions and embeddings."
    )

    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--teacher_name",
        type=str,
        required=True,
        help="Name used for output folder, e.g. fusion_trial_051."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="data/processed/teacher_outputs"
    )

    return parser.parse_args()


def infer_teacher_architecture_from_checkpoint(checkpoint):
    """
    Infer ConfigurableCNNRegressor architecture directly from checkpoint weights.
    """

    state_dict = checkpoint["model_state_dict"]

    conv_items = []

    for key, value in state_dict.items():

        if key.startswith("features.") and key.endswith(".weight"):

            if value.ndim == 4:

                layer_index = int(key.split(".")[1])
                out_channels = value.shape[0]
                kernel_size = value.shape[2]

                conv_items.append(
                    (
                        layer_index,
                        out_channels,
                        kernel_size
                    )
                )

    conv_items = sorted(
        conv_items,
        key=lambda x: x[0]
    )

    conv_channels = [
        item[1]
        for item in conv_items
    ]

    kernel_sizes = [
        item[2]
        for item in conv_items
    ]

    linear_items = []

    for key, value in state_dict.items():

        if key.startswith("regressor.") and key.endswith(".weight"):

            if value.ndim == 2:

                layer_index = int(key.split(".")[1])
                out_features = value.shape[0]

                linear_items.append(
                    (
                        layer_index,
                        out_features
                    )
                )

    linear_items = sorted(
        linear_items,
        key=lambda x: x[0]
    )

    dense_units = [
        item[1]
        for item in linear_items
        if item[1] != 1
    ]

    linear_indices = [
        item[0]
        for item in linear_items
    ]

    if len(linear_indices) >= 2:

        if linear_indices[1] - linear_indices[0] == 3:
            dropout_rate = 0.2
        else:
            dropout_rate = 0.0

    else:

        dropout_rate = 0.2

    config = checkpoint.get("config", None)

    if config is None:
        config = checkpoint.get("params", None)

    if config is not None:
        dropout_rate = config.get(
            "dropout_rate",
            dropout_rate
        )

    arch = {
        "in_channels": 3,
        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "use_batch_norm": True,
        "embedding_dim": conv_channels[-1]
    }

    return arch


def build_teacher(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    arch = infer_teacher_architecture_from_checkpoint(
        checkpoint
    )

    teacher = ConfigurableCNNRegressor(
        in_channels=3,
        conv_channels=arch["conv_channels"],
        kernel_sizes=arch["kernel_sizes"],
        dense_units=arch["dense_units"],
        dropout_rate=arch["dropout_rate"],
        use_batch_norm=True,
        use_pooling=True
    )

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher, arch, checkpoint


def precompute_split(
    teacher,
    split,
    output_dir,
    batch_size,
    device
):

    print()
    print(f"Precomputing split: {split}")

    dataset = ParticleDataset(
        data_dir="data/processed",
        split=split,
        channels=[0, 1, 2],
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    all_predictions = []
    all_embeddings = []
    all_targets = []
    all_particle_ids = []
    all_image_indices = []

    teacher.eval()

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            images = batch["image"].to(device)
            targets = batch["target"].cpu().numpy()

            teacher_pred, teacher_embedding = teacher.forward_with_embedding(
                images
            )

            all_predictions.append(
                teacher_pred.detach().cpu().numpy()
            )

            all_embeddings.append(
                teacher_embedding.detach().cpu().numpy()
            )

            all_targets.append(
                targets
            )

            all_particle_ids.extend(
                batch["particle_id"].numpy().tolist()
            )

            all_image_indices.extend(
                batch["image_index"].numpy().tolist()
            )

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches")

    predictions = np.concatenate(
        all_predictions,
        axis=0
    ).astype(np.float32)

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    ).astype(np.float32)

    targets = np.concatenate(
        all_targets,
        axis=0
    ).astype(np.float32)

    particle_ids = np.asarray(
        all_particle_ids
    )

    image_indices = np.asarray(
        all_image_indices
    )

    np.save(
        output_dir / f"{split}_teacher_predictions.npy",
        predictions
    )

    np.save(
        output_dir / f"{split}_teacher_embeddings.npy",
        embeddings
    )

    np.save(
        output_dir / f"{split}_targets.npy",
        targets
    )

    np.save(
        output_dir / f"{split}_particle_ids.npy",
        particle_ids
    )

    np.save(
        output_dir / f"{split}_image_indices.npy",
        image_indices
    )

    print(f"Saved {split} predictions: {predictions.shape}")
    print(f"Saved {split} embeddings: {embeddings.shape}")

    return {
        "split": split,
        "num_samples": int(len(predictions)),
        "prediction_shape": list(predictions.shape),
        "embedding_shape": list(embeddings.shape)
    }


def main():

    args = parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = (
        Path(args.output_root)
        / args.teacher_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("Precompute Fusion Teacher Outputs")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Teacher name: {args.teacher_name}")
    print(f"Output dir: {output_dir}")

    teacher, teacher_arch, checkpoint = build_teacher(
        checkpoint_path=args.teacher_checkpoint,
        device=device
    )

    print()
    print("Teacher architecture:")
    print(json.dumps(teacher_arch, indent=4))

    summary = {
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_name": args.teacher_name,
        "teacher_arch": teacher_arch,
        "splits": {}
    }

    for split in ["train", "val", "test"]:

        split_summary = precompute_split(
            teacher=teacher,
            split=split,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=device
        )

        summary["splits"][split] = split_summary

    with open(output_dir / "teacher_output_summary.json", "w") as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print("Precomputation finished")
    print("=" * 80)
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":

    main()