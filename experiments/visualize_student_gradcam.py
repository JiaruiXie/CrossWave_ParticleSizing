"""
Visualize Grad-CAM for a single-wavelength student model.

This script generates, for selected examples:

    1. Normalized input iSCAT patch
    2. Grad-CAM heatmap
    3. Overlay: input patch + Grad-CAM
    4. Optional combined panel

It supports:

    --student_channel 440
    --student_channel 555
    --student_channel 740

It also supports selecting which convolutional layer to use for Grad-CAM:

    --gradcam_layer first
    --gradcam_layer middle
    --gradcam_layer last

For 64x64 microscopy patches, "middle" or "first" is often more spatially meaningful
than "last", because the last convolutional feature map may be too coarse.

Example for 555 nm KD student:

    & "C:\\Users\\jiaru\\.conda\\envs\\LNP_image_analysis\\python.exe" experiments/visualize_student_gradcam.py --checkpoint "outputs\\optuna\\kd_student_from_fusion_teacher_v2_precomputed_student_555_at_1_ap_0.5_af_0.1_soft_sampler_alpha_0.3\\trial_XXX\\best_student_model.pt" --student_channel 555 --num_examples 12 --example_strategy high_error --gradcam_layer middle

Example for cleaner examples:

    & "C:\\Users\\jiaru\\.conda\\envs\\LNP_image_analysis\\python.exe" experiments/visualize_student_gradcam.py --checkpoint "outputs\\optuna\\kd_student_from_fusion_teacher_v2_precomputed_student_555_at_1_ap_0.5_af_0.1_soft_sampler_alpha_0.3\\trial_XXX\\best_student_model.pt" --student_channel 555 --num_examples 12 --example_strategy low_error --gradcam_layer middle
"""

import sys
from pathlib import Path
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.particle_dataset import ParticleDataset
from models.baseline_cnn import ConfigurableCNNRegressor
from utils.seed import set_seed


def parse_args():

    parser = argparse.ArgumentParser(
        description="Visualize Grad-CAM for a single-wavelength student model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_student_model.pt or best_model.pt."
    )

    parser.add_argument(
        "--student_channel",
        type=str,
        required=True,
        choices=["440", "555", "740"],
        help="Single wavelength used by the student model."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to visualize."
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=12,
        help="Number of examples to visualize."
    )

    parser.add_argument(
        "--example_strategy",
        type=str,
        default="high_error",
        choices=[
            "random",
            "high_error",
            "low_error",
            "high_intensity",
            "low_intensity",
            "under_prediction",
            "over_prediction"
        ],
        help="How to select examples."
    )

    parser.add_argument(
        "--gradcam_layer",
        type=str,
        default="middle",
        choices=["first", "middle", "last"],
        help="Which Conv2d layer to use for Grad-CAM."
    )

    parser.add_argument(
        "--gradcam_mode",
        type=str,
        default="absolute",
        choices=["positive", "absolute", "signed"],
        help=(
            "Grad-CAM mode. "
            "'positive' uses ReLU CAM, standard for classification. "
            "'absolute' is recommended for regression. "
            "'signed' keeps positive and negative CAM before visualization."
        )
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for prediction pass."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If omitted, save next to checkpoint."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    parser.add_argument(
        "--print_feature_shapes",
        action="store_true",
        help="Print Conv2d feature-map shapes before Grad-CAM."
    )

    return parser.parse_args()


def get_channel_index(student_channel):

    mapping = {
        "440": 0,
        "555": 1,
        "740": 2
    }

    return mapping[student_channel]


def get_state_dict_from_checkpoint(checkpoint):
    """
    Support both:
        checkpoint["student_state_dict"] for KD student
        checkpoint["model_state_dict"] for baseline model
    """

    if "student_state_dict" in checkpoint:

        state_dict = checkpoint["student_state_dict"]
        checkpoint_type = "student_state_dict"

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]
        checkpoint_type = "model_state_dict"

    else:

        raise KeyError(
            "Checkpoint must contain either 'student_state_dict' or 'model_state_dict'."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("student."):
            new_key = new_key.replace("student.", "", 1)

        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)

        cleaned_state_dict[new_key] = value

    return cleaned_state_dict, checkpoint_type


def infer_architecture_from_state_dict(state_dict):
    """
    Infer ConfigurableCNNRegressor architecture directly from checkpoint weights.

    This avoids dropout mismatch problems such as:
        checkpoint has regressor.3.weight
        rebuilt model expects regressor.4.weight
    """

    conv_items = []

    for key, value in state_dict.items():

        if key.startswith("features.") and key.endswith(".weight"):

            if value.ndim == 4:

                layer_index = int(
                    key.split(".")[1]
                )

                out_channels = int(
                    value.shape[0]
                )

                kernel_size = int(
                    value.shape[2]
                )

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

    if len(conv_channels) == 0:
        raise ValueError(
            "Could not infer convolutional layers from checkpoint."
        )

    linear_items = []

    for key, value in state_dict.items():

        if key.startswith("regressor.") and key.endswith(".weight"):

            if value.ndim == 2:

                layer_index = int(
                    key.split(".")[1]
                )

                out_features = int(
                    value.shape[0]
                )

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

    if len(linear_items) == 0:
        raise ValueError(
            "Could not infer regressor layers from checkpoint."
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

    dropout_rate = 0.0

    if len(linear_indices) >= 2:

        gaps = [
            linear_indices[i + 1] - linear_indices[i]
            for i in range(len(linear_indices) - 1)
        ]

        if any(gap == 3 for gap in gaps):
            dropout_rate = 0.2
        else:
            dropout_rate = 0.0

    architecture = {
        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "use_batch_norm": True
    }

    return architecture


def build_model_from_checkpoint(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    state_dict, checkpoint_type = get_state_dict_from_checkpoint(
        checkpoint
    )

    architecture = infer_architecture_from_state_dict(
        state_dict
    )

    model = ConfigurableCNNRegressor(
        in_channels=1,
        conv_channels=architecture["conv_channels"],
        kernel_sizes=architecture["kernel_sizes"],
        dense_units=architecture["dense_units"],
        dropout_rate=architecture["dropout_rate"],
        use_batch_norm=architecture["use_batch_norm"],
        use_pooling=True
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model, architecture, checkpoint, checkpoint_type


def find_conv_layer(model, layer_position="middle"):
    """
    Select a Conv2d layer for Grad-CAM.

    layer_position:
        "first"  = highest spatial resolution, low-level features
        "middle" = compromise, recommended for 64x64 patches
        "last"   = most semantic, but often too spatially coarse
    """

    conv_layers = []

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            conv_layers.append(module)

    if len(conv_layers) == 0:

        raise ValueError(
            "No Conv2d layer found in model."
        )

    if layer_position == "first":

        selected_index = 0

    elif layer_position == "middle":

        selected_index = len(conv_layers) // 2

    elif layer_position == "last":

        selected_index = len(conv_layers) - 1

    else:

        raise ValueError(
            f"Unknown layer_position: {layer_position}"
        )

    selected_layer = conv_layers[
        selected_index
    ]

    print()
    print("=" * 80)
    print("Grad-CAM target layer")
    print("=" * 80)
    print(f"Layer position: {layer_position}")
    print(f"Conv layer index among Conv2d layers: {selected_index}")
    print(selected_layer)

    return selected_layer, selected_index


def print_feature_map_shapes(
    model,
    input_shape=(1, 1, 64, 64),
    device="cpu"
):
    """
    Print output shape of every Conv2d layer.
    This is useful because late Grad-CAM layers may be too coarse for 64x64 patches.
    """

    model.eval()

    x = torch.zeros(
        input_shape,
        dtype=torch.float32
    ).to(device)

    print()
    print("=" * 80)
    print("Conv feature map shapes")
    print("=" * 80)

    hooks = []
    conv_index = 0

    def make_hook(index):

        def hook(module, input, output):

            print(
                f"Conv layer {index}: output shape = {tuple(output.shape)}"
            )

        return hook

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            hooks.append(
                module.register_forward_hook(
                    make_hook(conv_index)
                )
            )

            conv_index += 1

    with torch.no_grad():

        _ = model(x)

    for hook in hooks:

        hook.remove()


class GradCAM:
    """
    Grad-CAM for scalar regression output.

    For this model:
        output = predicted cube-root Cy3 max intensity

    The gradient is taken with respect to this scalar output.

    mode:
        positive = standard Grad-CAM, ReLU applied to CAM
        absolute = absolute CAM, recommended for regression
        signed   = signed CAM, normalized around magnitude
    """

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self.forward_hook
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, input, output):

        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, mode="absolute"):
        """
        input_tensor shape:
            [1, 1, 64, 64]
        """

        self.model.zero_grad()

        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        output = self.model(
            input_tensor
        )

        score = output.squeeze()

        score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Forward hook did not capture activations."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Backward hook did not capture gradients."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam_raw = (
            weights
            *
            self.activations
        ).sum(
            dim=1,
            keepdim=True
        )

        if mode == "positive":

            cam = F.relu(
                cam_raw
            )

        elif mode == "absolute":

            cam = torch.abs(
                cam_raw
            )

        elif mode == "signed":

            cam = cam_raw

        else:

            raise ValueError(
                f"Unknown Grad-CAM mode: {mode}"
            )

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        raw_cam_min = float(
            cam_raw.min().detach().cpu()
        )

        raw_cam_max = float(
            cam_raw.max().detach().cpu()
        )

        gradient_norm = float(
            self.gradients.norm().detach().cpu()
        )

        activation_norm = float(
            self.activations.norm().detach().cpu()
        )

        cam_min = cam.min()
        cam_max = cam.max()

        if mode == "signed":

            max_abs = max(
                abs(cam_min),
                abs(cam_max)
            )

            if max_abs > 0:

                # Convert signed map from [-max_abs, max_abs] to [0, 1].
                cam = (
                    cam + max_abs
                ) / (
                    2 * max_abs
                )

            else:

                cam = np.zeros_like(
                    cam
                )

        else:

            if cam_max > cam_min:

                cam = (
                    cam - cam_min
                ) / (
                    cam_max - cam_min
                )

            else:

                cam = np.zeros_like(
                    cam
                )

        prediction = float(
            output.detach().cpu().numpy().reshape(-1)[0]
        )

        debug_info = {
            "raw_cam_min": raw_cam_min,
            "raw_cam_max": raw_cam_max,
            "gradient_norm": gradient_norm,
            "activation_norm": activation_norm,
            "normalized_cam_min": float(cam.min()),
            "normalized_cam_max": float(cam.max())
        }

        return cam, prediction, debug_info


def predict_all_samples(model, dataloader, device):

    model.eval()

    rows = []

    sample_counter = 0

    with torch.no_grad():

        for batch in dataloader:

            images = batch["image"].to(device)

            targets = batch["target"].cpu().numpy()

            predictions = model(
                images
            ).cpu().numpy()

            particle_ids = batch["particle_id"].numpy()

            image_indices = batch["image_index"].numpy()

            batch_size = images.shape[0]

            for i in range(batch_size):

                target = float(
                    targets[i]
                )

                prediction = float(
                    predictions[i]
                )

                residual = prediction - target

                rows.append(
                    {
                        "dataset_index": sample_counter,
                        "particle_id": int(particle_ids[i]),
                        "image_index": int(image_indices[i]),
                        "target": target,
                        "prediction": prediction,
                        "residual": residual,
                        "abs_error": abs(residual)
                    }
                )

                sample_counter += 1

    prediction_df = pd.DataFrame(
        rows
    )

    return prediction_df


def select_examples(prediction_df, strategy, num_examples, seed):

    if strategy == "random":

        selected_df = prediction_df.sample(
            n=min(num_examples, len(prediction_df)),
            random_state=seed
        )

    elif strategy == "high_error":

        selected_df = prediction_df.sort_values(
            "abs_error",
            ascending=False
        ).head(
            num_examples
        )

    elif strategy == "low_error":

        selected_df = prediction_df.sort_values(
            "abs_error",
            ascending=True
        ).head(
            num_examples
        )

    elif strategy == "high_intensity":

        selected_df = prediction_df.sort_values(
            "target",
            ascending=False
        ).head(
            num_examples
        )

    elif strategy == "low_intensity":

        selected_df = prediction_df.sort_values(
            "target",
            ascending=True
        ).head(
            num_examples
        )

    elif strategy == "under_prediction":

        selected_df = prediction_df.sort_values(
            "residual",
            ascending=True
        ).head(
            num_examples
        )

    elif strategy == "over_prediction":

        selected_df = prediction_df.sort_values(
            "residual",
            ascending=False
        ).head(
            num_examples
        )

    else:

        raise ValueError(
            f"Unknown example strategy: {strategy}"
        )

    selected_df = selected_df.reset_index(
        drop=True
    )

    return selected_df


def normalize_for_display(image):

    image = np.asarray(
        image
    )

    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:

        image = (
            image - image_min
        ) / (
            image_max - image_min
        )

    else:

        image = np.zeros_like(
            image
        )

    return image


def save_single_example_figures(
    input_image,
    gradcam,
    prediction,
    target,
    residual,
    particle_id,
    image_index,
    dataset_index,
    output_dir,
    prefix,
    gradcam_mode,
    gradcam_layer
):

    input_display = normalize_for_display(
        input_image
    )

    # ========================================================
    # 1. Normalized input iSCAT patch
    # ========================================================

    plt.figure(figsize=(4, 4))

    plt.imshow(
        input_image,
        cmap="gray"
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04
    )

    plt.title(
        "Normalized input iSCAT patch"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{prefix}_input_patch.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # 2. Grad-CAM heatmap
    # ========================================================

    plt.figure(figsize=(4, 4))

    plt.imshow(
        gradcam,
        cmap="inferno",
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04,
        label="Grad-CAM intensity"
    )

    plt.title(
        f"Grad-CAM heatmap\nlayer={gradcam_layer}, mode={gradcam_mode}"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{prefix}_gradcam_heatmap.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # 3. Overlay: input patch + Grad-CAM
    # ========================================================

    plt.figure(figsize=(4, 4))

    plt.imshow(
        input_display,
        cmap="gray"
    )

    plt.imshow(
        gradcam,
        cmap="inferno",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04,
        label="Grad-CAM intensity"
    )

    plt.title(
        f"Input patch + Grad-CAM overlay\nlayer={gradcam_layer}, mode={gradcam_mode}"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{prefix}_overlay.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # Optional combined panel
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        input_image,
        cmap="gray"
    )

    axes[0].set_title(
        "Normalized input"
    )

    axes[0].axis(
        "off"
    )

    axes[1].imshow(
        gradcam,
        cmap="inferno",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        f"Grad-CAM\n{gradcam_layer}, {gradcam_mode}"
    )

    axes[1].axis(
        "off"
    )

    axes[2].imshow(
        input_display,
        cmap="gray"
    )

    axes[2].imshow(
        gradcam,
        cmap="inferno",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Overlay"
    )

    axes[2].axis(
        "off"
    )

    fig.suptitle(
        (
            f"Particle {particle_id} | image_index {image_index} | dataset_index {dataset_index}\n"
            f"True = {target:.4f}, Pred = {prediction:.4f}, Residual = {residual:.4f}"
        ),
        fontsize=10
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{prefix}_panel.png",
        dpi=300
    )

    plt.close()


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint: {checkpoint_path}"
        )

    if args.output_dir is None:

        output_dir = (
            checkpoint_path.parent
            / f"gradcam_{args.student_channel}_{args.split}_{args.example_strategy}_{args.gradcam_layer}_{args.gradcam_mode}"
        )

    else:

        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Student channel: {args.student_channel}")
    print(f"Split: {args.split}")
    print(f"Example strategy: {args.example_strategy}")
    print(f"Grad-CAM layer: {args.gradcam_layer}")
    print(f"Grad-CAM mode: {args.gradcam_mode}")
    print(f"Output directory: {output_dir}")

    channel_index = get_channel_index(
        args.student_channel
    )

    model, architecture, checkpoint, checkpoint_type = build_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device
    )

    print()
    print("=" * 80)
    print("Checkpoint and architecture")
    print("=" * 80)
    print("Checkpoint type:", checkpoint_type)
    print(
        json.dumps(
            architecture,
            indent=4
        )
    )

    if args.print_feature_shapes:

        print_feature_map_shapes(
            model=model,
            input_shape=(1, 1, 64, 64),
            device=device
        )

    # ========================================================
    # Load dataset using only selected wavelength
    # ========================================================

    dataset = ParticleDataset(
        data_dir="data/processed",
        split=args.split,
        channels=[channel_index],
        target_transform="cube_root",
        use_weights=False,
        mmap=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print()
    print("=" * 80)
    print("Dataset")
    print("=" * 80)
    print(f"Loaded {args.split} dataset: {len(dataset)} samples")
    print(f"Using channel index: {channel_index}")

    # ========================================================
    # Predict all samples first, then select examples
    # ========================================================

    prediction_df = predict_all_samples(
        model=model,
        dataloader=dataloader,
        device=device
    )

    prediction_df.to_csv(
        output_dir / "all_predictions_for_selection.csv",
        index=False
    )

    selected_df = select_examples(
        prediction_df=prediction_df,
        strategy=args.example_strategy,
        num_examples=args.num_examples,
        seed=args.seed
    )

    selected_df.to_csv(
        output_dir / "selected_examples.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("Selected examples")
    print("=" * 80)
    print(selected_df)

    # ========================================================
    # Grad-CAM setup
    # ========================================================

    target_layer, target_layer_index = find_conv_layer(
        model=model,
        layer_position=args.gradcam_layer
    )

    gradcam_engine = GradCAM(
        model=model,
        target_layer=target_layer
    )

    xai_rows = []

    try:

        for example_number, row in selected_df.iterrows():

            dataset_index = int(
                row["dataset_index"]
            )

            sample = dataset[
                dataset_index
            ]

            image = sample["image"]

            if isinstance(image, torch.Tensor):

                image_tensor = image.float()

            else:

                image_tensor = torch.tensor(
                    image,
                    dtype=torch.float32
                )

            # Expected shape from ParticleDataset:
            # [1, 64, 64]
            input_tensor = image_tensor.unsqueeze(
                0
            ).to(
                device
            )

            input_tensor.requires_grad_(
                True
            )

            gradcam, prediction, debug_info = gradcam_engine.generate(
                input_tensor=input_tensor,
                mode=args.gradcam_mode
            )

            target = float(
                sample["target"]
            )

            residual = prediction - target

            particle_id = int(
                sample["particle_id"]
            )

            image_index = int(
                sample["image_index"]
            )

            input_image = image_tensor.squeeze().cpu().numpy()

            prefix = (
                f"example_{example_number:03d}"
                f"_idx_{dataset_index}"
                f"_pid_{particle_id}"
            )

            save_single_example_figures(
                input_image=input_image,
                gradcam=gradcam,
                prediction=prediction,
                target=target,
                residual=residual,
                particle_id=particle_id,
                image_index=image_index,
                dataset_index=dataset_index,
                output_dir=output_dir,
                prefix=prefix,
                gradcam_mode=args.gradcam_mode,
                gradcam_layer=args.gradcam_layer
            )

            xai_rows.append(
                {
                    "example_number": example_number,
                    "dataset_index": dataset_index,
                    "particle_id": particle_id,
                    "image_index": image_index,
                    "target": target,
                    "prediction": prediction,
                    "residual": residual,
                    "abs_error": abs(residual),
                    "student_channel": args.student_channel,
                    "channel_index": channel_index,
                    "gradcam_layer": args.gradcam_layer,
                    "gradcam_layer_index": target_layer_index,
                    "gradcam_mode": args.gradcam_mode,
                    "raw_cam_min": debug_info["raw_cam_min"],
                    "raw_cam_max": debug_info["raw_cam_max"],
                    "gradient_norm": debug_info["gradient_norm"],
                    "activation_norm": debug_info["activation_norm"],
                    "normalized_cam_min": debug_info["normalized_cam_min"],
                    "normalized_cam_max": debug_info["normalized_cam_max"],
                    "input_patch_file": f"{prefix}_input_patch.png",
                    "gradcam_file": f"{prefix}_gradcam_heatmap.png",
                    "overlay_file": f"{prefix}_overlay.png",
                    "panel_file": f"{prefix}_panel.png"
                }
            )

            print(
                f"Saved example {example_number:03d} | "
                f"idx={dataset_index} | "
                f"target={target:.4f} | "
                f"pred={prediction:.4f} | "
                f"residual={residual:.4f} | "
                f"raw_cam_min={debug_info['raw_cam_min']:.6e} | "
                f"raw_cam_max={debug_info['raw_cam_max']:.6e} | "
                f"grad_norm={debug_info['gradient_norm']:.6e}"
            )

    finally:

        gradcam_engine.remove_hooks()

    xai_summary_df = pd.DataFrame(
        xai_rows
    )

    xai_summary_df.to_csv(
        output_dir / "xai_summary.csv",
        index=False
    )

    with open(output_dir / "gradcam_config.json", "w") as f:

        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_type": checkpoint_type,
                "student_channel": args.student_channel,
                "channel_index": channel_index,
                "split": args.split,
                "num_examples": args.num_examples,
                "example_strategy": args.example_strategy,
                "gradcam_layer": args.gradcam_layer,
                "gradcam_mode": args.gradcam_mode,
                "target_layer_index": target_layer_index,
                "architecture": architecture
            },
            f,
            indent=4
        )

    print()
    print("=" * 80)
    print("Grad-CAM visualization complete")
    print("=" * 80)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":

    main()