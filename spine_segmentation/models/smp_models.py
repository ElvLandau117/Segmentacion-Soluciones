"""
Model factory using segmentation_models_pytorch (SMP).
Provides easy creation of different encoder-decoder combinations.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

from spine_segmentation.config import MODEL_CONFIGS


def create_model(
    model_name: str,
    num_classes: int = 1,
    in_channels: int = 3,
) -> nn.Module:
    """
    Create a segmentation model from the predefined configurations.

    Args:
        model_name: Key from MODEL_CONFIGS ('unet_resnet50', 'unet_efficientnet_b4',
                    'deeplabv3plus_resnet50')
        num_classes: Number of output classes (1 for binary, 24 for vertebrae, etc.)
        in_channels: Number of input channels (3 for RGB)

    Returns:
        PyTorch nn.Module segmentation model
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]
    architecture = config["architecture"]

    # Get the SMP class
    model_class = getattr(smp, architecture)

    model = model_class(
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # Raw logits - we apply sigmoid/softmax in the loss
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} ({architecture} + {config['encoder_name']})")
    print(f"  Output classes: {num_classes}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Size: {total_params * 4 / 1024**2:.1f} MB (FP32)")

    return model


def get_model_params_groups(model: nn.Module, encoder_lr: float, decoder_lr: float) -> list:
    """
    Create parameter groups with differential learning rates.
    Encoder (pretrained) gets lower LR, decoder (random init) gets higher LR.

    Args:
        model: SMP model with .encoder, .decoder, .segmentation_head attributes
        encoder_lr: Learning rate for the encoder
        decoder_lr: Learning rate for the decoder and segmentation head

    Returns:
        List of param group dicts for the optimizer
    """
    param_groups = [
        {
            "params": model.encoder.parameters(),
            "lr": encoder_lr,
            "name": "encoder",
        },
        {
            "params": model.decoder.parameters(),
            "lr": decoder_lr,
            "name": "decoder",
        },
        {
            "params": model.segmentation_head.parameters(),
            "lr": decoder_lr,
            "name": "segmentation_head",
        },
    ]
    return param_groups


def list_available_models() -> list:
    """List all available model configurations."""
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name}: {config['architecture']} + {config['encoder_name']}")
    return list(MODEL_CONFIGS.keys())
