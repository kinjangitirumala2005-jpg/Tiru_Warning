"""
Wildfire Smoke Detection - CNN Model Training Script
Supports:
  1. MobileNetV2 with Transfer Learning (Recommended)
  2. Custom 4-Layer Baseline CNN
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore
    from tensorflow.keras.applications import MobileNetV2  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
except ImportError:
    print("TensorFlow is required to run the training script. Install with: pip install tensorflow matplotlib")
    exit(1)


def build_data_augmentation():
    """Augmentation pipeline to help model generalize across different smoke densities & illumination."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def build_mobilenetv2_model(input_shape=(224, 224, 3)):
    """Creates a Transfer Learning model using pre-trained MobileNetV2 backbone."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    # Freeze the pre-trained weights initially
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = build_data_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="smoke_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Wildfire_Smoke_MobileNetV2")
    return model, base_model


def build_custom_cnn_model(input_shape=(224, 224, 3)):
    """Creates a 4-layer custom CNN for baseline comparison in academic reports."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        build_data_augmentation(),
        layers.Rescaling(1.0 / 255.0),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid")
    ], name="Wildfire_Smoke_CustomCNN")
    return model, None


def train_model(dataset_dir, model_type="mobilenetv2", epochs=20, batch_size=32, output_dir="models"):
    dataset_path = Path(dataset_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = (224, 224)

    print(f"\n--- Loading Dataset from: {dataset_path} ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "train" if (dataset_path / "train").exists() else dataset_path,
        validation_split=0.2 if not (dataset_path / "val").exists() else None,
        subset="training" if not (dataset_path / "val").exists() else None,
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path / "val" if (dataset_path / "val").exists() else dataset_path,
        validation_split=0.2 if not (dataset_path / "val").exists() else None,
        subset="validation" if not (dataset_path / "val").exists() else None,
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f"Detected Classes: {class_names}")
    with open(out_dir / "class_indices.json", "w") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Build model
    if model_type == "mobilenetv2":
        model, base_model = build_mobilenetv2_model(input_shape=(224, 224, 3))
    else:
        model, base_model = build_custom_cnn_model(input_shape=(224, 224, 3))

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    model_filename = f"smoke_detector_{model_type}.keras"
    best_model_path = out_dir / model_filename

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(best_model_path), monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    print(f"\n--- Starting Training ({model_type.upper()}) for {epochs} Epochs ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final model as well
    model.save(str(best_model_path))
    print(f"\n[Success] Model successfully saved to: {best_model_path}")

    # Plot & Save Training Curves
    plot_training_curves(history, out_dir / f"training_curves_{model_type}.png")

    return model, history


def plot_training_curves(history, save_path):
    """Generates Loss and Accuracy charts."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(acc, label="Train Accuracy", lw=2)
    axes[0].plot(val_acc, label="Val Accuracy", lw=2)
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(loss, label="Train Loss", lw=2, color="orange")
    axes[1].plot(val_loss, label="Val Loss", lw=2, color="red")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    plt.close()
    print(f"[Metrics] Training curves saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wildfire Smoke CNN Detector")
    parser.add_argument("--data", type=str, default="data/processed", help="Path to prepared dataset directory")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2", "custom_cnn"], help="Architecture")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--out", type=str, default="models", help="Output directory for saved model")

    args = parser.parse_args()
    train_model(args.data, model_type=args.model, epochs=args.epochs, batch_size=args.batch, output_dir=args.out)

