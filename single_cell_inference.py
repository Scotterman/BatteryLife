import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from models import CPTransformer  # update if using another model
from accelerate import load_checkpoint_in_model
from data_provider.data_factory import data_provider_evaluate
from transformers import AutoTokenizer

# === STEP 1: Choose Checkpoint Folder ===
def select_model_checkpoint(checkpoint_dir="checkpoints"):
    dirs = sorted([d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))])
    if not dirs:
        raise FileNotFoundError("No checkpoints found.")
    
    print("Available saved models:")
    for idx, d in enumerate(dirs):
        print(f"[{idx}] {d}")
    
    selection = int(input("Select a model by number: "))
    chosen = os.path.join(checkpoint_dir, dirs[selection])
    print(f"Using model: {dirs[selection]}")
    return chosen

# === STEP 2: Load Config and Model ===
args_path = select_model_checkpoint("checkpoints")
args_json = json.load(open(os.path.join(args_path, "args.json")))

class Args: pass
args = Args()
args.__dict__ = args_json

# Override for inference
args.eval_dataset = args.dataset
args.eval_cycle_min = None
args.eval_cycle_max = None
args.batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load scalers
label_scaler = joblib.load(os.path.join(args_path, 'label_scaler'))
life_class_scaler = joblib.load(os.path.join(args_path, 'life_class_scaler'))
std = np.sqrt(label_scaler.var_[-1])
mean = label_scaler.mean_[-1]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert', trust_remote_code=True)
if tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# === STEP 3: Load Full Test Dataset ===
test_data, _ = data_provider_evaluate(
    args, flag='test',
    tokenizer=tokenizer,
    label_scaler=label_scaler,
    eval_cycle_min=None,
    eval_cycle_max=None,
    life_class_scaler=life_class_scaler
)

# Load model
model = CPTransformer.Model(args).float().to(device)
load_checkpoint_in_model(model, args_path)
model.eval()

# === LOOP: Prompt for battery index repeatedly ===
samples_per_battery = 100
max_battery_idx = len(test_data) // samples_per_battery - 1

while True:
    try:
        battery_index = int(input(f"\nEnter battery index to evaluate (0–{max_battery_idx}): "))
        if not (0 <= battery_index <= max_battery_idx):
            print(f"Invalid index. Please enter a number between 0 and {max_battery_idx}.")
            continue
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        continue

    start_idx = battery_index * samples_per_battery
    end_idx = start_idx + samples_per_battery

    predictions = []
    ground_truths = []

    for i in range(start_idx, end_idx):
        sample = test_data[i]
        X = sample['cycle_curve_data'].unsqueeze(0).float().to(device)
        mask = sample['curve_attn_mask'].unsqueeze(0).float().to(device)
        y = torch.tensor(sample['labels']).float().to(device)

        with torch.no_grad():
            pred = model(X, mask)

        pred = pred.cpu().numpy().flatten()[0] * std + mean
        y = y.cpu().numpy().flatten()[0] * std + mean
        predictions.append(pred)
        ground_truths.append(y)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(ground_truths, label='Ground Truth', linewidth=2)
    plt.plot(predictions, label='Prediction', linestyle='--')
    plt.title(f'Battery {battery_index} – Predicted vs Ground Truth (100 Cycles)')
    plt.xlabel('Cycle')
    plt.ylabel('State of Health / Capacity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(args_path, f'battery_{battery_index}_prediction_plot.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Saved plot to: {save_path}")

    # Prompt to repeat or exit
    repeat = input("\nWould you like to evaluate another battery? (y/n): ").strip().lower()
    if repeat != 'y':
        print("Exiting inference.")
        break
