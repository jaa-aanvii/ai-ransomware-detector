import torch

BASE = "models/proper_base_lstm.pth"
FINE = "models/proper_finetuned_lstm.pth"

base = torch.load(BASE, map_location="cpu")
fine = torch.load(FINE, map_location="cpu")

total_difference = 0.0
changed_parameters = 0

for key in base:

    difference = torch.abs(
        base[key] - fine[key]
    ).sum().item()

    total_difference += difference

    if difference > 0:
        changed_parameters += 1

print("================================")
print("FINE-TUNING VERIFICATION")
print("================================")

print(
    f"Parameters changed: "
    f"{changed_parameters}/{len(base)}"
)

print(
    f"Total weight difference: "
    f"{total_difference:.8f}"
)

if total_difference > 0:
    print("\nFine-tuning SUCCESSFUL.")
    print("Model weights changed after fine-tuning.")
else:
    print("\nWARNING: Model weights did not change.")