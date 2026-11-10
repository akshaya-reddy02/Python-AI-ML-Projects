import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

base_dir = Path(__file__).resolve().parent
metadata_path = base_dir / "models" / "model_metadata.json"
output_path = base_dir / "reports" / "confusion_matrix.png"

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

counts = metadata["final_test_confusion_matrix"]

tn = int(counts["true_negatives"])
fp = int(counts["false_positives"])
fn = int(counts["false_negatives"])
tp = int(counts["true_positives"])

matrix = np.array([[tn, fp], [fn, tp]])

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Legitimate", "Fraudulent"]
)

fig, ax = plt.subplots(figsize=(8, 6))
display.plot(ax=ax, values_format="d", colorbar=False)

ax.set_title(
    f"Final Test Confusion Matrix "
    f"(Threshold = {metadata['decision_threshold']})"
)

fig.tight_layout()
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nConfusion matrix image updated successfully!")
print(f"Threshold: {metadata['decision_threshold']}")
print(f"True Negatives: {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives: {tp:,}")
print(f"Image saved to: {output_path}")