import os, json, numpy as np, pandas as pd, torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from model_architectures import ResNet1D, StandardCNN1D, CNNBiLSTM1D

# Load manifest and build label mapping
manifest_path = os.path.join('dataset_manifest_labeled.csv')
manifest = pd.read_csv(manifest_path)
labels = sorted(manifest['label'].unique())
label_to_id = {lbl: idx for idx, lbl in enumerate(labels)}
id_to_label = {idx: lbl for lbl, idx in label_to_id.items()}

# Load test split
test_npz = np.load(os.path.join('processed', 'test.npz'))
X_test = torch.tensor(test_npz['X'], dtype=torch.float32)
y_true = torch.tensor(test_npz['y'], dtype=torch.long).numpy()

# Load model checkpoint
ckpt_path = os.path.join('models', 'best_model.pth')
state = torch.load(ckpt_path, map_location='cpu')

# Check which architecture state dict belongs to
if 'lstm.weight_ih_l0' in state:
    model = CNNBiLSTM1D(num_classes=len(labels))
elif 'features.0.weight' in state:
    model = StandardCNN1D(num_classes=len(labels))
else:
    model = ResNet1D(num_classes=len(labels))

model.load_state_dict(state)
model.eval()

with torch.no_grad():
    logits = model(X_test)
    y_pred = torch.argmax(logits, dim=1).numpy()

# Compute metrics
num_classes = len(labels)
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, labels=list(range(num_classes)), average='macro', zero_division=0)
bal_acc = balanced_accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
recall = recall_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)

# Assertions
assert cm.sum() == len(y_true), f"Confusion matrix sum {cm.sum()} != test size {len(y_true)}"
assert np.isclose(accuracy, np.trace(cm) / cm.sum()), "Accuracy does not equal diagonal proportion"

# Save outputs
np.savetxt('confusion_matrix.txt', cm, fmt='%d')
metrics = {
    'test_size': int(cm.sum()),
    'accuracy': float(accuracy),
    'macro_f1': float(macro_f1),
    'balanced_accuracy': float(bal_acc),
    'precision_per_class': [float(p) for p in precision],
    'recall_per_class': [float(r) for r in recall],
    'class_labels': [id_to_label[i] for i in range(num_classes)],
    'confusion_matrix': cm.tolist()
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('Metrics and confusion matrix successfully written to metrics.json and confusion_matrix.txt')
print(f'Test size: {len(y_true)}, Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, Balanced Acc: {bal_acc:.4f}')
print('Confusion matrix:')
print(cm)
