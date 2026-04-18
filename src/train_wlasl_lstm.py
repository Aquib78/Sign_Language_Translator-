import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import pickle

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# PATHS
# =========================
USER_DATA_PATH  = "data"          # your own recorded samples
EXTRA_DATA_PATH = "filtered_data" # WLASL / LSA64 / any extra dataset

# How many augmented copies to generate per sample.
# User data gets more copies because it matches your exact signing style
# and recording conditions — the model should be biased toward it.
USER_COPIES  = 8
EXTRA_COPIES = 4

# =========================
# NORMALIZATION
# =========================
def normalize_landmarks(sequence):
    norm_seq = []
    for frame in sequence:
        frame = np.array(frame).reshape(2, 21, 3)
        new_frame = []
        for hand in frame:
            if np.sum(hand) == 0:
                new_frame.extend(hand.flatten())
                continue
            base    = hand[0]
            hand    = hand - base
            max_val = np.max(np.abs(hand))
            if max_val != 0:
                hand = hand / max_val
            new_frame.extend(hand.flatten())
        norm_seq.append(new_frame)
    return np.array(norm_seq)

# =========================
# LOAD DATASET
# =========================
def load_dataset(base_path):
    X, y = [], []
    if not os.path.isdir(base_path):
        print(f"  Warning: path not found — {base_path}")
        return np.array([]), np.array([])

    for label in sorted(os.listdir(base_path)):
        class_path = os.path.join(base_path, label)
        if not os.path.isdir(class_path):
            continue
        for file in sorted(os.listdir(class_path)):
            if not file.endswith(".npy"):
                continue
            try:
                seq = np.load(os.path.join(class_path, file))
            except Exception as e:
                print(f"  Warning: skipping {file}: {e}")
                continue
            if seq.ndim != 2 or seq.shape[1] != 126:
                print(f"  Warning: bad shape {seq.shape} in {file} — skipping.")
                continue
            seq = normalize_landmarks(seq)
            X.append(seq)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)

# =========================
# AUGMENTATION
# =========================
def augment_sequence(seq):
    aug = seq.copy()

    if np.random.rand() < 0.8:
        aug += np.random.normal(0, 0.015, aug.shape)

    if np.random.rand() < 0.6:
        shift = np.random.randint(-4, 5)
        aug   = np.roll(aug, shift, axis=0)
        if shift > 0:
            aug[:shift] = aug[shift]
        elif shift < 0:
            aug[shift:] = aug[shift - 1]

    if np.random.rand() < 0.5:
        n      = len(aug)
        warp   = np.random.uniform(0.85, 1.15)
        new_n  = max(5, min(int(n * warp), n * 2))
        idx_fw = np.linspace(0, n - 1, new_n).astype(int)
        warped = aug[idx_fw]
        idx_bk = np.linspace(0, len(warped) - 1, n).astype(int)
        aug    = warped[idx_bk]

    if np.random.rand() < 0.5:
        aug *= np.random.uniform(0.9, 1.1)

    return aug


def build_augmented_set(X, y, copies, include_original=True):
    """
    Build an augmented training set.

    FIX: the original code used `if seq in X_train_user` to decide
    copy count. That numpy comparison is undefined/ambiguous on 2D arrays
    and silently applied the wrong copy count to everything.

    The correct approach is to pass `copies` explicitly per dataset so
    there is no ambiguity — user data calls this function with copies=8,
    extra data calls it with copies=4.

    include_original=True always keeps the unaugmented sample, ensuring
    the model sees clean data alongside the augmented versions.
    """
    X_out, y_out = [], []
    for seq, label in zip(X, y):
        if include_original:
            X_out.append(seq)
            y_out.append(label)
        n_aug = copies - 1 if include_original else copies
        for _ in range(n_aug):
            X_out.append(augment_sequence(seq))
            y_out.append(label)
    return np.array(X_out, dtype=np.float32), np.array(y_out)

# =========================
# LOAD
# =========================
X_user,  y_user  = load_dataset(USER_DATA_PATH)
X_extra, y_extra = load_dataset(EXTRA_DATA_PATH)

print(f"User samples  : {len(X_user)}")
print(f"Extra samples : {len(X_extra)}")

# =========================
# COMBINED ENCODER
# Fit on ALL labels from both datasets before splitting anything,
# so both train and test always share the same label space.
# =========================
all_labels = np.concatenate([y_user, y_extra]) if len(y_extra) else y_user
encoder    = LabelEncoder()
encoder.fit(all_labels)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print(f"\nClasses ({len(encoder.classes_)}): {list(encoder.classes_)}")

y_user_enc  = encoder.transform(y_user)
y_extra_enc = encoder.transform(y_extra) if len(y_extra) else np.array([])

# Class distribution
dist = Counter(all_labels)
print("\nClass distribution (total):")
for cls, cnt in sorted(dist.items()):
    print(f"  {cls:20s}  {cnt:4d}")

# =========================
# SPLIT
# Test set is drawn ONLY from user data — it reflects your real
# signing style, which is what inference will see.
# Extra dataset goes entirely into training.
# =========================
X_train_user, X_test, y_train_user, y_test = train_test_split(
    X_user, y_user_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_user_enc
)

print(f"\nUser train   : {len(X_train_user)}")
print(f"Extra train  : {len(X_extra)}")
print(f"Test (clean) : {len(X_test)}")

# =========================
# AUGMENT — separately per source so copy counts are explicit and correct
# =========================
X_aug_user,  y_aug_user  = build_augmented_set(
    X_train_user, y_train_user, copies=USER_COPIES,  include_original=True
)
X_aug_extra, y_aug_extra = build_augmented_set(
    X_extra, y_extra_enc, copies=EXTRA_COPIES, include_original=True
) if len(X_extra) else (np.empty((0, *X_user.shape[1:]), dtype=np.float32), np.array([]))

X_train = np.concatenate([X_aug_user, X_aug_extra]) if len(X_aug_extra) else X_aug_user
y_train = np.concatenate([y_aug_user, y_aug_extra]) if len(y_aug_extra) else y_aug_user

# Shuffle combined set so batches are not all-user or all-extra
perm    = np.random.permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]

print(f"Train after augmentation : {len(X_train)}")

# =========================
# TENSORS + DATALOADER
# =========================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# =========================
# MODEL — BiLSTM with Attention
# =========================
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attn    = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _  = self.lstm(x)
        weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(out * weights, dim=1)
        context = self.dropout(context)
        return self.fc(context)

num_classes = len(encoder.classes_)
model       = AttentionLSTM(126, 128, num_classes).to(device)
criterion   = nn.CrossEntropyLoss()
optimizer   = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=5, factor=0.5
)

print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# =========================
# TRAIN
# =========================
EPOCHS       = 50
PATIENCE     = 10
best_val_acc = 0.0
patience_ctr = 0
best_state   = None   # FIX: initialise to None — old code could fail if
                      # epoch 1 never set best_state before early stopping

print()
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_out = model(X_test_t.to(device))
        val_acc = (val_out.argmax(1) == y_test_t.to(device)).float().mean().item()

    scheduler.step(val_acc)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}  (best: {best_val_acc:.4f})")
            break

# =========================
# RESTORE BEST + EVAL
# =========================
if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    final_out  = model(X_test_t.to(device))
    final_preds = final_out.argmax(1).cpu().numpy()
    final_acc  = (final_preds == y_test_t.numpy()).mean()

print(f"\nFinal Test Accuracy: {final_acc*100:.2f}%")

# Per-class breakdown — shows exactly which signs are being confused
print("\nPer-class accuracy:")
for i, cls in enumerate(encoder.classes_):
    mask = y_test_t.numpy() == i
    if mask.sum() == 0:
        continue
    cls_acc = (final_preds[mask] == i).mean()
    bar = "█" * int(cls_acc * 20)
    print(f"  {cls:20s}  {cls_acc*100:5.1f}%  {bar}  ({mask.sum()} samples)")

torch.save(model.state_dict(), "model.pth")
print("\nModel saved!")