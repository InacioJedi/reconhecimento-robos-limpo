# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import numpy as np
from glob import glob
from pathlib import Path
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

# Mapeamento de rótulos
LABEL_MAP = {"A": 0, "B": 1, "none": 2}

class HitDataset(Dataset):
    def __init__(self, root="clips"):
        self.samples = []
        for path in glob(f"{root}/positives/*.npy"):
            lbl = Path(path).stem.split("_")[0]
            self.samples.append((path, LABEL_MAP.get(lbl, LABEL_MAP["none"])))
        for path in glob(f"{root}/negatives/*.npy"):
            self.samples.append((path, LABEL_MAP["none"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        clip = np.load(path)               # (T,H,W,3)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0  # (C,T,H,W)
        if lbl == LABEL_MAP["B"] and random.random() < 0.5:
            clip = clip.flip(-1)
        return clip, lbl

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        logp = nn.functional.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        ce = nn.functional.nll_loss(logp, targets, weight=self.weight, reduction="none")
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

def build_model(num_classes=len(LABEL_MAP)):
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(num_epochs=20, lr=1e-4, batch_size=4, val_split=0.2, num_workers=2):
    print("🚀 Iniciando treino (R3D-18)…", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = HitDataset("clips")
    idxs = list(range(len(full_ds)))
    labels = [full_ds.samples[i][1] for i in idxs]
    counts = Counter(labels)

    if min(counts.values() or [0]) < 2:
        train_idx, val_idx = train_test_split(idxs, test_size=val_split, random_state=42)
    else:
        train_idx, val_idx = train_test_split(idxs, test_size=val_split, stratify=labels, random_state=42)

    print(f"Total: {len(full_ds)}  Train: {len(train_idx)}  Val: {len(val_idx)}")
    print(f"Distribuição global: {counts}")
    print(f"Train: {Counter(labels[i] for i in train_idx)}")
    print(f" Val : {Counter(labels[i] for i in val_idx)}", flush=True)

    train_labels = [labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    num_train = len(train_labels)
    class_weights = {cls: num_train / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_train, replacement=True)

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    weight_list = [1.0 / class_counts.get(i, 1) for i in range(len(LABEL_MAP))]
    weights_tensor = torch.tensor(weight_list, dtype=torch.float, device=device)
    criterion = FocalLoss(gamma=2.0, weight=weights_tensor)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        tloss = tcorrect = 0
        for clips, lbls in train_loader:
            clips, lbls = clips.to(device), lbls.to(device)
            logits = model(clips)
            loss = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item() * lbls.size(0)
            tcorrect += (logits.argmax(1) == lbls).sum().item()
        train_loss = tloss / len(train_loader.dataset)
        train_acc = tcorrect / len(train_loader.dataset)

        model.eval()
        vloss = vcorrect = 0
        with torch.no_grad():
            for clips, lbls in val_loader:
                clips, lbls = clips.to(device), lbls.to(device)
                logits = model(clips)
                loss = criterion(logits, lbls)
                vloss += loss.item() * lbls.size(0)
                vcorrect += (logits.argmax(1) == lbls).sum().item()
        val_loss = vloss / len(val_loader.dataset)
        val_acc = vcorrect / len(val_loader.dataset)

        print(f"Epoch {epoch}/{num_epochs} | Train loss={train_loss:.4f}, acc={train_acc:.2%} | Val loss={val_loss:.4f}, acc={val_acc:.2%}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"✅ Concluído. Melhor Val Acc = {best_val_acc:.2%}", flush=True)

    # Relatório
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, lbls in val_loader:
            clips, lbls = clips.to(device), lbls.to(device)
            preds = model(clips).argmax(1)
            all_preds += preds.cpu().tolist()
            all_labels += lbls.cpu().tolist()

    print("\nClassification Report (val set):")
    print(classification_report(
        all_labels, all_preds,
        labels=list(LABEL_MAP.values()),
        target_names=list(LABEL_MAP.keys()),
        zero_division=0
    ))

if __name__ == "__main__":
    train(num_epochs=20, lr=1e-4, batch_size=4, val_split=0.2, num_workers=2)
