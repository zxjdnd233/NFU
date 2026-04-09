import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import random
import argparse
import itertools
from collections import OrderedDict, Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, TensorDataset

# ============================================================
# 0. reproducibility & device
# ============================================================
RANDOMSEED = 28
torch.manual_seed(RANDOMSEED)
np.random.seed(RANDOMSEED)
random.seed(RANDOMSEED)
DU_SELECT_K = 30

parser = argparse.ArgumentParser()
# parser.add_argument("--seed", type=int, required=True, help="random seed")
# parser.add_argument("--du_k", type=int, default=60, help="number of boundary samples selected as Du")
# args = parser.parse_args()
#
# RANDOMSEED = args.seed
# DU_SELECT_K = args.du_k

RETRAIN_INIT_SEED = RANDOMSEED + 99999
# kept only for compatibility with older scripts
RETRAIN_SCHEDULE_SEED = RANDOMSEED + 888888


def reset_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


reset_all_seeds(RANDOMSEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_cuda():
    if device.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# 1. config (put all important hyperparameters here)
# ============================================================
num_clients = 3
classes = 10
FIRST_FOUR_PER_CLASS = 1000
FORGET_CLIENT_ID = 0
NUM_WORKERS = 0

# Phase1 FL training
rounds = 10
local_epochs = 1
batch_size = 128
FL_LR = 1e-3

# Global neighbor table / Du selection
E2LSH_K = 32
E2LSH_L = 16
NEIGHBOR_THRESHOLD = 80
DU_LOWER_MARGIN = 30          # only for optional logging; selection is still top-k just below threshold

# NFU (ours): neighbor-weighted projected GA
NFU_LR = 1e-6
NFU_EPOCHS = 1
NFU_NEIGHBOR_TOPK = 20
NFU_GRAD_CLIP_NORM = 500.0
NFU_EXCLUDE_OTHER_DU = True

# Direct GA
DIRECT_GA_LR = 1e-6
DIRECT_GA_EPOCHS = 5
DIRECT_GA_BATCH_SIZE = 8
DIRECT_GA_GRAD_CLIP_NORM = 500.0

# PGD-based FU (paper-aligned: sequential single-sample GA + projection to an L2 ball around the Phase1 model)
PGD_FU_LR = 1e-6
PGD_FU_EPOCHS = 5
PGD_FU_BATCH_SIZE = 8
PGD_FU_GRAD_CLIP_NORM = 500.0
PGD_FU_RADIUS = 1
POST_TRAIN_ROUNDS = 1

# SFU (paper-aligned sample-level adaptation)
# - target update uses surrogate L_ul(w)=1/L(w)
# - each retained client contributes one stochastic descent mini-batch gradient
# - server performs per-layer SVD with epsilon-coverage rank selection
SFU_LR = 1e-5
SFU_EPOCHS = 2
SFU_DU_BATCH_SIZE = 1
SFU_RETAIN_BATCH_SIZE = 128
SFU_GRAD_CLIP_NORM = 500.0
SFU_EPSILON = 0.95
SFU_SURROGATE_LOSS_EPS = 1e-12
SFU_BATCH_SAMPLING_SEED = RANDOMSEED + 5050

# MIA
MIA_EPOCHS = 10
MIA_LR = 1e-3

print("=" * 120)
print("[Config]")
print("=" * 120)
print(f"num_clients={num_clients}, classes={classes}, FIRST_FOUR_PER_CLASS={FIRST_FOUR_PER_CLASS}")
print(f"FORGET_CLIENT_ID={FORGET_CLIENT_ID}")
print(f"rounds={rounds}, local_epochs={local_epochs}, batch_size={batch_size}, FL_LR={FL_LR}")
print(f"E2LSH_K={E2LSH_K}, E2LSH_L={E2LSH_L}, NEIGHBOR_THRESHOLD={NEIGHBOR_THRESHOLD}, DU_SELECT_K={DU_SELECT_K}")
print(f"NFU_LR={NFU_LR}, NFU_EPOCHS={NFU_EPOCHS}, NFU_NEIGHBOR_MODE=ALL_NEIGHBORS, NFU_GRAD_CLIP_NORM={NFU_GRAD_CLIP_NORM}")
print(f"DIRECT_GA_LR={DIRECT_GA_LR}, DIRECT_GA_EPOCHS={DIRECT_GA_EPOCHS}, DIRECT_GA_BATCH_SIZE={DIRECT_GA_BATCH_SIZE}")
print(f"PGD_FU_LR={PGD_FU_LR}, PGD_FU_EPOCHS={PGD_FU_EPOCHS}, PGD_FU_BATCH_SIZE={PGD_FU_BATCH_SIZE}, PGD_FU_RADIUS={PGD_FU_RADIUS}, POST_TRAIN_ROUNDS={POST_TRAIN_ROUNDS}")
print(f"SFU_LR={SFU_LR}, SFU_EPOCHS={SFU_EPOCHS}, SFU_DU_BATCH_SIZE={SFU_DU_BATCH_SIZE}, SFU_RETAIN_BATCH_SIZE={SFU_RETAIN_BATCH_SIZE}, SFU_EPSILON={SFU_EPSILON}, SFU_SURROGATE_LOSS_EPS={SFU_SURROGATE_LOSS_EPS}")
print(f"MIA_EPOCHS={MIA_EPOCHS}, MIA_LR={MIA_LR}")
print(f"NUM_WORKERS={NUM_WORKERS}")
print(f"RANDOMSEED={RANDOMSEED}, RETRAIN_INIT_SEED={RETRAIN_INIT_SEED}")

# ============================================================
# 2. data
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_data_full = datasets.MNIST(root="./dataset", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./dataset", train=False, download=True, transform=transform)
train_data_all = train_data_full

selected_targets = train_data_full.targets.cpu().numpy()
selected_indices = list(range(len(train_data_all)))
per_class_total = len(train_data_all) // classes
print(f"[Info] use full training set: {len(train_data_all)} | per_class_total={per_class_total}")

class_to_localidx = {c: np.where(selected_targets == c)[0].tolist() for c in range(classes)}
for c in range(classes):
    np.random.shuffle(class_to_localidx[c])

# Split rule aligned with the user's current source code
# Mode A: if all clients can receive FIRST_FOUR_PER_CLASS per class, assign equally and leave the rest unused.
# Mode B: otherwise, the first num_clients-1 clients each receive FIRST_FOUR_PER_CLASS per class,
#         and the last client receives all remaining samples in that class.
required_per_class_for_all_clients = FIRST_FOUR_PER_CLASS * num_clients
can_fill_all_clients_equally = all(
    len(class_to_localidx[c]) >= required_per_class_for_all_clients
    for c in range(classes)
)

counts = np.zeros((classes, num_clients), dtype=int)
if can_fill_all_clients_equally:
    split_mode = "ALL_CLIENTS_EQUAL_10x1000"
    for c in range(classes):
        for k in range(num_clients):
            counts[c, k] = FIRST_FOUR_PER_CLASS
else:
    split_mode = "FIRST_N_MINUS_1_FIXED_LAST_TAKES_REST"
    for c in range(classes):
        fixed_need = FIRST_FOUR_PER_CLASS * (num_clients - 1)
        remain_c = len(class_to_localidx[c])
        if remain_c < fixed_need:
            raise ValueError(
                f"class {c} only has {remain_c} samples, not enough for first {num_clients-1} clients with {FIRST_FOUR_PER_CLASS} per class."
            )
        for k in range(num_clients - 1):
            counts[c, k] = FIRST_FOUR_PER_CLASS
        counts[c, num_clients - 1] = remain_c - fixed_need

print(f"[Info] split_mode = {split_mode}")
print(f"[Info] required_per_class_for_all_clients = {required_per_class_for_all_clients}")

client_indices = [[] for _ in range(num_clients)]
unused_indices = []
for c in range(classes):
    ptr = 0
    cls_list = class_to_localidx[c]
    for k in range(num_clients):
        take_k = counts[c, k]
        if take_k > 0:
            client_indices[k].extend(cls_list[ptr:ptr + take_k])
            ptr += take_k
    if ptr < len(cls_list):
        unused_indices.extend(cls_list[ptr:])

for k in range(num_clients):
    random.shuffle(client_indices[k])
    targets_k = selected_targets[client_indices[k]]
    cnt = Counter(targets_k.tolist())
    dist_str = ", ".join([f"{cls}:{cnt.get(cls, 0)}" for cls in range(classes)])
    print(f"[Client {k}] size={len(client_indices[k])} | class dist: {dist_str}")

print("[Info] counts matrix (rows=class, cols=client):")
print(counts)
unused_class_cnt = Counter(selected_targets[idx] for idx in unused_indices)
unused_dist_str = ", ".join([f"{cls}:{unused_class_cnt.get(cls, 0)}" for cls in range(classes)])
print(f"[Unused] size={len(unused_indices)} | class dist: {unused_dist_str}")

original_client_indices_backup_phase1 = [list(lst) for lst in client_indices]


def get_client_loader(client_id, batch_size=128, shuffle=False, num_workers=NUM_WORKERS):
    subset = Subset(train_data_all, client_indices[client_id])
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_client_loader_from_given_indices(client_indices_ref, client_id, batch_size=128, shuffle=False, num_workers=NUM_WORKERS):
    subset = Subset(train_data_all, client_indices_ref[client_id])
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# ============================================================
# 3. model utils
# ============================================================
class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7
        )

        self.fc_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x):
        x = self.conv_block(x)
        x = self.fc_feature(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.classifier(feat)
        return logits


def build_model():
    return MNISTCNN(num_classes=classes, feature_dim=128)


def build_fixed_retrain_init_state(seed):
    reset_all_seeds(seed)
    model = build_model().to(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    if total == 0:
        return float("nan"), float("nan")
    return 100.0 * correct / total, total_loss / max(1, len(loader))


def make_loader_from_indices(local_indices, batch_size=256, shuffle=False, num_workers=NUM_WORKERS):
    if local_indices is None or len(local_indices) == 0:
        return None
    return DataLoader(
        Subset(train_data_all, list(local_indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def local_train_and_report_loss(model, loader, epochs=1, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    num_steps = 0
    for _ in range(epochs):
        total_loss = 0.0
        steps = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        running_loss += (total_loss / max(1, steps))
        num_steps += 1
    return running_loss / max(1, num_steps)


def aggregate_fedavg(global_model, client_states, client_sizes):
    total = float(sum(client_sizes))
    new_state = OrderedDict()
    for k in client_states[0].keys():
        agg = None
        for i in range(len(client_states)):
            w = client_sizes[i] / total
            v = client_states[i][k] * w
            agg = v if agg is None else agg + v
        new_state[k] = agg
    global_model.load_state_dict(new_state)


# ============================================================
# 4. standard FL training / retraining (aligned with the user's current source code)
# ============================================================
def federated_train_standard(phase_tag, init_state=None, client_indices_ref=None, rounds=rounds, local_epochs=local_epochs, batch_size=batch_size, lr=FL_LR):
    if client_indices_ref is None:
        client_indices_ref = original_client_indices_backup_phase1

    if init_state is None:
        reset_all_seeds(RANDOMSEED)
        model_phase = build_model().to(device)
    else:
        reset_all_seeds(RETRAIN_INIT_SEED)
        model_phase = build_model().to(device)
        model_phase.load_state_dict(init_state)

    print(f"\n[FL-{phase_tag}] start federated training/retraining...")
    _sync_cuda()
    t0 = time.perf_counter()

    for rnd in range(1, rounds + 1):
        client_states = []
        sizes = []
        weighted_loss_sum = 0.0
        total_samples = 0

        for cid in range(num_clients):
            client_model = build_model().to(device)
            client_model.load_state_dict(model_phase.state_dict())

            train_loader = get_client_loader_from_given_indices(
                client_indices_ref=client_indices_ref,
                client_id=cid,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )
            avg_loss_client = local_train_and_report_loss(client_model, train_loader, epochs=local_epochs, lr=lr)

            n_i = len(client_indices_ref[cid])
            weighted_loss_sum += avg_loss_client * n_i
            total_samples += n_i

            client_states.append({k: v.detach().cpu() for k, v in client_model.state_dict().items()})
            sizes.append(n_i)

            del client_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aggregate_fedavg(model_phase, client_states, sizes)
        round_train_loss = weighted_loss_sum / max(1, total_samples)
        print(f"[{phase_tag}][Round {rnd}] Train Loss: {round_train_loss:.4f}")

    _sync_cuda()
    t1 = time.perf_counter()
    return model_phase, (t1 - t0)


global_model, p1_train_time = federated_train_standard(
    phase_tag="Phase1",
    init_state=None,
    client_indices_ref=original_client_indices_backup_phase1,
    rounds=rounds,
    local_epochs=local_epochs,
    batch_size=batch_size,
    lr=FL_LR,
)
print(f"[Phase1] training time: {p1_train_time:.6f} s")


# ============================================================
# 6. MIA fixed data / attack model
# ============================================================
mia_member_nominal = int(0.2 * len(train_data_all))
mia_member = min(mia_member_nominal, len(test_data))
if mia_member <= 0:
    raise ValueError(f"Invalid mia_member={mia_member}. len(test_data)={len(test_data)}.")
print(f"[MIA] nominal member/non-member size={mia_member_nominal}, effective size={mia_member}")

all_local_idx = set(range(len(train_data_all)))
client0_local = set(original_client_indices_backup_phase1[FORGET_CLIENT_ID])
member_pool = sorted(all_local_idx - client0_local)

rng_mia = np.random.default_rng(RANDOMSEED + 777)
mia_member_indices = rng_mia.choice(member_pool, mia_member, replace=False).tolist()
non_member_indices = rng_mia.choice(len(test_data), mia_member, replace=False).tolist()
assert set(mia_member_indices).isdisjoint(client0_local)

member_loader = DataLoader(Subset(train_data_all, mia_member_indices), batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
non_member_loader = DataLoader(Subset(test_data, non_member_indices), batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


@torch.no_grad()
def collect_attack_data(model, loader, label_value):
    model.eval()
    feats, labs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        losses = F.cross_entropy(logits, y, reduction="none")
        batch_feat = torch.cat([probs, losses.unsqueeze(1)], dim=1).cpu()
        feats.append(batch_feat)
        labs.append(torch.full((y.size(0),), label_value, dtype=torch.long))
    if len(feats) == 0:
        return None, None
    X = torch.cat(feats, dim=0)
    y = torch.cat(labs, dim=0)
    return X, y


class AttackModel(nn.Module):
    def __init__(self, input_size=11):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(1)


def build_attack_features_for_target(target_model, member_loader, non_member_loader):
    X_m, y_m = collect_attack_data(target_model, member_loader, label_value=1)
    X_n, y_n = collect_attack_data(target_model, non_member_loader, label_value=0)
    X_a = torch.cat([X_m, X_n], dim=0)
    y_a = torch.cat([y_m, y_n], dim=0).long()
    return X_a, y_a


def _binary_acc_and_f1(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = 100.0 * float((y_true == y_pred).mean()) if len(y_true) > 0 else float("nan")
    tp = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
    fp = int(((y_true != positive_label) & (y_pred == positive_label)).sum())
    fn = int(((y_true == positive_label) & (y_pred != positive_label)).sum())
    denom = (2 * tp + fp + fn)
    f1 = 100.0 * (2 * tp / denom) if denom > 0 else 0.0
    return acc, f1


def eval_attack_three_metrics(attack_model, loader, name=""):
    attack_model.eval()
    y_true_all, y_pred_all = [], []
    correct_mem = total_mem = 0
    correct_non = total_non = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = (attack_model(xb) > 0.5).long()

            y_true_all.extend(yb.cpu().tolist())
            y_pred_all.extend(pred.cpu().tolist())

            mem_mask = (yb == 1)
            non_mask = (yb == 0)
            if mem_mask.any():
                correct_mem += (pred[mem_mask] == yb[mem_mask]).sum().item()
                total_mem += mem_mask.sum().item()
            if non_mask.any():
                correct_non += (pred[non_mask] == yb[non_mask]).sum().item()
                total_non += non_mask.sum().item()

    overall_acc, overall_f1 = _binary_acc_and_f1(y_true_all, y_pred_all, positive_label=1)
    mem_acc = 100.0 * correct_mem / max(1, total_mem) if total_mem > 0 else float("nan")
    non_acc = 100.0 * correct_non / max(1, total_non) if total_non > 0 else float("nan")

    if name:
        print(f"[{name}] Accuracy (Overall): {overall_acc:.2f}%")
        print(f"[{name}] F1      (Overall): {overall_f1:.2f}%")
        print(f"[{name}] Member (TP) Acc   : {mem_acc:.2f}%")
        print(f"[{name}] Non-Member (TN) Acc: {non_acc:.2f}%")
    return overall_acc, overall_f1, mem_acc, non_acc


def train_attack_model_with_fixed_split(X_attack_any, y_attack_any, train_idx, val_idx, epochs=10, lr=1e-3, tag="MIA"):
    attack_model_any = AttackModel(input_size=X_attack_any.shape[1]).to(device)
    opt = optim.Adam(attack_model_any.parameters(), lr=lr)
    crit = nn.BCELoss()

    train_loader = DataLoader(TensorDataset(X_attack_any[train_idx], y_attack_any[train_idx]), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_attack_any[val_idx], y_attack_any[val_idx]), batch_size=64, shuffle=False)

    print(f"\n[{tag}] train MIA attack model...")
    for ep in range(epochs):
        attack_model_any.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            opt.zero_grad()
            pred = attack_model_any(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[{tag}][Epoch {ep+1}] Train Loss: {total_loss / len(train_loader):.4f}")

    return attack_model_any, val_loader


def build_fixed_val_loader_from_features(X_attack_any, y_attack_any, val_idx, batch_size=64):
    return DataLoader(TensorDataset(X_attack_any[val_idx], y_attack_any[val_idx]), batch_size=batch_size, shuffle=False)


def collect_attack_data_for_named_indices(model, dataset, eval_indices, true_membership_label=None, true_membership_map=None, batch_size=256, num_workers=NUM_WORKERS):
    if eval_indices is None:
        return None, None
    ordered_indices = list(eval_indices)
    if len(ordered_indices) == 0:
        return None, None
    if true_membership_label is None and true_membership_map is None:
        raise ValueError("Either true_membership_label or true_membership_map must be provided.")

    model.eval()
    loader = DataLoader(
        Subset(dataset, ordered_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feats, labs = [], []
    ptr = 0
    with torch.no_grad():
        for xb, yb_cls in loader:
            xb = xb.to(device)
            yb_cls = yb_cls.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            losses = F.cross_entropy(logits, yb_cls, reduction="none")
            batch_feat = torch.cat([probs, losses.unsqueeze(1)], dim=1).cpu()
            feats.append(batch_feat)

            batch_indices = ordered_indices[ptr: ptr + xb.size(0)]
            if true_membership_map is not None:
                labs.extend([int(true_membership_map[idx]) for idx in batch_indices])
            else:
                labs.extend([int(true_membership_label)] * len(batch_indices))
            ptr += xb.size(0)

    X = torch.cat(feats, dim=0)
    y = torch.tensor(labs, dtype=torch.long)
    return X, y


def eval_named_set_attack_acc_f1(target_model, attack_model, eval_indices, phase_tag="MIA-PhaseX", set_name="Eval Set", true_membership_label=None, true_membership_map=None):
    X_eval, y_eval = collect_attack_data_for_named_indices(
        model=target_model,
        dataset=train_data_all,
        eval_indices=eval_indices,
        true_membership_label=true_membership_label,
        true_membership_map=true_membership_map,
        batch_size=256,
        num_workers=NUM_WORKERS,
    )
    if X_eval is None or y_eval is None or len(y_eval) == 0:
        print(f"[{phase_tag}][{set_name}] Skip (empty set).")
        return float("nan"), float("nan")

    loader = DataLoader(TensorDataset(X_eval, y_eval), batch_size=64, shuffle=False)
    y_true_all, y_pred_all = [], []
    pred_member_cnt = 0
    attack_model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = (attack_model(xb) > 0.5).long().cpu()
            y_true_all.extend(yb.tolist())
            y_pred_all.extend(pred.tolist())
            pred_member_cnt += int((pred == 1).sum().item())

    unique_labels = sorted(set(int(v) for v in y_true_all))
    positive_label = unique_labels[0] if len(unique_labels) == 1 else 1
    acc, f1 = _binary_acc_and_f1(y_true_all, y_pred_all, positive_label=positive_label)
    pred_member_rate = 100.0 * pred_member_cnt / max(1, len(y_true_all))

    if len(unique_labels) == 1:
        gt_text = "Member" if unique_labels[0] == 1 else "Non-Member"
    else:
        num_mem = int(sum(1 for v in y_true_all if v == 1))
        num_non = int(sum(1 for v in y_true_all if v == 0))
        gt_text = f"Mixed(Member={num_mem}, Non-Member={num_non})"

    print(
        f"[{phase_tag}][{set_name}] Attack Acc: {acc:.2f}% | F1: {f1:.2f}% | "
        f"GT={gt_text} | Pred-Member Rate: {pred_member_rate:.2f}%"
    )
    return acc, f1


X_attack_p1, y_attack_p1 = build_attack_features_for_target(global_model, member_loader, non_member_loader)
attack_all_idx = np.arange(len(y_attack_p1))
y_attack_np = y_attack_p1.detach().cpu().numpy()
attack_train_idx, attack_val_idx = train_test_split(attack_all_idx, test_size=0.2, random_state=42, stratify=y_attack_np)
attack_train_idx = torch.from_numpy(np.array(attack_train_idx)).long()
attack_val_idx = torch.from_numpy(np.array(attack_val_idx)).long()

attack_model_p1, attack_val_loader_p1 = train_attack_model_with_fixed_split(
    X_attack_p1, y_attack_p1, attack_train_idx, attack_val_idx, epochs=MIA_EPOCHS, lr=MIA_LR, tag="MIA-Phase1"
)
eval_attack_three_metrics(attack_model_p1, attack_val_loader_p1, name="MIA-Phase1")


# ============================================================
# 7. shared feature precompute for global neighbor table
# ============================================================
FEATURE_BACKBONE = global_model


class CNNFeature(nn.Module):
    def __init__(self, cnn_model):
        super().__init__()
        self.cnn_model = cnn_model

    def forward(self, x):
        return self.cnn_model.forward_features(x)

@torch.no_grad()
def precompute_client_features(client_id, feature_net, batch_size=256, normalize=True):
    feature_net.eval()
    loader = DataLoader(
        Subset(train_data_all, original_client_indices_backup_phase1[client_id]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    local_order = original_client_indices_backup_phase1[client_id]
    ptr = 0

    feats_chunks = []
    labels_all = []
    local_idx_all = []

    for xb, yb in loader:
        B = xb.size(0)
        xb = xb.to(device)
        yb = yb.to(device)
        feats = feature_net(xb)
        if normalize:
            feats = F.normalize(feats, p=2, dim=1)
        feats_chunks.append(feats)
        labels_all.extend(yb.tolist())
        local_idx_all.extend(local_order[ptr:ptr + B])
        ptr += B

    feats_all = torch.cat(feats_chunks, dim=0)
    labels_np = np.array(labels_all, dtype=np.int64)
    local_idx_np = np.array(local_idx_all, dtype=np.int64)
    return feats_all, labels_all, local_idx_all, labels_np, local_idx_np


@torch.no_grad()
def server_make_coslsh_params(L=8, K=16, d=128, seed=2025, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    params = []
    for _ in range(L):
        R = torch.randn((K, d), generator=g, device=device)
        params.append(R)
    return params


def client_hash_and_build_tables_from_cached_features(feats_all, labels_all, local_idx_all, R_list, hash_batch=2048):
    N = feats_all.shape[0]
    L = len(R_list)
    codes_L = [[] for _ in range(L)]
    local_tables_L = [defaultdict(Counter) for _ in range(L)]
    local_bucket_indices_L = [defaultdict(lambda: defaultdict(list)) for _ in range(L)]

    for ell, R in enumerate(R_list):
        for st in range(0, N, hash_batch):
            ed = min(N, st + hash_batch)
            feats_b = feats_all[st:ed]
            proj = feats_b @ R.t()
            h_cpu = (proj >= 0).to(torch.uint8).cpu().numpy()
            for i in range(h_cpu.shape[0]):
                code_str = "|".join(map(str, h_cpu[i].tolist()))
                codes_L[ell].append(code_str)
        for code, lab, loc_idx in zip(codes_L[ell], labels_all, local_idx_all):
            local_tables_L[ell][int(lab)][code] += 1
            local_bucket_indices_L[ell][int(lab)][code].append(int(loc_idx))

    return codes_L, local_tables_L, local_bucket_indices_L


def server_aggregate_tables_multi(local_tables_L_all_clients, local_bucket_indices_all_clients):
    L = len(local_tables_L_all_clients[0])
    global_tables_L = [defaultdict(Counter) for _ in range(L)]
    global_bucket_indices_L = [defaultdict(lambda: defaultdict(list)) for _ in range(L)]

    for tables_L, bucket_L in zip(local_tables_L_all_clients, local_bucket_indices_all_clients):
        for ell in range(L):
            local_tbl = tables_L[ell]
            for cls, counter in local_tbl.items():
                global_tables_L[ell][cls].update(counter)
            local_bucket = bucket_L[ell]
            for cls, code_map in local_bucket.items():
                for code, idx_list in code_map.items():
                    global_bucket_indices_L[ell][cls][code].extend(idx_list)
    return global_tables_L, global_bucket_indices_L


def neighbors_in_table_sameclass(code_str, label, global_table_ell, subtract_self=True):
    cls = int(label)
    cnt = global_table_ell[cls].get(code_str, 0)
    if subtract_self and cnt > 0:
        cnt -= 1
    return cnt


def build_global_neighbor_book(client_cached, seed=2025):
    cos_params = server_make_coslsh_params(L=E2LSH_L, K=E2LSH_K, d=128, seed=seed, device=device)

    _sync_cuda()
    t0 = time.perf_counter()

    local_tables_all_clients = []
    local_bucket_indices_all_clients = []
    client_books = {}
    local_idx_to_pos = {}
    for cid in range(num_clients):
        feats_all = client_cached[cid]["feats"]
        labels_all = client_cached[cid]["labels"]
        local_idx_all = client_cached[cid]["local_idx"]
        codes_L, local_tables_L, local_bucket_indices_L = client_hash_and_build_tables_from_cached_features(
            feats_all, labels_all, local_idx_all, cos_params, hash_batch=2048
        )
        local_tables_all_clients.append(local_tables_L)
        local_bucket_indices_all_clients.append(local_bucket_indices_L)
        client_books[cid] = {"codes_L": codes_L, "labels": labels_all, "local_idx": local_idx_all}
        local_idx_to_pos[cid] = {int(idx): pos for pos, idx in enumerate(local_idx_all)}

    global_tables_L, global_bucket_indices_L = server_aggregate_tables_multi(local_tables_all_clients, local_bucket_indices_all_clients)

    _sync_cuda()
    t1 = time.perf_counter()
    return {
        "client_books": client_books,
        "global_tables_L": global_tables_L,
        "global_bucket_indices_L": global_bucket_indices_L,
        "local_idx_to_pos": local_idx_to_pos,
        "build_time": t1 - t0,
    }


def compute_avg_neighbors_for_indices(neighbor_book, client_id, target_indices):
    client_book = neighbor_book["client_books"][client_id]
    global_tables_L = neighbor_book["global_tables_L"]
    pos_map = neighbor_book["local_idx_to_pos"][client_id]

    avg_neighbors = {}
    for loc_idx in target_indices:
        pos = pos_map[int(loc_idx)]
        lab = int(client_book["labels"][pos])
        counts = []
        for ell in range(E2LSH_L):
            code_str = client_book["codes_L"][ell][pos]
            counts.append(neighbors_in_table_sameclass(code_str, lab, global_tables_L[ell], subtract_self=True))
        avg_neighbors[int(loc_idx)] = float(np.mean(counts))
    return avg_neighbors


def select_boundary_du(avg_neighbors, candidate_indices, k, threshold):
    just_below = [idx for idx in candidate_indices if avg_neighbors[idx] < threshold]
    ranked = sorted(just_below, key=lambda idx: (-avg_neighbors[idx], int(idx)))
    selected = ranked[: min(k, len(ranked))]
    return selected


def build_du_neighbor_map(neighbor_book, du_indices, client_id=FORGET_CLIENT_ID, topk=None, exclude_other_du=True):
    """
    Full-neighbor version:
    - collect all valid neighbors across all L hash tables
    - weight of a neighbor = collision_count / sum_of_all_collision_counts
    This makes neighbor_weights sum to 1 over all retained neighbors for each Du.
    The argument `topk` is kept only for interface compatibility and is ignored.
    """
    client_book = neighbor_book["client_books"][client_id]
    pos_map = neighbor_book["local_idx_to_pos"][client_id]
    global_bucket_indices_L = neighbor_book["global_bucket_indices_L"]
    du_set = set(int(x) for x in du_indices)

    neighbor_map = {}
    for du_idx in du_indices:
        pos = pos_map[int(du_idx)]
        lab = int(client_book["labels"][pos])
        collision_counter = Counter()

        for ell in range(E2LSH_L):
            code_str = client_book["codes_L"][ell][pos]
            same_bucket = global_bucket_indices_L[ell][lab].get(code_str, [])
            for cand_idx in same_bucket:
                cand_idx = int(cand_idx)
                if cand_idx == int(du_idx):
                    continue
                if exclude_other_du and cand_idx in du_set:
                    continue
                collision_counter[cand_idx] += 1

        ranked = sorted(collision_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        n_indices = [idx for idx, _ in ranked]
        n_scores = np.array([cnt for _, cnt in ranked], dtype=np.float32)
        total_collision_mass = float(n_scores.sum()) if len(n_scores) > 0 else 0.0
        if total_collision_mass > 1e-12:
            n_weights = (n_scores / total_collision_mass).tolist()
        else:
            n_weights = []
        neighbor_map[int(du_idx)] = {
            "neighbor_indices": n_indices,
            "collision_counts": [int(cnt) for _, cnt in ranked],
            "neighbor_weights": n_weights,
            "num_neighbors": len(n_indices),
            "total_collision_mass": total_collision_mass,
        }
    return neighbor_map


print("\n[Shared Precompute] forward embedding + L2 normalize for all clients ...")
feature_net = CNNFeature(FEATURE_BACKBONE).to(device)
client_cached = {}
for cid in range(num_clients):
    feats_all, labels_all, local_idx_all, labels_np, local_idx_np = precompute_client_features(cid, feature_net, batch_size=256, normalize=True)
    client_cached[cid] = {
        "feats": feats_all,
        "labels": labels_all,
        "local_idx": local_idx_all,
        "labels_np": labels_np,
        "local_idx_np": local_idx_np,
    }

print("\n[Neighbor Table] build global hash tables / buckets ...")
neighbor_book = build_global_neighbor_book(client_cached=client_cached, seed=2025)
print(f"[Neighbor Table] build_time = {neighbor_book['build_time']:.6f} s")

client0_candidate_indices = list(original_client_indices_backup_phase1[FORGET_CLIENT_ID])
avg_neighbors_client0 = compute_avg_neighbors_for_indices(neighbor_book, client_id=FORGET_CLIENT_ID, target_indices=client0_candidate_indices)

boundary_du_indices = select_boundary_du(
    avg_neighbors=avg_neighbors_client0,
    candidate_indices=client0_candidate_indices,
    k=DU_SELECT_K,
    threshold=NEIGHBOR_THRESHOLD,
)
if len(boundary_du_indices) < DU_SELECT_K:
    print(f"[Warning] only {len(boundary_du_indices)} client-0 samples are below threshold={NEIGHBOR_THRESHOLD}; DU_SELECT_K={DU_SELECT_K} cannot be fully satisfied.")

du_indices = sorted(boundary_du_indices)
du_loader = make_loader_from_indices(du_indices, batch_size=256, shuffle=False)
remaining_excluding_du_indices = sorted(list(set(range(len(train_data_all))) - set(du_indices)))
remaining_excluding_du_loader = make_loader_from_indices(remaining_excluding_du_indices, batch_size=256, shuffle=False)
retain_other_clients_indices = sorted(list(set(range(len(train_data_all))) - set(original_client_indices_backup_phase1[FORGET_CLIENT_ID])))

print("\n" + "=" * 120)
print(f"[Du Selection] top-{len(du_indices)} client-0 samples with avg_neighbors just below threshold={NEIGHBOR_THRESHOLD}")
print("=" * 120)
print("rank\tcifar10_idx\tlabel\tavg_neighbors")
for rank, idx in enumerate(sorted(du_indices, key=lambda x: (-avg_neighbors_client0[x], x)), start=1):
    print(f"{rank}\t{idx}\t{int(selected_targets[idx])}\t{avg_neighbors_client0[idx]:.6f}")
print(f"[Du][CIFAR10 Real Idx List] = {du_indices}")
print(f"[Dr] size = {len(remaining_excluding_du_indices)}")

neighbor_map_for_du = build_du_neighbor_map(
    neighbor_book=neighbor_book,
    du_indices=du_indices,
    client_id=FORGET_CLIENT_ID,
    topk=NFU_NEIGHBOR_TOPK,
    exclude_other_du=NFU_EXCLUDE_OTHER_DU,
)


# ============================================================
# 8. baseline evaluations on Phase1 model
# ============================================================
acc_test1, loss_test1 = evaluate(global_model, test_loader)
acc_du1, loss_du1 = evaluate(global_model, du_loader)
acc_dr1, loss_dr1 = evaluate(global_model, remaining_excluding_du_loader)
print(f"\n[Phase1][Test All] Accuracy: {acc_test1:.2f}% | Avg Loss: {loss_test1:.4f}")
print(f"[Phase1][Dr]       Accuracy: {acc_dr1:.2f}% | Avg Loss: {loss_dr1:.4f}")
print(f"[Phase1][Du]       Accuracy: {acc_du1:.2f}% | Avg Loss: {loss_du1:.4f}")
eval_named_set_attack_acc_f1(
    target_model=global_model,
    attack_model=attack_model_p1,
    eval_indices=du_indices,
    true_membership_label=1,
    phase_tag="MIA-Phase1",
    set_name="Du",
)


# ============================================================
# 9. common helpers for oracle retrain + fast unlearning methods
# ============================================================
def metric_dict(acc, loss):
    return {"acc": float(acc), "loss": float(loss)}


def mia_metric_dict(acc, f1):
    return {"acc": float(acc), "f1": float(f1)}


def evaluate_loader_with_log(target_model, loader, phase_tag, set_name):
    if loader is None:
        print(f"[{phase_tag}][{set_name}] Skip (empty set).")
        return metric_dict(float("nan"), float("nan"))
    acc, loss = evaluate(target_model, loader)
    print(f"[{phase_tag}][{set_name}] Accuracy: {acc:.2f}% | Avg Loss: {loss:.4f}")
    return metric_dict(acc, loss)


def evaluate_named_attack_with_log(target_model, attack_model, eval_indices, phase_tag, set_name, true_membership_label=None, true_membership_map=None):
    if eval_indices is None or len(eval_indices) == 0:
        print(f"[MIA-{phase_tag}][{set_name}] Skip (empty set).")
        return mia_metric_dict(float("nan"), float("nan"))
    acc, f1 = eval_named_set_attack_acc_f1(
        target_model=target_model,
        attack_model=attack_model,
        eval_indices=eval_indices,
        phase_tag=f"MIA-{phase_tag}",
        set_name=set_name,
        true_membership_label=true_membership_label,
        true_membership_map=true_membership_map,
    )
    return mia_metric_dict(acc, f1)


def evaluate_method(phase_tag, target_model, algo_time_s):
    metrics = OrderedDict()
    metrics["test"] = evaluate_loader_with_log(target_model, test_loader, phase_tag, "Dt(Test All)")
    metrics["dr"] = evaluate_loader_with_log(target_model, remaining_excluding_du_loader, phase_tag, "Dr")
    metrics["du"] = evaluate_loader_with_log(target_model, du_loader, phase_tag, "Du")
    metrics["du_mia"] = evaluate_named_attack_with_log(
        target_model=target_model,
        attack_model=attack_model_p1,
        eval_indices=du_indices,
        phase_tag=phase_tag,
        set_name="Du",
        true_membership_label=0,
    )
    metrics["time_s"] = float(algo_time_s)
    print(f"[{phase_tag}] Algorithm Time: {algo_time_s:.6f} s")
    return metrics


def abs_delta(a, b):
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return abs(float(a) - float(b))


def print_model_metric_line(prefix, metric_item):
    print(f"{prefix} Accuracy: {metric_item['acc']:.2f}% | Avg Loss: {metric_item['loss']:.4f}")


def print_mia_metric_line(prefix, metric_item):
    print(f"{prefix} Attack Acc: {metric_item['acc']:.2f}% | F1: {metric_item['f1']:.2f}%")


def print_comparison_vs_oracle(method_name, oracle_metrics, method_metrics):
    print("\n" + "-" * 120)
    print(f"[Comparison vs Oracle Retrain] {method_name}")
    print("-" * 120)
    print(f"[Oracle Time] {oracle_metrics['time_s']:.6f} s | [{method_name} Time] {method_metrics['time_s']:.6f} s")
    print_model_metric_line("Oracle[Dt] ", oracle_metrics["test"])
    print_model_metric_line("Oracle[Dr] ", oracle_metrics["dr"])
    print_model_metric_line("Oracle[Du] ", oracle_metrics["du"])
    print_mia_metric_line("Oracle[Du] ", oracle_metrics["du_mia"])
    print_model_metric_line(f"{method_name}[Dt] ", method_metrics["test"])
    print_model_metric_line(f"{method_name}[Dr] ", method_metrics["dr"])
    print_model_metric_line(f"{method_name}[Du] ", method_metrics["du"])
    print_mia_metric_line(f"{method_name}[Du] ", method_metrics["du_mia"])
    print(
        f"|Δ| Dt : Acc={abs_delta(oracle_metrics['test']['acc'], method_metrics['test']['acc']):.2f}, "
        f"Loss={abs_delta(oracle_metrics['test']['loss'], method_metrics['test']['loss']):.4f}"
    )
    print(
        f"|Δ| Dr : Acc={abs_delta(oracle_metrics['dr']['acc'], method_metrics['dr']['acc']):.2f}, "
        f"Loss={abs_delta(oracle_metrics['dr']['loss'], method_metrics['dr']['loss']):.4f}"
    )
    print(
        f"|Δ| Du : Acc={abs_delta(oracle_metrics['du']['acc'], method_metrics['du']['acc']):.2f}, "
        f"Loss={abs_delta(oracle_metrics['du']['loss'], method_metrics['du']['loss']):.4f}"
    )
    print(
        f"|Δ| Du-MIA : Acc={abs_delta(oracle_metrics['du_mia']['acc'], method_metrics['du_mia']['acc']):.2f}, "
        f"F1={abs_delta(oracle_metrics['du_mia']['f1'], method_metrics['du_mia']['f1']):.2f}"
    )


def make_phase_client_indices_from_removed_set(removed_set, phase_tag="PhaseX"):
    base_client_indices = [list(lst) for lst in original_client_indices_backup_phase1]
    phase_client_indices = []
    for cid in range(num_clients):
        if cid == FORGET_CLIENT_ID:
            kept = [idx for idx in base_client_indices[cid] if idx not in removed_set]
            phase_client_indices.append(kept)
            print(f"[{phase_tag}] client{cid} removed {len(base_client_indices[cid]) - len(kept)} | remain {len(kept)}")
        else:
            phase_client_indices.append(list(base_client_indices[cid]))
    return phase_client_indices


def clone_model_from(source_model):
    m = build_model().to(device)
    m.load_state_dict({k: v.detach().cpu() for k, v in source_model.state_dict().items()})
    return m


def fetch_samples_by_indices(dataset, indices):
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[int(idx)]
        xs.append(x)
        ys.append(y)
    if len(xs) == 0:
        return None, None
    xb = torch.stack(xs, dim=0).to(device)
    yb = torch.tensor(ys, dtype=torch.long, device=device)
    return xb, yb


def model_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _zero_grad_compat(model):
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()


def compute_grad_list(model, loss):
    _zero_grad_compat(model)
    loss.backward()
    grads = []
    for p in model_trainable_params(model):
        if p.grad is None:
            grads.append(torch.zeros_like(p))
        else:
            grads.append(p.grad.detach().clone())
    _zero_grad_compat(model)
    return grads


def grad_list_dot(g1, g2):
    return sum((a * b).sum() for a, b in zip(g1, g2))


def grad_list_norm(g):
    val = grad_list_dot(g, g)
    return float(torch.sqrt(torch.clamp(val, min=0.0)).item())


def maybe_clip_grad_list(g, max_norm=None):
    if max_norm is None:
        return [x.clone() for x in g]
    norm = grad_list_norm(g)
    if norm <= max_norm or norm <= 1e-12:
        return [x.clone() for x in g]
    scale = max_norm / norm
    return [x * scale for x in g]


def apply_grad_ascent_(model, grads, lr):
    with torch.no_grad():
        for p, g in zip(model_trainable_params(model), grads):
            p.add_(lr * g)


def project_grad_orthogonal_to_basis(grad, basis_list):
    out = [g.clone() for g in grad]
    for basis in basis_list:
        denom = grad_list_dot(basis, basis)
        denom_val = float(denom.item()) if torch.is_tensor(denom) else float(denom)
        if denom_val <= 1e-12:
            continue
        coeff = grad_list_dot(out, basis) / denom
        out = [o - coeff * b for o, b in zip(out, basis)]
    return out


def model_delta_l2_norm(model, ref_state_device):
    total = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            diff = param - ref_state_device[name]
            total += float((diff * diff).sum().item())
    return float(np.sqrt(total))


def project_model_to_l2_ball_(model, ref_state_device, radius):
    with torch.no_grad():
        total = 0.0
        for name, param in model.named_parameters():
            diff = param - ref_state_device[name]
            total += float((diff * diff).sum().item())
        norm = float(np.sqrt(total))
        if norm <= radius or norm <= 1e-12:
            return norm
        scale = radius / norm
        for name, param in model.named_parameters():
            param.copy_(ref_state_device[name] + (param - ref_state_device[name]) * scale)
        return radius


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def add_grad_list_(dst, src, alpha=1.0):
    with torch.no_grad():
        for d, s in zip(dst, src):
            d.add_(s, alpha=alpha)
    return dst


def scale_grad_list(g, scalar):
    return [x * scalar for x in g]


def compute_mean_grad_from_indices(model, indices):
    if indices is None or len(indices) == 0:
        return None
    xb, yb = fetch_samples_by_indices(train_data_all, indices)
    if xb is None:
        return None
    loss = F.cross_entropy(model(xb), yb)
    return compute_grad_list(model, loss)


def compute_sfu_inverse_loss_grad_single_sample(model, du_idx, eps=SFU_SURROGATE_LOSS_EPS):
    xb_du, yb_du = fetch_samples_by_indices(train_data_all, [du_idx])
    base_loss = F.cross_entropy(model(xb_du), yb_du)
    surrogate = 1.0 / torch.clamp(base_loss, min=eps)
    return compute_grad_list(model, surrogate)


def sample_client_minibatch_indices(client_indices_ref, client_id, batch_size, rng):
    pool = np.array(client_indices_ref[client_id], dtype=np.int64)
    if pool.size == 0:
        return []
    take = min(int(batch_size), int(pool.size))
    picked = rng.choice(pool, size=take, replace=False)
    return picked.tolist()


def apply_grad_descent_(model, grads, lr):
    with torch.no_grad():
        for p, g in zip(model_trainable_params(model), grads):
            p.add_(-lr * g)


def project_grad_orthogonal_to_client_svd_per_layer(target_grad, retain_client_grads, epsilon=SFU_EPSILON):
    if len(retain_client_grads) == 0:
        return [g.clone() for g in target_grad], [], []

    projected = []
    layer_ranks = []
    layer_coverages = []
    for layer_idx, g_t in enumerate(target_grad):
        retain_cols = []
        for g_r in retain_client_grads:
            if g_r is None:
                continue
            retain_cols.append(g_r[layer_idx].reshape(-1))

        if len(retain_cols) == 0:
            projected.append(g_t.clone())
            layer_ranks.append(0)
            layer_coverages.append(0.0)
            continue

        R = torch.stack(retain_cols, dim=1)
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "svd"):
            U, S, _ = torch.linalg.svd(R, full_matrices=False)
        else:
            U, S, _ = torch.svd(R, some=True)

        sq = S * S
        total_energy = float(sq.sum().item())
        if total_energy <= 1e-12:
            projected.append(g_t.clone())
            layer_ranks.append(0)
            layer_coverages.append(0.0)
            continue

        cumsum = torch.cumsum(sq, dim=0)
        target_energy = float(epsilon) * total_energy
        cumsum_np = cumsum.detach().cpu().numpy()
        k = int(np.searchsorted(cumsum_np, target_energy, side="left")) + 1
        k = min(k, int(S.numel()))

        U_keep = U[:, :k]
        g_flat = g_t.reshape(-1)
        g_proj_flat = g_flat - U_keep @ (U_keep.transpose(0, 1) @ g_flat)
        projected.append(g_proj_flat.reshape_as(g_t))
        layer_ranks.append(k)
        layer_coverages.append(float(cumsum[k - 1].item() / total_energy))
    return projected, layer_ranks, layer_coverages


def run_post_train_from_model(base_model, client_indices_ref, phase_tag, rounds=POST_TRAIN_ROUNDS):
    init_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    return federated_train_standard(
        phase_tag=phase_tag,
        init_state=init_state,
        client_indices_ref=client_indices_ref,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=FL_LR,
    )


# ============================================================
# 10. Oracle retrain (kept as oracle reference, not part of P2-P5)
# ============================================================
sfu_retain_client_ids = [int(cid) for cid in range(num_clients) if cid != FORGET_CLIENT_ID]
print(f"[SFU] retained clients participating in subspace construction: {sfu_retain_client_ids}")

retained_posttrain_client_indices = make_phase_client_indices_from_removed_set(
    removed_set=set(du_indices),
    phase_tag="RetainedAll-PostTrain",
)

base_retrain_state = build_fixed_retrain_init_state(RETRAIN_INIT_SEED)
print(f"\n[Retrain Init] fixed retrain init state created with seed={RETRAIN_INIT_SEED}")

client_indices_oracle = make_phase_client_indices_from_removed_set(removed_set=set(du_indices), phase_tag="Oracle-Retrain")
global_model_oracle, oracle_retrain_time = federated_train_standard(
    phase_tag="Oracle-Retrain",
    init_state=base_retrain_state,
    client_indices_ref=client_indices_oracle,
    rounds=rounds,
    local_epochs=local_epochs,
    batch_size=batch_size,
    lr=FL_LR,
)
oracle_metrics = evaluate_method("Oracle-Retrain", global_model_oracle, oracle_retrain_time)


# ============================================================
# 11. P2-P5: fast unlearning methods
# ============================================================
def get_ordered_du_indices(du_indices):
    return sorted(list(du_indices), key=lambda idx: (-avg_neighbors_client0[idx], idx))


# def run_direct_ga_unlearning(base_model, du_indices):
#     model = clone_model_from(base_model)
#     # Keep eval mode during fast unlearning so BatchNorm running statistics are not
#     # updated by repeated forward passes on Du/retain data.
#     model.eval()
#     loader = make_loader_from_indices(du_indices, batch_size=DIRECT_GA_BATCH_SIZE, shuffle=False)
#     _sync_cuda()
#     t0 = time.perf_counter()
#     for ep in range(DIRECT_GA_EPOCHS):
#         for xb, yb in loader:
#             xb, yb = xb.to(device), yb.to(device)
#             loss_du = F.cross_entropy(model(xb), yb)
#             g_du = compute_grad_list(model, loss_du)
#             g_du = maybe_clip_grad_list(g_du, DIRECT_GA_GRAD_CLIP_NORM)
#             apply_grad_ascent_(model, g_du, DIRECT_GA_LR)
#         print(f"[Phase3-DirectGA][Epoch {ep+1}] finished")
#     _sync_cuda()
#     t1 = time.perf_counter()
#     return model, (t1 - t0)

def run_direct_ga_unlearning(base_model, du_indices):
    model = clone_model_from(base_model)
    # Keep eval mode during fast unlearning so BatchNorm running statistics are not
    # updated by repeated forward passes on Du/retain data.
    model.eval()

    ordered_du = get_ordered_du_indices(du_indices)

    _sync_cuda()
    t0 = time.perf_counter()
    for ep in range(DIRECT_GA_EPOCHS):
        for du_idx in ordered_du:
            xb, yb = fetch_samples_by_indices(train_data_all, [du_idx])
            loss_du = F.cross_entropy(model(xb), yb)
            g_du = compute_grad_list(model, loss_du)
            g_du = maybe_clip_grad_list(g_du, DIRECT_GA_GRAD_CLIP_NORM)
            apply_grad_ascent_(model, g_du, DIRECT_GA_LR)

        print(
            f"[Phase3-DirectGA][Epoch {ep+1}] finished | "
            f"protocol=sequential_single_sample"
        )

    _sync_cuda()
    t1 = time.perf_counter()
    return model, (t1 - t0)


def run_pgd_based_fu(base_model, du_indices):
    model = clone_model_from(base_model)
    # Keep eval mode during fast unlearning so BatchNorm running statistics are not
    # updated by repeated forward passes on Du/retain data.
    model.eval()
    ordered_du = get_ordered_du_indices(du_indices)
    ref_state_device = {k: v.detach().to(device) for k, v in base_model.state_dict().items()}

    _sync_cuda()
    t0 = time.perf_counter()
    current_norm = 0.0
    for ep in range(PGD_FU_EPOCHS):
        for du_idx in ordered_du:
            xb, yb = fetch_samples_by_indices(train_data_all, [du_idx])
            loss_du = F.cross_entropy(model(xb), yb)
            g_du = compute_grad_list(model, loss_du)
            g_du = maybe_clip_grad_list(g_du, PGD_FU_GRAD_CLIP_NORM)
            apply_grad_ascent_(model, g_du, PGD_FU_LR)
            current_norm = project_model_to_l2_ball_(model, ref_state_device, PGD_FU_RADIUS)
        print(
            f"[Phase4-PGDFU][Epoch {ep+1}] finished | protocol=sequential_single_sample | "
            f"projected_delta_norm<={PGD_FU_RADIUS:.4f} | current={current_norm:.4f}"
        )
    _sync_cuda()
    t1 = time.perf_counter()
    return model, (t1 - t0)


def run_sfu_unlearning(base_model, du_indices, retain_client_ids, client_indices_ref):
    model = clone_model_from(base_model)
    # Keep eval mode during fast unlearning so BatchNorm running statistics are not
    # updated by repeated forward passes on Du/retain data.

    ordered_du = get_ordered_du_indices(du_indices)
    rng = np.random.default_rng(SFU_BATCH_SAMPLING_SEED)

    _sync_cuda()
    t0 = time.perf_counter()
    last_retain_client_batch_sizes = OrderedDict()
    last_layer_ranks = []
    last_layer_coverages = []

    for ep in range(SFU_EPOCHS):
        for du_idx in ordered_du:
            g_du = compute_sfu_inverse_loss_grad_single_sample(
                model=model,
                du_idx=du_idx,
                eps=SFU_SURROGATE_LOSS_EPS,
            )

            retain_client_grads = []
            retain_client_batch_sizes = OrderedDict()
            for cid in retain_client_ids:
                sampled_idx = sample_client_minibatch_indices(
                    client_indices_ref=client_indices_ref,
                    client_id=cid,
                    batch_size=SFU_RETAIN_BATCH_SIZE,
                    rng=rng,
                )
                retain_client_batch_sizes[int(cid)] = len(sampled_idx)
                g_client = compute_mean_grad_from_indices(model, sampled_idx)
                if g_client is not None:
                    retain_client_grads.append(g_client)

            if len(retain_client_grads) == 0:
                g_sfu = g_du
                layer_ranks = []
                layer_coverages = []
            else:
                g_sfu, layer_ranks, layer_coverages = project_grad_orthogonal_to_client_svd_per_layer(
                    target_grad=g_du,
                    retain_client_grads=retain_client_grads,
                    epsilon=SFU_EPSILON,
                )

            g_sfu = maybe_clip_grad_list(g_sfu, SFU_GRAD_CLIP_NORM)
            apply_grad_descent_(model, g_sfu, SFU_LR)

            last_retain_client_batch_sizes = retain_client_batch_sizes
            last_layer_ranks = layer_ranks
            last_layer_coverages = layer_coverages

        avg_rank = float(np.mean(last_layer_ranks)) if len(last_layer_ranks) > 0 else 0.0
        avg_cov = float(np.mean(last_layer_coverages)) if len(last_layer_coverages) > 0 else 0.0
        print(
            f"[Phase5-SFU][Epoch {ep+1}] finished | protocol=sequential_single_sample_paper_aligned | "
            f"retain_clients={retain_client_ids} | retain_batch_sizes={dict(last_retain_client_batch_sizes)} | "
            f"avg_layer_rank={avg_rank:.2f} | avg_layer_coverage={avg_cov:.4f} | epsilon={SFU_EPSILON:.2f}"
        )
    _sync_cuda()
    t1 = time.perf_counter()
    return model, (t1 - t0)


def run_nfu_unlearning(base_model, du_indices, neighbor_map):
    model = clone_model_from(base_model)
    # Keep eval mode during fast unlearning so BatchNorm running statistics are not
    # updated by repeated forward passes on Du/retain data.
    model.eval()
    ordered_du = sorted(list(du_indices), key=lambda idx: (-avg_neighbors_client0[idx], idx))

    _sync_cuda()
    t0 = time.perf_counter()
    for ep in range(NFU_EPOCHS):
        for du_idx in ordered_du:
            xb_u, yb_u = fetch_samples_by_indices(train_data_all, [du_idx])
            loss_u = F.cross_entropy(model(xb_u), yb_u)
            g_u = compute_grad_list(model, loss_u)

            n_info = neighbor_map.get(int(du_idx), {})
            n_indices = list(n_info.get("neighbor_indices", []))
            n_weights = list(n_info.get("neighbor_weights", []))
            if len(n_indices) == 0:
                g_proj = g_u
            else:
                xb_n, yb_n = fetch_samples_by_indices(train_data_all, n_indices)
                weight_t = torch.tensor(n_weights, dtype=torch.float32, device=device)
                logits_n = model(xb_n)
                losses_n = F.cross_entropy(logits_n, yb_n, reduction="none")
                loss_neighbors = (weight_t * losses_n).sum()
                g_n = compute_grad_list(model, loss_neighbors)
                g_proj = project_grad_orthogonal_to_basis(g_u, [g_n])

            g_proj = maybe_clip_grad_list(g_proj, NFU_GRAD_CLIP_NORM)
            apply_grad_ascent_(model, g_proj, NFU_LR)
        print(f"[Phase2-NFU][Epoch {ep+1}] finished")

    _sync_cuda()
    t1 = time.perf_counter()
    pure_unlearn_time = t1 - t0
    print(f"[Phase2-NFU] pure_unlearn_time={pure_unlearn_time:.6f} s")
    return model, pure_unlearn_time


# P2: NFU (ours)
global_model_p2, p2_pure_unlearn_time = run_nfu_unlearning(
    base_model=global_model,
    du_indices=du_indices,
    neighbor_map=neighbor_map_for_du,
)
p2_metrics = evaluate_method("Phase2-NFU", global_model_p2, p2_pure_unlearn_time)

# P3: Direct GA
global_model_p3, p3_time = run_direct_ga_unlearning(base_model=global_model, du_indices=du_indices)
p3_metrics = evaluate_method("Phase3-DirectGA", global_model_p3, p3_time)

# P4: PGD-based FU (paper-aligned implementation)
pgd_model_unlearn, p4_unlearn_time = run_pgd_based_fu(base_model=global_model, du_indices=du_indices)
p4_unlearn_metrics = evaluate_method("Phase4-PGDFU-Unlearn", pgd_model_unlearn, p4_unlearn_time)

pgd_model_post, p4_post_train_time = run_post_train_from_model(
    base_model=pgd_model_unlearn,
    client_indices_ref=retained_posttrain_client_indices,
    phase_tag="Phase4-PGDFU-PostTrain1",
    rounds=POST_TRAIN_ROUNDS,
)
p4_total_time = p4_unlearn_time + p4_post_train_time
print(f"[Phase4-PGDFU] post_train_only_time={p4_post_train_time:.6f} s | cumulative_time={p4_total_time:.6f} s")
p4_post_metrics = evaluate_method("Phase4-PGDFU+1PT", pgd_model_post, p4_total_time)

# P5: SFU (paper-aligned implementation)
global_model_p5, p5_time = run_sfu_unlearning(
    base_model=global_model,
    du_indices=du_indices,
    retain_client_ids=sfu_retain_client_ids,
    client_indices_ref=original_client_indices_backup_phase1,
)
p5_metrics = evaluate_method("Phase5-SFU-PaperAligned", global_model_p5, p5_time)


# ============================================================
# 12. final summary
# ============================================================
def summary_row(tag, metrics):
    return OrderedDict(
        method=tag,
        time_s=metrics["time_s"],
        dt_acc=metrics["test"]["acc"],
        dt_loss=metrics["test"]["loss"],
        dr_acc=metrics["dr"]["acc"],
        dr_loss=metrics["dr"]["loss"],
        du_acc=metrics["du"]["acc"],
        du_loss=metrics["du"]["loss"],
        du_mia_acc=metrics["du_mia"]["acc"],
        du_mia_f1=metrics["du_mia"]["f1"],
    )


summary_rows = [
    summary_row("Oracle-Retrain", oracle_metrics),
    summary_row("Phase2-NFU(Ours)", p2_metrics),
    summary_row("Phase3-DirectGA", p3_metrics),
    summary_row("Phase4-PGDFU(Unlearn)", p4_unlearn_metrics),
    summary_row("Phase4-PGDFU(+1PT)", p4_post_metrics),
    summary_row("Phase5-SFU(PaperAligned)", p5_metrics),
]

print("\n" + "=" * 120)
print("[FINAL SUMMARY]")
print("=" * 120)
print(f"[Phase1 Train Time] {p1_train_time:.6f} s")
print("[Training Backend] standard FedAvg training aligned with the user's current source code (not public-schedule training)")
print(f"[Neighbor Table Build Time] {neighbor_book['build_time']:.6f} s (reported separately, NOT counted in NFU algorithm time)")
print(f"[Du Size] {len(du_indices)} | [Dr Size] {len(remaining_excluding_du_indices)} | threshold={NEIGHBOR_THRESHOLD}")
print(f"[Du Neighbor Avg Range] max={max(avg_neighbors_client0[idx] for idx in du_indices):.6f} | min={min(avg_neighbors_client0[idx] for idx in du_indices):.6f}")
print(f"[Du][CIFAR10 Real Idx List] = {du_indices}")
print("[Unlearning Protocol] PGD uses sequential single-sample updates plus +1 retained-data post-train; SFU uses inverse-loss target gradient and per-retained-client mini-batch SVD projection.")
print("[PGDFU Post-Train] remaining all data with Du removed from client0, rounds=1")
print("[SFU Projection] each retained client contributes one stochastic descent mini-batch gradient; server performs per-layer SVD with epsilon-coverage rank selection and projects the target gradient onto the orthogonal complement.")
print("\n[Summary Table]")
header = [
    "method", "time_s", "Dt_acc", "Dt_loss", "Dr_acc", "Dr_loss", "Du_acc", "Du_loss", "Du_MIA_acc", "Du_MIA_F1"
]
print("\t".join(header))
for row in summary_rows:
    print(
        f"{row['method']}\t{row['time_s']:.6f}\t{row['dt_acc']:.2f}\t{row['dt_loss']:.4f}\t"
        f"{row['dr_acc']:.2f}\t{row['dr_loss']:.4f}\t{row['du_acc']:.2f}\t{row['du_loss']:.4f}\t"
        f"{row['du_mia_acc']:.2f}\t{row['du_mia_f1']:.2f}"
    )

print_comparison_vs_oracle("Phase2-NFU(Ours)", oracle_metrics, p2_metrics)
print_comparison_vs_oracle("Phase3-DirectGA", oracle_metrics, p3_metrics)
print_comparison_vs_oracle("Phase4-PGDFU(Unlearn)", oracle_metrics, p4_unlearn_metrics)
print_comparison_vs_oracle("Phase4-PGDFU(+1PT)", oracle_metrics, p4_post_metrics)
print_comparison_vs_oracle("Phase5-SFU(PaperAligned)", oracle_metrics, p5_metrics)
