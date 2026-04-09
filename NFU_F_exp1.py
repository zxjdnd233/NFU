import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import random
import argparse
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
TOP_FILTER_NUM = 30

# parser = argparse.ArgumentParser()
# parser.add_argument("--seed", type=int, required=True, help="random seed")
# parser.add_argument("--top_filter_num", type=int, required=True, help="TOP_FILTER_NUM")
# args = parser.parse_args()
#
# RANDOMSEED = args.seed
# TOP_FILTER_NUM = args.top_filter_num

RETRAIN_INIT_SEED = RANDOMSEED + 99999


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
# 1. config
# ============================================================
num_clients = 3
classes = 10
FIRST_FOUR_PER_CLASS = 1000

FORGET_CLIENT_ID = 0
FORGET_PER_CLASS = 30

rounds = 10
local_epochs = 1
batch_size = 128

# five filters
E2LSH_K = 16
E2LSH_L = 9
E2LSH_W = 64  # unused
# TOP_FILTER_NUM = 60

CONFIDENCE_THRESHOLD = 0.8

FEDKM_NUM_CLUSTERS = 5
FEDKM_ITERS = 25

FEDPLVM_LOCAL_CLUSTERS = 2
FEDPLVM_LOCAL_ITERS = 20
FEDPLVM_GLOBAL_CLUSTERS = 5
FEDPLVM_GLOBAL_ITERS = 20

NUM_WORKERS = 0

print(f"num_clients={num_clients}, classes={classes}, FIRST_FOUR_PER_CLASS={FIRST_FOUR_PER_CLASS}")
print(f"FORGET_CLIENT_ID={FORGET_CLIENT_ID}, FORGET_PER_CLASS={FORGET_PER_CLASS}")
print(f"rounds={rounds}, local_epochs={local_epochs}, batch_size={batch_size}")
print(f"cosLSH_K={E2LSH_K}, cosLSH_L={E2LSH_L}, TOP_FILTER_NUM={TOP_FILTER_NUM}, CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}")
print(f"FEDKM_NUM_CLUSTERS={FEDKM_NUM_CLUSTERS}, FEDKM_ITERS={FEDKM_ITERS}")
print(
    f"FEDPLVM_LOCAL_CLUSTERS={FEDPLVM_LOCAL_CLUSTERS}, FEDPLVM_LOCAL_ITERS={FEDPLVM_LOCAL_ITERS}, "
    f"FEDPLVM_GLOBAL_CLUSTERS={FEDPLVM_GLOBAL_CLUSTERS}, FEDPLVM_GLOBAL_ITERS={FEDPLVM_GLOBAL_ITERS}"
)
print(f"NUM_WORKERS={NUM_WORKERS}")
print(f"RANDOMSEED={RANDOMSEED} | RETRAIN_INIT_SEED={RETRAIN_INIT_SEED}")

# ============================================================
# 2. data
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data_full = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

train_data_all = train_data_full
if torch.is_tensor(train_data_full.targets):
    targets_full = train_data_full.targets.cpu().numpy()
else:
    targets_full = np.array(train_data_full.targets)

selected_targets = targets_full
selected_indices = list(range(len(train_data_full)))

print(f"[Info] use full training set: {len(train_data_all)}")

class_to_localidx = {c: np.where(selected_targets == c)[0].tolist() for c in range(classes)}
for c in range(classes):
    np.random.shuffle(class_to_localidx[c])


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
                f"class {c} only has {remain_c} samples, "
                f"not enough for first {num_clients-1} clients with "
                f"{FIRST_FOUR_PER_CLASS} per class."
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


test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ============================================================
# 3. model utils
# ============================================================
class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 7x7 -> 7x7
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))                 # -> 128 x 1 x 1
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)   # [B, 128]
        logits = self.classifier(feat)
        return logits

    def extract_features(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)   # [B, 128]
        return feat


def build_model():
    return MNISTCNN(num_classes=classes, feat_dim=128)


def build_fixed_retrain_init_state(seed):
    reset_all_seeds(seed)
    model = build_model().to(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
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
        return float('nan'), float('nan')
    return 100.0 * correct / total, total_loss / max(1, len(loader))


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
# 4. forget set
# ============================================================
class_to_client_local = {c: [] for c in range(classes)}
for loc_idx in client_indices[FORGET_CLIENT_ID]:
    c = int(selected_targets[loc_idx])
    class_to_client_local[c].append(loc_idx)

forget_local_indices = []
rng = np.random.default_rng(RANDOMSEED + 2025)
for c in range(classes):
    lst = class_to_client_local[c]
    rng.shuffle(lst)
    take = lst[:FORGET_PER_CLASS]
    assert len(take) == FORGET_PER_CLASS, f"client {FORGET_CLIENT_ID} class {c} insufficient"
    forget_local_indices.extend(take)

forget_local_indices = sorted(forget_local_indices)
forget_loader = make_loader_from_indices(forget_local_indices)
remaining_excluding_forget_indices = sorted(list(set(range(len(train_data_all))) - set(forget_local_indices)))
remaining_excluding_forget_loader = make_loader_from_indices(remaining_excluding_forget_indices)
print(f"[Forget] client{FORGET_CLIENT_ID} forget size: {len(forget_local_indices)} (per class {FORGET_PER_CLASS})")
print(f"[Forget][mnist Real Idx List] = {forget_local_indices}")
print(f"[Remain Excluding Forget] size: {len(remaining_excluding_forget_indices)}")

# ============================================================
# 5. Phase1 FL training
# ============================================================
reset_all_seeds(RANDOMSEED)
global_model = build_model().to(device)

print("\n[FL-Phase1] start federated training...")
for rnd in range(1, rounds + 1):
    client_states = []
    sizes = []
    weighted_loss_sum = 0.0
    total_samples = 0

    for cid in range(num_clients):
        client_model = build_model().to(device)
        client_model.load_state_dict(global_model.state_dict())

        train_loader = get_client_loader(cid, batch_size=batch_size, shuffle=False)
        avg_loss_client = local_train_and_report_loss(client_model, train_loader, epochs=local_epochs, lr=1e-3)

        n_i = len(client_indices[cid])
        weighted_loss_sum += avg_loss_client * n_i
        total_samples += n_i

        client_states.append({k: v.detach().cpu() for k, v in client_model.state_dict().items()})
        sizes.append(n_i)

        del client_model
        torch.cuda.empty_cache()

    aggregate_fedavg(global_model, client_states, sizes)
    round_train_loss = weighted_loss_sum / max(1, total_samples)
    print(f"[Phase1][Round {rnd}] Train Loss: {round_train_loss:.4f}")

acc_test1, loss_test1 = evaluate(global_model, test_loader)
acc_forget1, loss_forget1 = evaluate(global_model, forget_loader)
acc_remain1, loss_remain1 = evaluate(global_model, remaining_excluding_forget_loader)
print(f"\n[Phase1][Test All] Accuracy: {acc_test1:.2f}% | Avg Loss: {loss_test1:.4f}")
print(f"[Phase1][Remain Excluding Forget] Accuracy: {acc_remain1:.2f}% | Avg Loss: {loss_remain1:.4f}")
print(f"[Phase1][Forget]  Accuracy: {acc_forget1:.2f}% | Avg Loss: {loss_forget1:.4f}")

# ============================================================
# 6. MIA fixed data / attack model
# ============================================================
# Only the attack-model training stage needs a member / non-member pair.
# For Forget / UU evaluation sets, we evaluate directly on the target set
# itself without sampling an equal-sized extra non-member set.
mia_member_nominal = int(0.2 * len(train_data_all))
mia_member = min(mia_member_nominal, len(test_data))
if mia_member <= 0:
    raise ValueError(
        f"Invalid mia_member={mia_member}. len(test_data)={len(test_data)}."
    )
print(f"[MIA] nominal member/non-member size={mia_member_nominal}, effective size={mia_member}")

all_local_idx = set(range(len(train_data_all)))
client0_local = set(client_indices[FORGET_CLIENT_ID])
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
        losses = F.cross_entropy(logits, y, reduction='none')
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
    acc = 100.0 * float((y_true == y_pred).mean()) if len(y_true) > 0 else float('nan')

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
    mem_acc = 100.0 * correct_mem / max(1, total_mem) if total_mem > 0 else float('nan')
    non_acc = 100.0 * correct_non / max(1, total_non) if total_non > 0 else float('nan')

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
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    feats, labs = [], []
    ptr = 0
    with torch.no_grad():
        for xb, yb_cls in loader:
            xb = xb.to(device)
            yb_cls = yb_cls.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            losses = F.cross_entropy(logits, yb_cls, reduction='none')
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
        return float('nan'), float('nan')

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
    X_attack_p1, y_attack_p1, attack_train_idx, attack_val_idx, epochs=10, lr=1e-3, tag="MIA-Phase1"
)
eval_attack_three_metrics(attack_model_p1, attack_val_loader_p1, name="MIA-Phase1")
eval_named_set_attack_acc_f1(
    target_model=global_model,
    attack_model=attack_model_p1,
    eval_indices=forget_local_indices,
    true_membership_label=1,
    phase_tag="MIA-Phase1",
    set_name="Forget",
)

# ============================================================
# 7. shared feature precompute for five filters
# ============================================================
FEATURE_BACKBONE = global_model

class CNNFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.extract_features(x)

@torch.no_grad()
def precompute_client_features(client_id, feature_net, batch_size=256, normalize=True):
    feature_net.eval()
    loader = get_client_loader(client_id, batch_size=batch_size, shuffle=False)
    local_order = client_indices[client_id]
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


print("\n[Shared Precompute] forward embedding + L2 normalize (NOT timed for filters) ...")
feature_net = CNNFeatureExtractor(FEATURE_BACKBONE).to(device)

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


@torch.no_grad()
def collect_softmax_confidence_for_local_indices(model, dataset, local_indices, batch_size=256, num_workers=NUM_WORKERS):
    conf_dict = {}
    if len(local_indices) == 0:
        return conf_dict

    model.eval()
    ordered_local_indices = list(local_indices)
    loader = DataLoader(Subset(dataset, ordered_local_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    ptr = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        max_conf, pred = probs.max(dim=1)
        true_conf = probs.gather(1, yb.unsqueeze(1)).squeeze(1)
        B = xb.size(0)
        for i in range(B):
            loc_idx = ordered_local_indices[ptr + i]
            conf_dict[loc_idx] = {
                "true_label": int(yb[i].item()),
                "pred_label": int(pred[i].item()),
                "max_confidence": float(max_conf[i].item()),
                "true_label_confidence": float(true_conf[i].item()),
            }
        ptr += B

    return conf_dict


def print_filter_summary(method_name, uu_local_indices, score_dict=None, extra_info=None):
    print("\n" + "=" * 100)
    print(f"[{method_name}]")
    print("=" * 100)
    print(f"UU picked: {len(uu_local_indices)} / {len(forget_local_indices)}")

    uu_class_counts = Counter(int(selected_targets[idx]) for idx in uu_local_indices)
    print("UU count per class:")
    for c in range(classes):
        print(f"  Class {c}: {uu_class_counts.get(c, 0)}")

    if extra_info is not None:
        for k, v in extra_info.items():
            print(f"{k}: {v}")

    if score_dict is not None and len(score_dict) > 0:
        vals = [score_dict[idx] for idx in forget_local_indices if idx in score_dict]
        if len(vals) > 0:
            print(
                f"Forget-set score mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
                f"min={np.min(vals):.6f}, max={np.max(vals):.6f}"
            )


def print_uu_detailed_table(method_name, uu_local_indices, metric_dict, metric_name, conf_dict, descending=True):
    print("\n" + "-" * 120)
    print(f"[{method_name}][UU DETAILS]")
    print("-" * 120)
    if len(uu_local_indices) == 0:
        print("Empty UU set.")
        return

    ranked = sorted(
        list(uu_local_indices),
        key=lambda idx: (-float(metric_dict.get(idx, float('-inf'))), int(idx)) if descending else (float(metric_dict.get(idx, float('inf'))), int(idx))
    )

    print(f"rank\tmnist_idx\t{metric_name}\tpred_label\ttrue_label\tmax_confidence\ttrue_label_confidence")
    for rank, idx in enumerate(ranked, start=1):
        info = conf_dict.get(idx, None)
        metric_val = metric_dict.get(idx, float('nan'))
        if info is None:
            print(f"{rank}\t{idx}\t{metric_val:.6f}\tNA\t{int(selected_targets[idx])}\tNA\tNA")
        else:
            print(
                f"{rank}\t{idx}\t{metric_val:.6f}\t{info['pred_label']}\t{info['true_label']}\t"
                f"{info['max_confidence']:.6f}\t{info['true_label_confidence']:.6f}"
            )


def select_top_indices(score_dict, candidate_indices, top_num=100, descending=True):
    candidate_indices = list(candidate_indices)
    if len(candidate_indices) == 0:
        return []
    if descending:
        ranked = sorted(candidate_indices, key=lambda idx: (-float(score_dict[idx]), int(idx)))
    else:
        ranked = sorted(candidate_indices, key=lambda idx: (float(score_dict[idx]), int(idx)))
    return ranked[:min(top_num, len(ranked))]


print("\n[Shared Precompute] collect forget-set confidence / labels (NOT timed for filters) ...")
forget_conf_dict_all = collect_softmax_confidence_for_local_indices(
    model=global_model,
    dataset=train_data_all,
    local_indices=forget_local_indices,
    batch_size=256,
    num_workers=NUM_WORKERS,
)

# ============================================================
# 8. filter 1: Single SimHash
# ============================================================
FEATURE_DIM = 128
@torch.no_grad()
def server_make_coslsh_params(L=8, K=16, d=128, seed=2025, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    params = []
    for _ in range(L):
        R = torch.randn((K, d), generator=g, device=device)
        params.append(R)
    return params


def client_hash_and_build_tables_from_cached_features(feats_all, labels_all, R_list, hash_batch=2048):
    N = feats_all.shape[0]
    L = len(R_list)
    codes_L = [[] for _ in range(L)]
    local_tables_L = [defaultdict(Counter) for _ in range(L)]

    for ell, R in enumerate(R_list):
        for st in range(0, N, hash_batch):
            ed = min(N, st + hash_batch)
            feats_b = feats_all[st:ed]
            proj = feats_b @ R.t()
            h_cpu = (proj >= 0).to(torch.uint8).cpu().numpy()
            for i in range(h_cpu.shape[0]):
                code_str = "|".join(map(str, h_cpu[i].tolist()))
                codes_L[ell].append(code_str)
        for code, lab in zip(codes_L[ell], labels_all):
            local_tables_L[ell][int(lab)][code] += 1

    return codes_L, local_tables_L


def server_aggregate_tables_multi(local_tables_L_all_clients):
    L = len(local_tables_L_all_clients[0])
    global_tables_L = [defaultdict(Counter) for _ in range(L)]
    for tables_L in local_tables_L_all_clients:
        for ell in range(L):
            local_tbl = tables_L[ell]
            for cls, counter in local_tbl.items():
                global_tables_L[ell][cls].update(counter)
    return global_tables_L


def neighbors_in_table_sameclass(code_str, label, global_table_ell, subtract_self=True):
    cls = int(label)
    cnt = global_table_ell[cls].get(code_str, 0)
    if subtract_self and cnt > 0:
        cnt -= 1
    return cnt


def run_single_simhash_filter(client_cached, forget_local_indices, top_num=100, seed=2025):
    cos_params = server_make_coslsh_params(L=E2LSH_L, K=E2LSH_K, d=128, seed=seed, device=device)

    _sync_cuda()
    t_build_start = time.perf_counter()

    local_tables_all_clients = []
    client_books = {}
    for cid in range(num_clients):
        feats_all = client_cached[cid]["feats"]
        labels_all = client_cached[cid]["labels"]
        local_idx_all = client_cached[cid]["local_idx"]
        codes_L, local_tables_L = client_hash_and_build_tables_from_cached_features(feats_all, labels_all, cos_params, hash_batch=2048)
        local_tables_all_clients.append(local_tables_L)
        client_books[cid] = {"codes_L": codes_L, "labels": labels_all, "local_idx": local_idx_all}

    global_tables_L = server_aggregate_tables_multi(local_tables_all_clients)

    _sync_cuda()
    t_build_end = time.perf_counter()
    build_time = t_build_end - t_build_start

    cid = FORGET_CLIENT_ID
    codes_L = client_books[cid]["codes_L"]
    labels = client_books[cid]["labels"]
    local_i = client_books[cid]["local_idx"]
    forget_set = set(forget_local_indices)

    _sync_cuda()
    t_query_start = time.perf_counter()

    avg_neighbors = {}
    for idx_pos, loc_idx in enumerate(local_i):
        if loc_idx not in forget_set:
            continue
        lab = int(labels[idx_pos])
        neigh_counts = []
        for ell in range(E2LSH_L):
            code_str = codes_L[ell][idx_pos]
            total_n = neighbors_in_table_sameclass(code_str, lab, global_tables_L[ell], subtract_self=True)
            neigh_counts.append(total_n)
        avg_neighbors[loc_idx] = float(np.mean(neigh_counts))

    unnecessary_local_indices = select_top_indices(avg_neighbors, forget_local_indices, top_num=top_num, descending=True)

    _sync_cuda()
    t_query_end = time.perf_counter()
    query_time = t_query_end - t_query_start

    return {
        "uu_local_indices": sorted(unnecessary_local_indices),
        "avg_neighbors": avg_neighbors,
        "time": build_time + query_time,
        "build_time": build_time,
        "query_time": query_time,
        "global_tables_L": global_tables_L,
    }


simhash_result = run_single_simhash_filter(client_cached=client_cached, forget_local_indices=forget_local_indices, top_num=TOP_FILTER_NUM, seed=2025)
print(f"\n[TIME][Single SimHash][Build Tables] {simhash_result['build_time']:.6f} s")
print(f"[TIME][Single SimHash][Query+Select] {simhash_result['query_time']:.6f} s")
print(f"[TIME][Single SimHash][Total] {simhash_result['time']:.6f} s")
print_filter_summary(
    method_name="Single SimHash Filter",
    uu_local_indices=simhash_result["uu_local_indices"],
    score_dict=simhash_result["avg_neighbors"],
    extra_info={
        "top_num": TOP_FILTER_NUM,
        "build_tables_time_s": f"{simhash_result['build_time']:.6f}",
        "query_select_time_s": f"{simhash_result['query_time']:.6f}",
        "total_filter_time_s": f"{simhash_result['time']:.6f}",
    },
)
print_uu_detailed_table(
    method_name="Single SimHash Filter",
    uu_local_indices=simhash_result["uu_local_indices"],
    metric_dict=simhash_result["avg_neighbors"],
    metric_name="avg_neighbors",
    conf_dict=forget_conf_dict_all,
    descending=True,
)
print(f'[Single SimHash Filter][UU mnist Real Idx List] = {simhash_result["uu_local_indices"]}')

# ============================================================
# 9. filter 2: SimHash + Confidence
# ============================================================
def run_simhash_confidence_filter(client_cached, forget_local_indices, top_num=100, conf_threshold=0.9, seed=2025):
    simhash_result_local = run_single_simhash_filter(client_cached=client_cached, forget_local_indices=forget_local_indices, top_num=len(forget_local_indices), seed=seed)

    _sync_cuda()
    t_conf_start = time.perf_counter()

    forget_conf_scores = {idx: forget_conf_dict_all[idx]["true_label_confidence"] for idx in forget_local_indices}
    simhash_ranked_all = select_top_indices(
        score_dict=simhash_result_local["avg_neighbors"],
        candidate_indices=forget_local_indices,
        top_num=len(forget_local_indices),
        descending=True,
    )
    eligible_indices = [idx for idx in simhash_ranked_all if forget_conf_dict_all[idx]["true_label_confidence"] >= conf_threshold]
    refined_uu = eligible_indices[:min(top_num, len(eligible_indices))]

    _sync_cuda()
    t_conf_end = time.perf_counter()
    confidence_refine_time = t_conf_end - t_conf_start

    return {
        "uu_local_indices": sorted(refined_uu),
        "time": simhash_result_local["build_time"] + simhash_result_local["query_time"] + confidence_refine_time,
        "build_time": simhash_result_local["build_time"],
        "query_time": simhash_result_local["query_time"],
        "confidence_refine_time": confidence_refine_time,
        "avg_neighbors": simhash_result_local["avg_neighbors"],
        "forget_conf_scores": forget_conf_scores,
        "conf_threshold": float(conf_threshold),
        "eligible_after_confidence": len(eligible_indices),
    }


simhash_conf_result = run_simhash_confidence_filter(
    client_cached=client_cached,
    forget_local_indices=forget_local_indices,
    top_num=TOP_FILTER_NUM,
    conf_threshold=CONFIDENCE_THRESHOLD,
    seed=2025,
)
print(f"\n[TIME][SimHash + Confidence][Build Tables] {simhash_conf_result['build_time']:.6f} s")
print(f"[TIME][SimHash + Confidence][Query+Select] {simhash_conf_result['query_time']:.6f} s")
print(f"[TIME][SimHash + Confidence][Confidence Refine] {simhash_conf_result['confidence_refine_time']:.6f} s")
print(f"[TIME][SimHash + Confidence][Total] {simhash_conf_result['time']:.6f} s")
print_filter_summary(
    method_name="SimHash + Confidence Filter",
    uu_local_indices=simhash_conf_result["uu_local_indices"],
    score_dict=simhash_conf_result["avg_neighbors"],
    extra_info={
        "top_num": TOP_FILTER_NUM,
        "confidence_threshold": f"{simhash_conf_result['conf_threshold']:.6f}",
        "eligible_after_confidence": simhash_conf_result["eligible_after_confidence"],
        "build_tables_time_s": f"{simhash_conf_result['build_time']:.6f}",
        "query_select_time_s": f"{simhash_conf_result['query_time']:.6f}",
        "confidence_refine_time_s": f"{simhash_conf_result['confidence_refine_time']:.6f}",
        "total_filter_time_s": f"{simhash_conf_result['time']:.6f}",
    },
)
print_uu_detailed_table(
    method_name="SimHash + Confidence Filter",
    uu_local_indices=simhash_conf_result["uu_local_indices"],
    metric_dict=simhash_conf_result["avg_neighbors"],
    metric_name="avg_neighbors",
    conf_dict=forget_conf_dict_all,
    descending=True,
)
print(f'[SimHash + Confidence Filter][UU mnist Real Idx List] = {simhash_conf_result["uu_local_indices"]}')

# ============================================================
# 10. filter 3: Federated K-means
# ============================================================
def _sample_initial_centers(feats_global, num_clusters, seed):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    n = feats_global.shape[0]
    if n >= num_clusters:
        perm = torch.randperm(n, generator=g)[:num_clusters]
        if feats_global.device.type != 'cpu':
            perm = perm.to(feats_global.device)
        centers = feats_global[perm].clone()
    else:
        rep_idx = torch.randint(low=0, high=n, size=(num_clusters,), generator=g)
        if feats_global.device.type != 'cpu':
            rep_idx = rep_idx.to(feats_global.device)
        centers = feats_global[rep_idx].clone()
    return centers


def federated_kmeans_per_class(client_cached, num_clusters=5, num_iters=10, seed=2025):
    centers_by_class = {}
    for cls in range(classes):
        local_feats_by_client = []
        global_chunks = []
        for cid in range(num_clients):
            feats = client_cached[cid]["feats"]
            labels_np = client_cached[cid]["labels_np"]
            mask_np = (labels_np == cls)
            feats_cls = feats[mask_np]
            local_feats_by_client.append(feats_cls)
            if feats_cls.shape[0] > 0:
                global_chunks.append(feats_cls)

        feats_global = torch.cat(global_chunks, dim=0)
        centers = _sample_initial_centers(feats_global, num_clusters, seed + cls)

        for _ in range(num_iters):
            sum_acc = torch.zeros_like(centers)
            cnt_acc = torch.zeros((num_clusters,), device=centers.device, dtype=torch.long)

            for feats_local in local_feats_by_client:
                if feats_local.shape[0] == 0:
                    continue
                dist = torch.cdist(feats_local, centers, p=2)
                assign = dist.argmin(dim=1)
                for k in range(num_clusters):
                    mask_k = (assign == k)
                    if mask_k.any():
                        sum_acc[k] += feats_local[mask_k].sum(dim=0)
                        cnt_acc[k] += int(mask_k.sum().item())

            for k in range(num_clusters):
                if cnt_acc[k] > 0:
                    centers[k] = sum_acc[k] / cnt_acc[k]
                else:
                    centers[k] = feats_global[random.randrange(feats_global.shape[0])]
            centers = F.normalize(centers, p=2, dim=1)

        centers_by_class[cls] = centers.detach().clone()
    return centers_by_class


def compute_fedkmeans_scores_for_client0(client_cached, centers_by_class):
    cid = FORGET_CLIENT_ID
    feats_client0 = client_cached[cid]["feats"]
    labels_client0 = client_cached[cid]["labels_np"]
    local_idx_client0 = client_cached[cid]["local_idx"]

    raw_dist_dict = {}
    for pos, loc_idx in enumerate(local_idx_client0):
        cls = int(labels_client0[pos])
        feat = feats_client0[pos:pos + 1]
        centers = centers_by_class[cls]
        dist = torch.cdist(feat, centers, p=2).min().item()
        raw_dist_dict[loc_idx] = float(dist)

    client0_all_dists = np.array([raw_dist_dict[idx] for idx in local_idx_client0], dtype=np.float32)
    d_min = float(np.min(client0_all_dists))
    d_max = float(np.max(client0_all_dists))

    score_dict = {}
    if d_max - d_min < 1e-12:
        for idx in local_idx_client0:
            score_dict[idx] = 1.0
    else:
        for idx in local_idx_client0:
            norm_dist = (raw_dist_dict[idx] - d_min) / (d_max - d_min)
            score_dict[idx] = float(1.0 - norm_dist)

    client0_scores = np.array([score_dict[idx] for idx in local_idx_client0], dtype=np.float32)
    score_avg = float(np.mean(client0_scores))
    score_std = float(np.std(client0_scores))
    thresholds = {
        "min(avg+std,0.99)": float(min(score_avg + score_std, 0.99)),
        "avg": float(score_avg),
        "max(avg-std,0.01)": float(max(score_avg - score_std, 0.01)),
    }
    return score_dict, raw_dist_dict, score_avg, score_std, thresholds


def run_federated_kmeans_filter(client_cached, forget_local_indices, num_clusters=5, num_iters=10, top_num=100, seed=2025):
    _sync_cuda()
    t0 = time.perf_counter()

    centers_by_class = federated_kmeans_per_class(client_cached=client_cached, num_clusters=num_clusters, num_iters=num_iters, seed=seed)
    score_dict, raw_dist_dict, score_avg, score_std, thresholds = compute_fedkmeans_scores_for_client0(client_cached=client_cached, centers_by_class=centers_by_class)
    uu_local_indices = select_top_indices(score_dict=score_dict, candidate_indices=forget_local_indices, top_num=top_num, descending=True)

    _sync_cuda()
    t1 = time.perf_counter()
    return {
        "time": t1 - t0,
        "score_dict": score_dict,
        "raw_dist_dict": raw_dist_dict,
        "score_avg": score_avg,
        "score_std": score_std,
        "thresholds_reference": thresholds,
        "uu_local_indices": sorted(uu_local_indices),
    }


fedkm_result = run_federated_kmeans_filter(
    client_cached=client_cached,
    forget_local_indices=forget_local_indices,
    num_clusters=FEDKM_NUM_CLUSTERS,
    num_iters=FEDKM_ITERS,
    top_num=TOP_FILTER_NUM,
    seed=2025,
)
print(f"\n[TIME][Federated K-means Filter] {fedkm_result['time']:.6f} s")
print(f"[Federated K-means] client0 normalized score avg={fedkm_result['score_avg']:.6f}, std={fedkm_result['score_std']:.6f}")
print_filter_summary(
    method_name="Federated K-means Filter",
    uu_local_indices=fedkm_result["uu_local_indices"],
    score_dict={idx: fedkm_result["score_dict"][idx] for idx in forget_local_indices},
    extra_info={
        "top_num": TOP_FILTER_NUM,
        "num_clusters_per_class": FEDKM_NUM_CLUSTERS,
        "kmeans_iters": FEDKM_ITERS,
        "total_filter_time_s": f"{fedkm_result['time']:.6f}",
        "reference_min(avg+std,0.99)": f"{fedkm_result['thresholds_reference']['min(avg+std,0.99)']:.6f}",
        "reference_avg": f"{fedkm_result['thresholds_reference']['avg']:.6f}",
        "reference_max(avg-std,0.01)": f"{fedkm_result['thresholds_reference']['max(avg-std,0.01)']:.6f}",
    },
)
print_uu_detailed_table(
    method_name="Federated K-means Filter",
    uu_local_indices=fedkm_result["uu_local_indices"],
    metric_dict=fedkm_result["score_dict"],
    metric_name="score",
    conf_dict=forget_conf_dict_all,
    descending=True,
)
print(f'[Federated K-means Filter][UU mnist Real Idx List] = {fedkm_result["uu_local_indices"]}')

# ============================================================
# 11. filter 4: FedProto
# ============================================================
def _compute_thresholds_from_client0_scores(score_dict, client0_all_indices):
    client0_scores = np.array([score_dict[idx] for idx in client0_all_indices], dtype=np.float32)
    score_avg = float(np.mean(client0_scores))
    score_std = float(np.std(client0_scores))
    thresholds = {
        "min(avg+std,0.99)": float(min(score_avg + score_std, 0.99)),
        "avg": float(score_avg),
        "max(avg-std,0.01)": float(max(score_avg - score_std, 0.01)),
    }
    return score_avg, score_std, thresholds


def _normalize_distance_to_score(raw_dist_dict, ref_indices):
    ref_dists = np.array([raw_dist_dict[idx] for idx in ref_indices], dtype=np.float32)
    d_min = float(np.min(ref_dists))
    d_max = float(np.max(ref_dists))

    score_dict = {}
    if d_max - d_min < 1e-12:
        for idx in ref_indices:
            score_dict[idx] = 1.0
    else:
        for idx in ref_indices:
            norm_dist = (raw_dist_dict[idx] - d_min) / (d_max - d_min)
            score_dict[idx] = float(1.0 - norm_dist)
    return score_dict


def federated_proto_per_class(client_cached):
    prototypes_by_class = {}
    for cls in range(classes):
        proto_sum = None
        total_count = 0
        for cid in range(num_clients):
            feats = client_cached[cid]["feats"]
            labels_np = client_cached[cid]["labels_np"]
            mask_np = (labels_np == cls)
            feats_cls = feats[mask_np]
            if feats_cls.shape[0] == 0:
                continue
            local_proto = feats_cls.mean(dim=0)
            local_count = feats_cls.shape[0]
            if proto_sum is None:
                proto_sum = local_proto * local_count
            else:
                proto_sum += local_proto * local_count
            total_count += local_count

        global_proto = proto_sum / max(1, total_count)
        global_proto = F.normalize(global_proto.unsqueeze(0), p=2, dim=1).squeeze(0)
        prototypes_by_class[cls] = global_proto.detach().clone()
    return prototypes_by_class


def compute_fedproto_scores_for_client0(client_cached, prototypes_by_class):
    cid = FORGET_CLIENT_ID
    feats_client0 = client_cached[cid]["feats"]
    labels_client0 = client_cached[cid]["labels_np"]
    local_idx_client0 = client_cached[cid]["local_idx"]

    raw_dist_dict = {}
    for pos, loc_idx in enumerate(local_idx_client0):
        cls = int(labels_client0[pos])
        feat = feats_client0[pos]
        proto = prototypes_by_class[cls]
        dist = torch.norm(feat - proto, p=2).item()
        raw_dist_dict[loc_idx] = float(dist)

    score_dict = _normalize_distance_to_score(raw_dist_dict, local_idx_client0)
    score_avg, score_std, thresholds = _compute_thresholds_from_client0_scores(score_dict, local_idx_client0)
    return score_dict, raw_dist_dict, score_avg, score_std, thresholds


def run_fedproto_filter(client_cached, forget_local_indices, top_num=100):
    _sync_cuda()
    t0 = time.perf_counter()
    prototypes_by_class = federated_proto_per_class(client_cached)
    score_dict, raw_dist_dict, score_avg, score_std, thresholds = compute_fedproto_scores_for_client0(client_cached, prototypes_by_class)
    uu_local_indices = select_top_indices(score_dict=score_dict, candidate_indices=forget_local_indices, top_num=top_num, descending=True)
    _sync_cuda()
    t1 = time.perf_counter()
    return {
        "time": t1 - t0,
        "score_dict": score_dict,
        "raw_dist_dict": raw_dist_dict,
        "score_avg": score_avg,
        "score_std": score_std,
        "thresholds_reference": thresholds,
        "uu_local_indices": sorted(uu_local_indices),
    }


fedproto_result = run_fedproto_filter(client_cached=client_cached, forget_local_indices=forget_local_indices, top_num=TOP_FILTER_NUM)
print(f"\n[TIME][FedProto Filter] {fedproto_result['time']:.6f} s")
print(f"[FedProto] client0 normalized score avg={fedproto_result['score_avg']:.6f}, std={fedproto_result['score_std']:.6f}")
print_filter_summary(
    method_name="FedProto Filter",
    uu_local_indices=fedproto_result["uu_local_indices"],
    score_dict={idx: fedproto_result["score_dict"][idx] for idx in forget_local_indices},
    extra_info={
        "top_num": TOP_FILTER_NUM,
        "prototype_type": "single global class prototype",
        "total_filter_time_s": f"{fedproto_result['time']:.6f}",
        "reference_min(avg+std,0.99)": f"{fedproto_result['thresholds_reference']['min(avg+std,0.99)']:.6f}",
        "reference_avg": f"{fedproto_result['thresholds_reference']['avg']:.6f}",
        "reference_max(avg-std,0.01)": f"{fedproto_result['thresholds_reference']['max(avg-std,0.01)']:.6f}",
    },
)
print_uu_detailed_table(
    method_name="FedProto Filter",
    uu_local_indices=fedproto_result["uu_local_indices"],
    metric_dict=fedproto_result["score_dict"],
    metric_name="score",
    conf_dict=forget_conf_dict_all,
    descending=True,
)
print(f'[FedProto Filter][UU mnist Real Idx List] = {fedproto_result["uu_local_indices"]}')

# ============================================================
# 12. filter 5: FedPLVM
# ============================================================
def _kmeans_single_set(feats, num_clusters, num_iters, seed):
    n = feats.shape[0]
    if n == 0:
        return feats
    actual_k = min(num_clusters, n)
    centers = _sample_initial_centers(feats, actual_k, seed)
    for _ in range(num_iters):
        dist = torch.cdist(feats, centers, p=2)
        assign = dist.argmin(dim=1)
        new_centers = []
        for k in range(actual_k):
            mask_k = (assign == k)
            if mask_k.any():
                ck = feats[mask_k].mean(dim=0)
            else:
                ck = feats[random.randrange(n)]
            new_centers.append(ck)
        centers = torch.stack(new_centers, dim=0)
        centers = F.normalize(centers, p=2, dim=1)
    return centers


def fedplvm_global_multi_prototypes(client_cached, local_clusters=3, local_iters=5, global_clusters=5, global_iters=10, seed=2025):
    global_proto_by_class = {}
    for cls in range(classes):
        all_local_protos = []
        for cid in range(num_clients):
            feats = client_cached[cid]["feats"]
            labels_np = client_cached[cid]["labels_np"]
            mask_np = (labels_np == cls)
            feats_cls = feats[mask_np]
            if feats_cls.shape[0] == 0:
                continue
            local_centers = _kmeans_single_set(feats_cls, num_clusters=local_clusters, num_iters=local_iters, seed=seed + 100 * cls + cid)
            all_local_protos.append(local_centers)

        merged_local_protos = torch.cat(all_local_protos, dim=0)
        global_centers = _kmeans_single_set(merged_local_protos, num_clusters=global_clusters, num_iters=global_iters, seed=seed + 1000 + cls)
        global_proto_by_class[cls] = global_centers.detach().clone()
    return global_proto_by_class


def compute_fedplvm_scores_for_client0(client_cached, global_proto_by_class):
    cid = FORGET_CLIENT_ID
    feats_client0 = client_cached[cid]["feats"]
    labels_client0 = client_cached[cid]["labels_np"]
    local_idx_client0 = client_cached[cid]["local_idx"]

    raw_dist_dict = {}
    for pos, loc_idx in enumerate(local_idx_client0):
        cls = int(labels_client0[pos])
        feat = feats_client0[pos:pos + 1]
        centers = global_proto_by_class[cls]
        dist = torch.cdist(feat, centers, p=2).min().item()
        raw_dist_dict[loc_idx] = float(dist)

    score_dict = _normalize_distance_to_score(raw_dist_dict, local_idx_client0)
    score_avg, score_std, thresholds = _compute_thresholds_from_client0_scores(score_dict, local_idx_client0)
    return score_dict, raw_dist_dict, score_avg, score_std, thresholds


def run_fedplvm_filter(client_cached, forget_local_indices, local_clusters=3, local_iters=5, global_clusters=5, global_iters=10, top_num=100, seed=2025):
    _sync_cuda()
    t0 = time.perf_counter()
    global_proto_by_class = fedplvm_global_multi_prototypes(
        client_cached=client_cached,
        local_clusters=local_clusters,
        local_iters=local_iters,
        global_clusters=global_clusters,
        global_iters=global_iters,
        seed=seed,
    )
    score_dict, raw_dist_dict, score_avg, score_std, thresholds = compute_fedplvm_scores_for_client0(client_cached=client_cached, global_proto_by_class=global_proto_by_class)
    uu_local_indices = select_top_indices(score_dict=score_dict, candidate_indices=forget_local_indices, top_num=top_num, descending=True)
    _sync_cuda()
    t1 = time.perf_counter()
    return {
        "time": t1 - t0,
        "score_dict": score_dict,
        "raw_dist_dict": raw_dist_dict,
        "score_avg": score_avg,
        "score_std": score_std,
        "thresholds_reference": thresholds,
        "uu_local_indices": sorted(uu_local_indices),
    }


fedplvm_result = run_fedplvm_filter(
    client_cached=client_cached,
    forget_local_indices=forget_local_indices,
    local_clusters=FEDPLVM_LOCAL_CLUSTERS,
    local_iters=FEDPLVM_LOCAL_ITERS,
    global_clusters=FEDPLVM_GLOBAL_CLUSTERS,
    global_iters=FEDPLVM_GLOBAL_ITERS,
    top_num=TOP_FILTER_NUM,
    seed=2025,
)
print(f"\n[TIME][FedPLVM Filter] {fedplvm_result['time']:.6f} s")
print(f"[FedPLVM] client0 normalized score avg={fedplvm_result['score_avg']:.6f}, std={fedplvm_result['score_std']:.6f}")
print_filter_summary(
    method_name="FedPLVM Filter",
    uu_local_indices=fedplvm_result["uu_local_indices"],
    score_dict={idx: fedplvm_result["score_dict"][idx] for idx in forget_local_indices},
    extra_info={
        "top_num": TOP_FILTER_NUM,
        "local_clusters_per_client_class": FEDPLVM_LOCAL_CLUSTERS,
        "local_kmeans_iters": FEDPLVM_LOCAL_ITERS,
        "global_clusters_per_class": FEDPLVM_GLOBAL_CLUSTERS,
        "global_kmeans_iters": FEDPLVM_GLOBAL_ITERS,
        "total_filter_time_s": f"{fedplvm_result['time']:.6f}",
        "reference_min(avg+std,0.99)": f"{fedplvm_result['thresholds_reference']['min(avg+std,0.99)']:.6f}",
        "reference_avg": f"{fedplvm_result['thresholds_reference']['avg']:.6f}",
        "reference_max(avg-std,0.01)": f"{fedplvm_result['thresholds_reference']['max(avg-std,0.01)']:.6f}",
    },
)
print_uu_detailed_table(
    method_name="FedPLVM Filter",
    uu_local_indices=fedplvm_result["uu_local_indices"],
    metric_dict=fedplvm_result["score_dict"],
    metric_name="score",
    conf_dict=forget_conf_dict_all,
    descending=True,
)
print(f'[FedPLVM Filter][UU mnist Real Idx List] = {fedplvm_result["uu_local_indices"]}')

# ============================================================
# 13. summary of five filters
# ============================================================
print("\n" + "=" * 100)
print("[SUMMARY OF FILTERS]")
print("=" * 100)
print(f"Single SimHash UU count                : {len(simhash_result['uu_local_indices'])} | total_time={simhash_result['time']:.6f}s")
print(f"SimHash + Confidence UU count          : {len(simhash_conf_result['uu_local_indices'])} | total_time={simhash_conf_result['time']:.6f}s")
print(f"Federated K-means UU count             : {len(fedkm_result['uu_local_indices'])} | total_time={fedkm_result['time']:.6f}s")
print(f"FedProto UU count                      : {len(fedproto_result['uu_local_indices'])} | total_time={fedproto_result['time']:.6f}s")
print(f"FedPLVM UU count                       : {len(fedplvm_result['uu_local_indices'])} | total_time={fedplvm_result['time']:.6f}s")

print("\n" + "=" * 100)
print("[INDEX LIST SUMMARY]")
print("=" * 100)
print(f"[Forget][mnist Real Idx List] = {forget_local_indices}")
print(f"[Single SimHash][UU mnist Real Idx List] = {simhash_result['uu_local_indices']}")
print(f"[SimHash + Confidence][UU mnist Real Idx List] = {simhash_conf_result['uu_local_indices']}")
print(f"[Federated K-means][UU mnist Real Idx List] = {fedkm_result['uu_local_indices']}")
print(f"[FedProto][UU mnist Real Idx List] = {fedproto_result['uu_local_indices']}")
print(f"[FedPLVM][UU mnist Real Idx List] = {fedplvm_result['uu_local_indices']}")

# ============================================================
# 14. phase helpers for P2-P7 retrain/evaluation
# ============================================================
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


def federated_retrain_on_given_indices(phase_tag, phase_client_indices, init_state, rounds=rounds, local_epochs=local_epochs, batch_size=batch_size, lr=1e-3):
    reset_all_seeds(RETRAIN_INIT_SEED)
    model_phase = build_model().to(device)
    model_phase.load_state_dict(init_state)

    print(f"\n[FL-{phase_tag}] start federated retraining...")
    for rnd in range(1, rounds + 1):
        client_states = []
        sizes = []
        weighted_loss_sum = 0.0
        total_samples = 0

        for cid in range(num_clients):
            client_model = build_model().to(device)
            client_model.load_state_dict(model_phase.state_dict())
            train_loader = get_client_loader_from_given_indices(phase_client_indices, cid, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
            avg_loss_client = local_train_and_report_loss(client_model, train_loader, epochs=local_epochs, lr=lr)
            n_i = len(phase_client_indices[cid])
            weighted_loss_sum += avg_loss_client * n_i
            total_samples += n_i
            client_states.append({k: v.detach().cpu() for k, v in client_model.state_dict().items()})
            sizes.append(n_i)
            del client_model
            torch.cuda.empty_cache()

        aggregate_fedavg(model_phase, client_states, sizes)
        round_train_loss = weighted_loss_sum / max(1, total_samples)
        print(f"[{phase_tag}][Round {rnd}] Train Loss: {round_train_loss:.4f}")

    return model_phase


def metric_dict(acc, loss):
    return {"acc": float(acc), "loss": float(loss)}


def mia_metric_dict(acc, f1):
    return {"acc": float(acc), "f1": float(f1)}


def print_model_metric_line(prefix, metric_item):
    print(f"{prefix} Accuracy: {metric_item['acc']:.2f}% | Avg Loss: {metric_item['loss']:.4f}")


def print_mia_metric_line(prefix, metric_item):
    print(f"{prefix} Attack Acc: {metric_item['acc']:.2f}% | F1: {metric_item['f1']:.2f}%")


def evaluate_loader_with_log(target_model, loader, phase_tag, set_name):
    if loader is None:
        print(f"[{phase_tag}][{set_name}] Skip (empty set).")
        return metric_dict(float('nan'), float('nan'))
    acc, loss = evaluate(target_model, loader)
    print(f"[{phase_tag}][{set_name}] Accuracy: {acc:.2f}% | Avg Loss: {loss:.4f}")
    return metric_dict(acc, loss)


def evaluate_named_attack_with_log(target_model, attack_model, eval_indices, phase_tag, set_name, true_membership_label=None, true_membership_map=None):
    if eval_indices is None or len(eval_indices) == 0:
        print(f"[MIA-{phase_tag}][{set_name}] Skip (empty set).")
        return mia_metric_dict(float('nan'), float('nan'))
    acc, f1 = eval_named_set_attack_acc_f1(
        target_model=target_model,
        attack_model=attack_model,
        eval_indices=eval_indices,
        true_membership_label=true_membership_label,
        true_membership_map=true_membership_map,
        phase_tag=f"MIA-{phase_tag}",
        set_name=set_name,
    )
    return mia_metric_dict(acc, f1)


def evaluate_phase_base_sets(phase_tag, target_model, forget_mia_label=None, forget_mia_label_map=None):
    print(f"\n[{phase_tag}] base-set evaluation")
    base_metrics = OrderedDict()
    base_metrics["test"] = evaluate_loader_with_log(target_model, test_loader, phase_tag, "Test All")
    base_metrics["remain"] = evaluate_loader_with_log(
        target_model, remaining_excluding_forget_loader, phase_tag, "Remain Excluding Forget"
    )
    base_metrics["forget"] = evaluate_loader_with_log(target_model, forget_loader, phase_tag, "Forget")

    print(f"\n[MIA-{phase_tag}] Evaluate {phase_tag} target with fixed attack model trained in Phase1...")
    X_attack_px, y_attack_px = build_attack_features_for_target(target_model, member_loader, non_member_loader)
    attack_val_loader_px = build_fixed_val_loader_from_features(X_attack_px, y_attack_px, attack_val_idx, batch_size=64)
    eval_attack_three_metrics(attack_model_p1, attack_val_loader_px, name=f"MIA-{phase_tag}")

    base_metrics["forget_mia"] = evaluate_named_attack_with_log(
        target_model=target_model,
        attack_model=attack_model_p1,
        eval_indices=forget_local_indices,
        phase_tag=phase_tag,
        set_name="Forget",
        true_membership_label=forget_mia_label,
        true_membership_map=forget_mia_label_map,
    )
    return base_metrics


def evaluate_method_specific_uu_set(phase_tag, method_name, target_model, uu_indices, uu_membership_label):
    uu_loader = make_loader_from_indices(uu_indices)
    model_metrics = evaluate_loader_with_log(
        target_model=target_model,
        loader=uu_loader,
        phase_tag=phase_tag,
        set_name=f"{method_name} UU Set",
    )
    mia_metrics = evaluate_named_attack_with_log(
        target_model=target_model,
        attack_model=attack_model_p1,
        eval_indices=uu_indices,
        phase_tag=phase_tag,
        set_name=f"{method_name} UU Set",
        true_membership_label=uu_membership_label,
    )
    return {
        "model": model_metrics,
        "mia": mia_metrics,
    }


def abs_delta(a, b):
    if np.isnan(a) or np.isnan(b):
        return float('nan')
    return abs(float(a) - float(b))


def print_comparison_block(method_name, no_filter_tag, no_filter_metrics, filter_tag, filter_metrics):
    print("\n" + "-" * 120)
    print(f"[FINAL SUMMARY][{method_name}] {no_filter_tag} vs {filter_tag}")
    print("-" * 120)

    print_model_metric_line(f"{no_filter_tag}[Test All]              ", no_filter_metrics["test"])
    print_model_metric_line(f"{no_filter_tag}[Remain Excluding Forget]", no_filter_metrics["remain"])
    print_model_metric_line(f"{no_filter_tag}[UU Set]                ", no_filter_metrics["uu"])
    print_mia_metric_line(f"{no_filter_tag}[UU Set]                ", no_filter_metrics["uu_mia"])

    print_model_metric_line(f"{filter_tag}[Test All]              ", filter_metrics["test"])
    print_model_metric_line(f"{filter_tag}[Remain Excluding Forget]", filter_metrics["remain"])
    print_model_metric_line(f"{filter_tag}[UU Set]                ", filter_metrics["uu"])
    print_mia_metric_line(f"{filter_tag}[UU Set]                ", filter_metrics["uu_mia"])

    print(
        f"|Δ| Test All              : "
        f"Acc={abs_delta(no_filter_metrics['test']['acc'], filter_metrics['test']['acc']):.2f}, "
        f"Loss={abs_delta(no_filter_metrics['test']['loss'], filter_metrics['test']['loss']):.4f}"
    )
    print(
        f"|Δ| Remain Excluding Forget: "
        f"Acc={abs_delta(no_filter_metrics['remain']['acc'], filter_metrics['remain']['acc']):.2f}, "
        f"Loss={abs_delta(no_filter_metrics['remain']['loss'], filter_metrics['remain']['loss']):.4f}"
    )
    print(
        f"|Δ| UU Set                : "
        f"Acc={abs_delta(no_filter_metrics['uu']['acc'], filter_metrics['uu']['acc']):.2f}, "
        f"Loss={abs_delta(no_filter_metrics['uu']['loss'], filter_metrics['uu']['loss']):.4f}"
    )
    print(
        f"|Δ| UU-Set MIA            : "
        f"Acc={abs_delta(no_filter_metrics['uu_mia']['acc'], filter_metrics['uu_mia']['acc']):.2f}, "
        f"F1={abs_delta(no_filter_metrics['uu_mia']['f1'], filter_metrics['uu_mia']['f1']):.2f}"
    )


# fixed retrain init for P2-P7
base_retrain_state = build_fixed_retrain_init_state(RETRAIN_INIT_SEED)
print(f"\n[Retrain Init] fixed retrain init state created with seed={RETRAIN_INIT_SEED}")

method_results = OrderedDict([
    ("Phase3", {"name": "Single SimHash", "uu_indices": simhash_result["uu_local_indices"], "score_dict": simhash_result["avg_neighbors"]}),
    ("Phase4", {"name": "SimHash + Confidence", "uu_indices": simhash_conf_result["uu_local_indices"], "score_dict": simhash_conf_result["avg_neighbors"]}),
    ("Phase5", {"name": "Federated K-means", "uu_indices": fedkm_result["uu_local_indices"], "score_dict": fedkm_result["score_dict"]}),
    ("Phase6", {"name": "FedProto", "uu_indices": fedproto_result["uu_local_indices"], "score_dict": fedproto_result["score_dict"]}),
    ("Phase7", {"name": "FedPLVM", "uu_indices": fedplvm_result["uu_local_indices"], "score_dict": fedplvm_result["score_dict"]}),
])

for phase_key in method_results:
    uu_indices = method_results[phase_key]["uu_indices"]
    method_results[phase_key]["uu_loader"] = make_loader_from_indices(uu_indices)
    method_results[phase_key]["nu_indices"] = sorted(list(set(forget_local_indices) - set(uu_indices)))
    method_results[phase_key]["filter_ratio"] = len(uu_indices) / max(1, len(forget_local_indices))

comparison_summary = OrderedDict()

# ============================================================
# 15. Phase2: delete all forget samples and retrain
# ============================================================
forget_set = set(forget_local_indices)
client_indices_phase2 = make_phase_client_indices_from_removed_set(removed_set=forget_set, phase_tag="Retrain-Phase2")
global_model_p2 = federated_retrain_on_given_indices(phase_tag="Phase2", phase_client_indices=client_indices_phase2, init_state=base_retrain_state)

phase2_base_metrics = evaluate_phase_base_sets(
    phase_tag="Phase2",
    target_model=global_model_p2,
    forget_mia_label=0,
)

print("\n" + "=" * 100)
print("[Phase2] evaluate each method-specific UU set on the no-filtering retrained model")
print("=" * 100)
for phase_key, info in method_results.items():
    method_name = info["name"]
    print("\n" + "-" * 100)
    print(f"[Phase2][{method_name}] method-specific UU-set evaluation on P2 model")
    print("-" * 100)
    p2_uu_metrics = evaluate_method_specific_uu_set(
        phase_tag="Phase2",
        method_name=method_name,
        target_model=global_model_p2,
        uu_indices=info["uu_indices"],
        uu_membership_label=1,
    )

    method_results[phase_key]["p2_metrics"] = {
        "test": dict(phase2_base_metrics["test"]),
        "remain": dict(phase2_base_metrics["remain"]),
        "uu": dict(p2_uu_metrics["model"]),
        "uu_mia": dict(p2_uu_metrics["mia"]),
    }
    comparison_summary[phase_key] = {
        "method_name": method_name,
        "p2": method_results[phase_key]["p2_metrics"],
    }

# ============================================================
# 16. Phase3-Phase7: keep UU, delete NU, then retrain
# ============================================================
for phase_key, info in method_results.items():
    method_name = info["name"]
    uu_indices = info["uu_indices"]
    nu_indices = info["nu_indices"]
    print("\n" + "=" * 100)
    print(f"[{phase_key}] {method_name}")
    print("=" * 100)
    print(f"UU kept: {len(uu_indices)} | NU deleted: {len(nu_indices)} | Filtering Ratio: {info['filter_ratio']:.4f}")

    phase_client_indices = make_phase_client_indices_from_removed_set(removed_set=set(nu_indices), phase_tag=f"Retrain-{phase_key}")
    global_model_phase = federated_retrain_on_given_indices(phase_tag=phase_key, phase_client_indices=phase_client_indices, init_state=base_retrain_state)

    forget_membership_map = {idx: 1 for idx in uu_indices}
    forget_membership_map.update({idx: 0 for idx in nu_indices})

    phase_base_metrics = evaluate_phase_base_sets(
        phase_tag=phase_key,
        target_model=global_model_phase,
        forget_mia_label_map=forget_membership_map,
    )
    phase_uu_metrics = evaluate_method_specific_uu_set(
        phase_tag=phase_key,
        method_name=method_name,
        target_model=global_model_phase,
        uu_indices=uu_indices,
        uu_membership_label=1,
    )

    method_results[phase_key]["phase_metrics"] = {
        "test": dict(phase_base_metrics["test"]),
        "remain": dict(phase_base_metrics["remain"]),
        "uu": dict(phase_uu_metrics["model"]),
        "uu_mia": dict(phase_uu_metrics["mia"]),
    }
    comparison_summary[phase_key]["filtered_phase"] = phase_key
    comparison_summary[phase_key]["filtered"] = method_results[phase_key]["phase_metrics"]

# ============================================================
# 17. final summary
# ============================================================
print("\n" + "=" * 120)
print("[FINAL SUMMARY]")
print("=" * 120)
print(f"Phase2 delete-all forget size : {len(forget_local_indices)}")
for phase_key, info in method_results.items():
    print(
        f"{phase_key} | {info['name']:<22} | UU kept={len(info['uu_indices']):4d} | "
        f"NU deleted={len(info['nu_indices']):4d} | filtering_ratio={info['filter_ratio']:.4f}"
    )

for phase_key, info in comparison_summary.items():
    print_comparison_block(
        method_name=info["method_name"],
        no_filter_tag="Phase2(no filtering)",
        no_filter_metrics=info["p2"],
        filter_tag=info["filtered_phase"],
        filter_metrics=info["filtered"],
    )
