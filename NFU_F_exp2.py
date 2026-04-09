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
# parser = argparse.ArgumentParser()
# parser.add_argument("--seed", type=int, required=True, help="random seed")
# parser.add_argument("--forget_per_class", type=int, required=True, help="forget samples per class")
# args = parser.parse_args()
#
# RANDOMSEED = args.seed
# FORGET_PER_CLASS = args.forget_per_class


RANDOMSEED = 30
FORGET_PER_CLASS = 30
RETRAIN_INIT_SEED = RANDOMSEED + 99999
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
# 1. config
# ============================================================
num_clients = 3
classes = 10
FIRST_FOUR_PER_CLASS = 1000

FORGET_CLIENT_ID = 0
# FORGET_PER_CLASS = 30

rounds = 50                       # public config total rounds = 50
local_epochs = 1
batch_size = 100                  # batch size = 100
LOCAL_BATCHES_PER_ROUND = 10      # local 10 batches / round
SAMPLES_PER_ROUND = LOCAL_BATCHES_PER_ROUND * batch_size

# LSH + confidence filter
E2LSH_K = 32
E2LSH_L = 16
NEIGHBOR_THRESHOLD = 80
CONFIDENCE_THRESHOLD = 0.9

NUM_WORKERS = 0

print(f"num_clients={num_clients}, classes={classes}, FIRST_FOUR_PER_CLASS={FIRST_FOUR_PER_CLASS}")
print(f"FORGET_CLIENT_ID={FORGET_CLIENT_ID}, FORGET_PER_CLASS={FORGET_PER_CLASS}")
print(f"rounds={rounds}, local_epochs={local_epochs}, batch_size={batch_size}, LOCAL_BATCHES_PER_ROUND={LOCAL_BATCHES_PER_ROUND}")
print(f"SAMPLES_PER_ROUND={SAMPLES_PER_ROUND}")
print(f"cosLSH_K={E2LSH_K}, cosLSH_L={E2LSH_L}, NEIGHBOR_THRESHOLD={NEIGHBOR_THRESHOLD}, CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}")
print(f"NUM_WORKERS={NUM_WORKERS}")
print(f"RANDOMSEED={RANDOMSEED} | RETRAIN_INIT_SEED={RETRAIN_INIT_SEED} | RETRAIN_SCHEDULE_SEED={RETRAIN_SCHEDULE_SEED}")

# ============================================================
# 2. data
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data_full = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

train_data_all = train_data_full
targets_full = np.array(train_data_full.targets)
selected_targets = targets_full
selected_indices = list(range(len(train_data_full)))

per_class_total = len(train_data_all) // classes
print(f"[Info] use full training set: {len(train_data_all)}")
print(f"[Info] per_class_total={per_class_total}")

class_to_localidx = {c: np.where(selected_targets == c)[0].tolist() for c in range(classes)}
for c in range(classes):
    np.random.shuffle(class_to_localidx[c])

target_per_class_for_regular_clients = FIRST_FOUR_PER_CLASS
max_regular_clients_by_class = min(len(class_to_localidx[c]) // target_per_class_for_regular_clients for c in range(classes))
num_regular_clients = min(num_clients - 1, max_regular_clients_by_class)

print(
    f"[Partition Rule] first {num_regular_clients} client(s) each get "
    f"{target_per_class_for_regular_clients} samples/class when possible; "
    f"all remaining samples go to the last client {num_clients - 1}."
)
if num_regular_clients < num_clients - 1:
    print(
        f"[Partition Rule] clients {num_regular_clients}..{num_clients - 2} (if any) "
        f"receive 0 samples because the remainder is merged into the last client."
    )

client_indices = [[] for _ in range(num_clients)]
for c in range(classes):
    cls_indices = class_to_localidx[c]
    ptr = 0

    for k in range(num_regular_clients):
        take_k = min(target_per_class_for_regular_clients, len(cls_indices) - ptr)
        if take_k > 0:
            client_indices[k].extend(cls_indices[ptr:ptr + take_k])
            ptr += take_k

    remaining_cls_indices = cls_indices[ptr:]
    if len(remaining_cls_indices) > 0:
        client_indices[num_clients - 1].extend(remaining_cls_indices)

assigned_total = sum(len(lst) for lst in client_indices)
assert assigned_total == len(train_data_all), (
    f"Assigned total {assigned_total} != full train size {len(train_data_all)}"
)

original_client_indices_backup_phase1 = [list(lst) for lst in client_indices]

test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# ============================================================
# 3. model utils
# ============================================================
def build_model():
    m = models.resnet18(num_classes=classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


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
        return float('nan'), float('nan')
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
# 4. round schedule utils (FATS-style public schedule)
# ============================================================
def build_round_schedule_for_client(client_pool_indices, total_rounds, samples_per_round, seed):
    """
    Build a deterministic public round schedule.
    Each client uses exactly `samples_per_round` samples per round.
    We repeatedly shuffle the whole local pool and concatenate passes until
    enough samples are available, then split into rounds.
    This guarantees all client-0 samples participate in the 50-round public schedule.
    """
    pool = list(client_pool_indices)
    if len(pool) == 0:
        return [[] for _ in range(total_rounds)]

    total_needed = total_rounds * samples_per_round
    stream = []
    pass_id = 0
    while len(stream) < total_needed:
        perm = np.array(pool, dtype=np.int64)
        rng_pass = np.random.default_rng(seed + pass_id * 7919)
        rng_pass.shuffle(perm)
        stream.extend(perm.tolist())
        pass_id += 1

    stream = stream[:total_needed]
    round_lists = [stream[r * samples_per_round:(r + 1) * samples_per_round] for r in range(total_rounds)]
    return round_lists


def build_public_round_schedule(client_indices_ref, total_rounds, samples_per_round, base_seed):
    schedule = {}
    for cid in range(num_clients):
        schedule[cid] = build_round_schedule_for_client(
            client_pool_indices=client_indices_ref[cid],
            total_rounds=total_rounds,
            samples_per_round=samples_per_round,
            seed=base_seed + cid * 100003,
        )
    return schedule


def summarize_first_rounds_from_schedule(schedule, client_indices_ref):
    first_round_seen = {}
    seen_count = Counter()
    for cid in range(num_clients):
        for rnd in range(1, len(schedule[cid]) + 1):
            for idx in schedule[cid][rnd - 1]:
                seen_count[idx] += 1
                if idx not in first_round_seen:
                    first_round_seen[idx] = rnd

    for cid in range(num_clients):
        local_pool = client_indices_ref[cid]
        unseen = [idx for idx in local_pool if idx not in first_round_seen]
        print(
            f"[Schedule Summary][Client {cid}] total_local={len(local_pool)} | "
            f"seen_unique={len(local_pool) - len(unseen)} | unseen_unique={len(unseen)}"
        )
    return first_round_seen, seen_count


def make_loader_for_round_indices(round_indices, batch_size=100, num_workers=NUM_WORKERS):
    return DataLoader(
        Subset(train_data_all, list(round_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================
# 5. Phase1 FL training under public 50-round schedule
# ============================================================
def federated_train_with_public_schedule(phase_tag, init_state=None, client_indices_ref=None, schedule=None, lr=1e-3):
    if init_state is None:
        reset_all_seeds(RANDOMSEED)
        model_phase = build_model().to(device)
    else:
        reset_all_seeds(RETRAIN_INIT_SEED)
        model_phase = build_model().to(device)
        model_phase.load_state_dict(init_state)

    print(f"\n[FL-{phase_tag}] start federated training/retraining with public schedule...")
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

            round_indices = schedule[cid][rnd - 1]
            train_loader = make_loader_for_round_indices(round_indices, batch_size=batch_size, num_workers=NUM_WORKERS)
            avg_loss_client = local_train_and_report_loss(client_model, train_loader, epochs=local_epochs, lr=lr)

            n_i = len(round_indices)
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


p1_public_schedule = build_public_round_schedule(
    client_indices_ref=original_client_indices_backup_phase1,
    total_rounds=rounds,
    samples_per_round=SAMPLES_PER_ROUND,
    base_seed=RANDOMSEED + 123456,
)
p1_first_round_seen, p1_seen_count = summarize_first_rounds_from_schedule(
    schedule=p1_public_schedule,
    client_indices_ref=original_client_indices_backup_phase1,
)

global_model, p1_train_time = federated_train_with_public_schedule(
    phase_tag="Phase1",
    init_state=None,
    client_indices_ref=original_client_indices_backup_phase1,
    schedule=p1_public_schedule,
    lr=1e-3,
)
print(f"[Phase1] public-schedule training time: {p1_train_time:.6f} s")


# ============================================================
# 6. choose forget set from participating data only
# ============================================================
client0_participated_indices = [idx for idx in original_client_indices_backup_phase1[FORGET_CLIENT_ID] if idx in p1_first_round_seen]
class_to_client0_participated = {c: [] for c in range(classes)}
for loc_idx in client0_participated_indices:
    c = int(selected_targets[loc_idx])
    class_to_client0_participated[c].append(loc_idx)

forget_local_indices = []
rng_forget = np.random.default_rng(RANDOMSEED + 2025)
for c in range(classes):
    lst = list(class_to_client0_participated[c])
    rng_forget.shuffle(lst)
    take = lst[:FORGET_PER_CLASS]
    assert len(take) == FORGET_PER_CLASS, f"client {FORGET_CLIENT_ID} participated class {c} insufficient"
    forget_local_indices.extend(take)

forget_local_indices = sorted(forget_local_indices)
forget_loader = make_loader_from_indices(forget_local_indices)
remaining_excluding_forget_indices = sorted(list(set(range(len(train_data_all))) - set(forget_local_indices)))
remaining_excluding_forget_loader = make_loader_from_indices(remaining_excluding_forget_indices)
print(f"\n[Forget] client{FORGET_CLIENT_ID} forget size: {len(forget_local_indices)} (per class {FORGET_PER_CLASS})")
print(f"[Forget][CIFAR10 Real Idx List] = {forget_local_indices}")
print(f"[Remain Excluding Forget] size: {len(remaining_excluding_forget_indices)}")

acc_test1, loss_test1 = evaluate(global_model, test_loader)
acc_forget1, loss_forget1 = evaluate(global_model, forget_loader)
acc_remain1, loss_remain1 = evaluate(global_model, remaining_excluding_forget_loader)
print(f"\n[Phase1][Test All] Accuracy: {acc_test1:.2f}% | Avg Loss: {loss_test1:.4f}")
print(f"[Phase1][Remain Excluding Forget] Accuracy: {acc_remain1:.2f}% | Avg Loss: {loss_remain1:.4f}")
print(f"[Phase1][Forget]  Accuracy: {acc_forget1:.2f}% | Avg Loss: {loss_forget1:.4f}")


# ============================================================
# 7. MIA fixed data / attack model
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
# 8. shared feature precompute for LSH + confidence
# ============================================================
FEATURE_BACKBONE = global_model


class ResNet18Feature(nn.Module):
    def __init__(self, resnet18_model):
        super().__init__()
        m = resnet18_model
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


@torch.no_grad()
def precompute_client_features(client_id, feature_net, batch_size=256, normalize=True):
    feature_net.eval()
    loader = DataLoader(
        Subset(train_data_all, original_client_indices_backup_phase1[client_id]),
        batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
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


print("\n[Shared Precompute] forward embedding + L2 normalize (NOT timed for filters) ...")
feature_net = ResNet18Feature(FEATURE_BACKBONE).to(device)

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

    print(f"rank\tcifar10_idx\t{metric_name}\tpred_label\ttrue_label\tmax_confidence\ttrue_label_confidence")
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
# 9. SimHash + Confidence filter
# ============================================================
@torch.no_grad()
def server_make_coslsh_params(L=8, K=16, d=512, seed=2025, device="cpu"):
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
    cos_params = server_make_coslsh_params(L=E2LSH_L, K=E2LSH_K, d=512, seed=seed, device=device)

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
    }


def run_simhash_confidence_filter(client_cached, forget_local_indices, neighbor_threshold=80, conf_threshold=0.9, seed=2025):
    simhash_result_local = run_single_simhash_filter(client_cached=client_cached, forget_local_indices=forget_local_indices, top_num=len(forget_local_indices), seed=seed)

    _sync_cuda()
    t_conf_start = time.perf_counter()

    forget_conf_scores = {idx: forget_conf_dict_all[idx]["true_label_confidence"] for idx in forget_local_indices}
    eligible_indices = [
        idx for idx in forget_local_indices
        if simhash_result_local["avg_neighbors"][idx] > neighbor_threshold
        and forget_conf_dict_all[idx]["true_label_confidence"] >= conf_threshold
    ]
    refined_uu = sorted(eligible_indices)

    _sync_cuda()
    t_conf_end = time.perf_counter()
    confidence_refine_time = t_conf_end - t_conf_start

    return {
        "uu_local_indices": refined_uu,
        "time": simhash_result_local["build_time"] + simhash_result_local["query_time"] + confidence_refine_time,
        "build_time": simhash_result_local["build_time"],
        "query_time": simhash_result_local["query_time"],
        "confidence_refine_time": confidence_refine_time,
        "avg_neighbors": simhash_result_local["avg_neighbors"],
        "forget_conf_scores": forget_conf_scores,
        "neighbor_threshold": float(neighbor_threshold),
        "conf_threshold": float(conf_threshold),
        "eligible_after_threshold_and_confidence": len(refined_uu),
    }


simhash_conf_result = run_simhash_confidence_filter(
    client_cached=client_cached,
    forget_local_indices=forget_local_indices,
    neighbor_threshold=NEIGHBOR_THRESHOLD,
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
        "neighbor_threshold": f"{simhash_conf_result['neighbor_threshold']:.6f}",
        "confidence_threshold": f"{simhash_conf_result['conf_threshold']:.6f}",
        "eligible_after_threshold_and_confidence": simhash_conf_result["eligible_after_threshold_and_confidence"],
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
print(f'[SimHash + Confidence Filter][UU CIFAR10 Real Idx List] = {simhash_conf_result["uu_local_indices"]}')


# ============================================================
# 10. participation-position summary for Forget and UU
# ============================================================
def print_first_round_summary(set_name, indices, first_round_map):
    details = []
    unseen = []
    pos_sum = 0
    for idx in sorted(indices):
        rnd = first_round_map.get(idx, None)
        if rnd is None:
            unseen.append(idx)
        else:
            details.append((idx, int(rnd)))
            pos_sum += int(rnd)
    print("\n" + "=" * 100)
    print(f"[Phase1 Public Schedule][{set_name}] first participation round summary")
    print("=" * 100)
    print(f"[{set_name}][First Round List] = {details}")
    print(f"[{set_name}][First Round Sum] = {pos_sum}")
    print(f"[{set_name}][Seen Count] = {len(details)} / {len(indices)}")
    if len(unseen) > 0:
        print(f"[{set_name}][Unseen Idx List] = {sorted(unseen)}")
    return details, pos_sum, unseen


forget_pos_details, forget_pos_sum, forget_unseen = print_first_round_summary(
    set_name="Forget",
    indices=forget_local_indices,
    first_round_map=p1_first_round_seen,
)

uu_local_indices = simhash_conf_result["uu_local_indices"]
nu_local_indices = sorted(list(set(forget_local_indices) - set(uu_local_indices)))
filter_ratio = len(uu_local_indices) / max(1, len(forget_local_indices))

uu_pos_details, uu_pos_sum, uu_unseen = print_first_round_summary(
    set_name="UU",
    indices=uu_local_indices,
    first_round_map=p1_first_round_seen,
)
print(f"\n[Filter Ratio] {len(uu_local_indices)} / {len(forget_local_indices)} = {filter_ratio:.6f}")
print(f"[NU][CIFAR10 Real Idx List] = {nu_local_indices}")


# ============================================================
# 11. phase helpers for P2/P3 retrain/evaluation
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


def metric_dict(acc, loss):
    return {"acc": float(acc), "loss": float(loss)}


def mia_metric_dict(acc, f1):
    return {"acc": float(acc), "f1": float(f1)}


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
        phase_tag=f"MIA-{phase_tag}",
        set_name=set_name,
        true_membership_label=true_membership_label,
        true_membership_map=true_membership_map,
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


def print_model_metric_line(prefix, metric_item):
    print(f"{prefix} Accuracy: {metric_item['acc']:.2f}% | Avg Loss: {metric_item['loss']:.4f}")


def print_mia_metric_line(prefix, metric_item):
    print(f"{prefix} Attack Acc: {metric_item['acc']:.2f}% | F1: {metric_item['f1']:.2f}%")


def print_comparison_block(method_name, no_filter_tag, no_filter_metrics, filter_tag, filter_metrics):
    print("\n" + "-" * 120)
    print(f"[FINAL SUMMARY][{method_name}] {no_filter_tag} vs {filter_tag}")
    print("-" * 120)

    print_model_metric_line(f"{no_filter_tag}[Test All]              ", no_filter_metrics["test"])
    print_model_metric_line(f"{no_filter_tag}[Remain Excluding Forget]", no_filter_metrics["remain"])
    print_model_metric_line(f"{no_filter_tag}[Forget]                ", no_filter_metrics["forget"])
    print_mia_metric_line(f"{no_filter_tag}[Forget]                ", no_filter_metrics["forget_mia"])
    print_model_metric_line(f"{no_filter_tag}[UU Set]                ", no_filter_metrics["uu"])
    print_mia_metric_line(f"{no_filter_tag}[UU Set]                ", no_filter_metrics["uu_mia"])

    print_model_metric_line(f"{filter_tag}[Test All]              ", filter_metrics["test"])
    print_model_metric_line(f"{filter_tag}[Remain Excluding Forget]", filter_metrics["remain"])
    print_model_metric_line(f"{filter_tag}[Forget]                ", filter_metrics["forget"])
    print_mia_metric_line(f"{filter_tag}[Forget]                ", filter_metrics["forget_mia"])
    print_model_metric_line(f"{filter_tag}[UU Set]                ", filter_metrics["uu"])
    print_mia_metric_line(f"{filter_tag}[UU Set]                ", filter_metrics["uu_mia"])

    print(
        f"|Δ| Test All               : "
        f"Acc={abs_delta(no_filter_metrics['test']['acc'], filter_metrics['test']['acc']):.2f}, "
        f"Loss={abs_delta(no_filter_metrics['test']['loss'], filter_metrics['test']['loss']):.4f}"
    )
    print(
        f"|Δ| Remain Excluding Forget: "
        f"Acc={abs_delta(no_filter_metrics['remain']['acc'], filter_metrics['remain']['acc']):.2f}, "
        f"Loss={abs_delta(no_filter_metrics['remain']['loss'], filter_metrics['remain']['loss']):.4f}"
    )
    print(
        f"|Δ| Forget                 : "
        f"Acc={abs_delta(no_filter_metrics['forget']['acc'], filter_metrics['forget']['acc']):.2f}, "
        f"Loss={abs_delta(no_filter_metrics['forget']['loss'], filter_metrics['forget']['loss']):.4f}"
    )
    print(
        f"|Δ| Forget-Set MIA         : "
        f"Acc={abs_delta(no_filter_metrics['forget_mia']['acc'], filter_metrics['forget_mia']['acc']):.2f}, "
        f"F1={abs_delta(no_filter_metrics['forget_mia']['f1'], filter_metrics['forget_mia']['f1']):.2f}"
    )
    print(
        f"|Δ| UU Set                 : "
        f"Acc={abs_delta(no_filter_metrics['uu']['acc'], filter_metrics['uu']['acc']):.2f}, "
        f"Loss={abs_delta(no_filter_metrics['uu']['loss'], filter_metrics['uu']['loss']):.4f}"
    )
    print(
        f"|Δ| UU-Set MIA             : "
        f"Acc={abs_delta(no_filter_metrics['uu_mia']['acc'], filter_metrics['uu_mia']['acc']):.2f}, "
        f"F1={abs_delta(no_filter_metrics['uu_mia']['f1'], filter_metrics['uu_mia']['f1']):.2f}"
    )


# ============================================================
# 12. P2: delete all forget samples and retrain
# ============================================================
base_retrain_state = build_fixed_retrain_init_state(RETRAIN_INIT_SEED)
print(f"\n[Retrain Init] fixed retrain init state created with seed={RETRAIN_INIT_SEED}")

forget_set = set(forget_local_indices)
client_indices_phase2 = make_phase_client_indices_from_removed_set(removed_set=forget_set, phase_tag="Retrain-Phase2")
p2_public_schedule = build_public_round_schedule(
    client_indices_ref=client_indices_phase2,
    total_rounds=rounds,
    samples_per_round=SAMPLES_PER_ROUND,
    base_seed=RETRAIN_SCHEDULE_SEED,
)
global_model_p2, p2_retrain_time = federated_train_with_public_schedule(
    phase_tag="Phase2",
    init_state=base_retrain_state,
    client_indices_ref=client_indices_phase2,
    schedule=p2_public_schedule,
    lr=1e-3,
)
print(f"[Phase2] retrain time: {p2_retrain_time:.6f} s")

phase2_base_metrics = evaluate_phase_base_sets(
    phase_tag="Phase2",
    target_model=global_model_p2,
    forget_mia_label=0,
)
phase2_uu_metrics = evaluate_method_specific_uu_set(
    phase_tag="Phase2",
    method_name="SimHash + Confidence",
    target_model=global_model_p2,
    uu_indices=uu_local_indices,
    uu_membership_label=1,
)

phase2_metrics = {
    "test": dict(phase2_base_metrics["test"]),
    "remain": dict(phase2_base_metrics["remain"]),
    "forget": dict(phase2_base_metrics["forget"]),
    "forget_mia": dict(phase2_base_metrics["forget_mia"]),
    "uu": dict(phase2_uu_metrics["model"]),
    "uu_mia": dict(phase2_uu_metrics["mia"]),
}


# ============================================================
# 13. P3: keep UU, delete NU, then retrain
# ============================================================
client_indices_phase3 = make_phase_client_indices_from_removed_set(removed_set=set(nu_local_indices), phase_tag="Retrain-Phase3")
p3_public_schedule = build_public_round_schedule(
    client_indices_ref=client_indices_phase3,
    total_rounds=rounds,
    samples_per_round=SAMPLES_PER_ROUND,
    base_seed=RETRAIN_SCHEDULE_SEED,
)
global_model_p3, p3_retrain_time = federated_train_with_public_schedule(
    phase_tag="Phase3",
    init_state=base_retrain_state,
    client_indices_ref=client_indices_phase3,
    schedule=p3_public_schedule,
    lr=1e-3,
)
print(f"[Phase3] retrain time: {p3_retrain_time:.6f} s")

forget_membership_map = {idx: 1 for idx in uu_local_indices}
forget_membership_map.update({idx: 0 for idx in nu_local_indices})

phase3_base_metrics = evaluate_phase_base_sets(
    phase_tag="Phase3",
    target_model=global_model_p3,
    forget_mia_label_map=forget_membership_map,
)
phase3_uu_metrics = evaluate_method_specific_uu_set(
    phase_tag="Phase3",
    method_name="SimHash + Confidence",
    target_model=global_model_p3,
    uu_indices=uu_local_indices,
    uu_membership_label=1,
)

phase3_metrics = {
    "test": dict(phase3_base_metrics["test"]),
    "remain": dict(phase3_base_metrics["remain"]),
    "forget": dict(phase3_base_metrics["forget"]),
    "forget_mia": dict(phase3_base_metrics["forget_mia"]),
    "uu": dict(phase3_uu_metrics["model"]),
    "uu_mia": dict(phase3_uu_metrics["mia"]),
}


# ============================================================
# 14. final summary
# ============================================================
print("\n" + "=" * 120)
print("[FINAL SUMMARY]")
print("=" * 120)
print(f"[Config] rounds={rounds}, local_batches_per_round={LOCAL_BATCHES_PER_ROUND}, batch_size={batch_size}, samples_per_round={SAMPLES_PER_ROUND}")
print(f"[Forget Size] {len(forget_local_indices)}")
print(f"[UU Size] {len(uu_local_indices)} | [NU Size] {len(nu_local_indices)} | [Filtering Ratio] {filter_ratio:.6f}")
print(f"[Forget First-Round Sum] {forget_pos_sum}")
print(f"[UU First-Round Sum] {uu_pos_sum}")
print(f"[Phase2 Retrain Time] {p2_retrain_time:.6f} s")
print(f"[Phase3 Retrain Time] {p3_retrain_time:.6f} s")
print(f"[Retrain Time Saved by Filtering] {p2_retrain_time - p3_retrain_time:.6f} s")

print_comparison_block(
    method_name="SimHash + Confidence",
    no_filter_tag="Phase2(no filtering)",
    no_filter_metrics=phase2_metrics,
    filter_tag="Phase3(filter keep-UU)",
    filter_metrics=phase3_metrics,
)
