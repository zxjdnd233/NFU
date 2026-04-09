# NFU Experimental README

This repository contains three standalone experimental scripts for **NFU (Neighbor-Assisted Federated Unlearning)**.

NFU is designed for efficient federated unlearning by leveraging the observation that some forget-requested samples can be approximately replaced by their neighbors in the feature space. Based on this idea, NFU consists of two main modules:

- **NFU-F (Filtering Module):** identifies **UU samples** (*unnecessary-to-unlearn samples*) in the forget set. These samples are considered highly replaceable by their neighbors and therefore may be kept without significantly affecting unlearning quality.
- **NFU-U (Unlearning Module):** performs efficient unlearning for the remaining **NU samples** (*necessary-to-unlearn samples*) using a neighbor-assisted projected gradient update.

The three scripts correspond to the following experiments:

- `NFU_F_exp1.py`: filtering experiments including comparison with multiple filtering baselines.
```bash
python NFU_F_exp1.py
```
- `NFU_F_exp2.py`: filtering + FATS experiment, mainly for evaluating the effect of **SimHash + Confidence** on retraining cost and performance.
```bash
python NFU_F_exp2.py
```
- `NFU_U_exp3.py`: unlearning experiment, comparing **NFU-U** with several fast unlearning baselines.
```bash
python NFU_U_exp3.py
```

---

##  Environment

Recommended environment:

- Python 3.7+
- PyTorch 1.4.0
- torchvision 0.5.0
- numpy 1.19.2
- scikit-learn 1.0.2





