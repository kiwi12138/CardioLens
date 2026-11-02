"""
Multi-sequence cardiac MRI dataset for MIL training.

Each example is a (patient, sequence) bag of 2D slices, optionally filtered to
the views required by that sequence's tasks. Labels are sequence-level and may
be partially missing; a per-task validity mask is provided alongside.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


SEQUENCES = ("cine", "LGE", "T2", "perfusion")


def normalize_view_name(view: str, sequence: str) -> str:
    """Standardize view names by prefixing them with the sequence (lowercased).

    Examples:
        ('4ch', 'cine') -> 'cine_4ch'
        ('sax', 'LGE')  -> 'lge_sax'
    """
    view = view.strip().lower()
    sequence = sequence.strip().lower()
    if view.startswith(sequence + "_"):
        return view
    return f"{sequence}_{view}"


class MedicalImageAugmentation:
    """Conservative augmentation suitable for cardiac MRI slices.

    Applied per-slice on tensors of shape [1, H, W] in the [0, 1] range.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()

        # Horizontal flip.
        if random.random() < self.p:
            image = torch.flip(image, dims=[2])

        # Small-angle rotation.
        if random.random() < self.p:
            angle = random.uniform(-10, 10)
            image = T.functional.rotate(image, angle)

        # Brightness perturbation.
        if random.random() < self.p:
            image = image * random.uniform(0.9, 1.1)

        # Contrast perturbation.
        if random.random() < self.p:
            mean_val = image.mean()
            image = (image - mean_val) * random.uniform(0.9, 1.1) + mean_val

        # Mild Gaussian noise.
        if random.random() < self.p:
            image = image + torch.randn_like(image) * random.uniform(0.01, 0.03)

        return torch.clamp(image, 0.0, 1.0)


class MILCardiacDataset(Dataset):
    """
    Multi-sequence cardiac MRI dataset.

    The cache directory is expected to follow this layout:
        cache_dir/
            index.json                  # patient -> {"sequences": [...]}
            <pid>/
                metadata.json           # per-sequence list of {"view": ...}
                <sequence>.npy          # array of shape [N, H, W]

    The label JSON is expected to provide:
        - "task_configs": {task_name -> {"sequence", "type", "num_classes",
                                         "choices", "views"?}}
        - "labels":       {patient_id -> {task_name -> [label_text, ...]}}
        - optionally "patient_id_list" for class-balanced training.
    """

    def __init__(
        self,
        cache_dir: str,
        label_json: str,
        split: str = "train",
        val_ratio: float = 0.2,
        random_seed: int = 42,
        use_augmentation: bool = False,
        augmentation_prob: float = 0.5,
        max_slices: int = 512,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

        self.cache_dir = Path(cache_dir)
        self.split = split
        self.max_slices = max_slices

        self.use_augmentation = use_augmentation and (split == "train")
        self.augmentation = (
            MedicalImageAugmentation(p=augmentation_prob)
            if self.use_augmentation
            else None
        )

        # ---- Load index. -------------------------------------------------- #
        with open(self.cache_dir / "index.json", "r") as f:
            raw_index = json.load(f)

        # Normalize patient IDs to integer-string form so labels and index agree.
        self.index: Dict[str, Dict] = {}
        for pid, info in raw_index.items():
            normalized = str(int(pid))
            info = dict(info)
            info["original_pid"] = pid
            self.index[normalized] = info

        # ---- Load labels. ------------------------------------------------- #
        with open(label_json, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        self.all_task_configs: Dict[str, Dict] = label_data["task_configs"]
        self.all_labels: Dict[str, Dict] = label_data["labels"]
        self.use_balanced = "patient_id_list" in label_data

        # Group tasks by sequence.
        self.sequence_tasks: Dict[str, Dict[str, Dict]] = {s: {} for s in SEQUENCES}
        for task_name, cfg in self.all_task_configs.items():
            self.sequence_tasks[cfg["sequence"]][task_name] = cfg

        # ---- Build patient list. ----------------------------------------- #
        if self.use_balanced and split == "train":
            candidate_ids = [
                pid for pid in label_data["patient_id_list"]
                if pid in self.index and len(self.index[pid]["sequences"]) > 0
            ]
            print("Using balanced patient list (patient_id_list) for training.")
        else:
            candidate_ids = [
                pid for pid, info in self.index.items()
                if len(info["sequences"]) > 0 and pid in self.all_labels
            ]
            if self.use_balanced:
                print("Using original patient list (labels) for validation.")

        rng = np.random.default_rng(random_seed)
        ids_arr = np.array(candidate_ids)
        if not (self.use_balanced and split == "train"):
            rng.shuffle(ids_arr)
        split_idx = int(len(ids_arr) * (1 - val_ratio))

        self.patient_ids = (
            ids_arr[:split_idx].tolist() if split == "train"
            else ids_arr[split_idx:].tolist()
        )
        self.label_map = self._build_label_map()

        unique_patients = len(set(self.patient_ids))
        print(
            f"[{split}] Loaded {len(self.patient_ids)} samples "
            f"(unique patients: {unique_patients})"
        )
        print(
            f"  Total candidates: {len(ids_arr)} | "
            f"train: {split_idx} | val: {len(ids_arr) - split_idx}"
        )
        for s in SEQUENCES:
            print(f"  {s}: {len(self.sequence_tasks[s])} tasks")

    def _build_label_map(self) -> Dict[str, Dict[str, Dict]]:
        label_map: Dict[str, Dict[str, Dict]] = {}
        for pid in set(self.patient_ids):
            raw_labels = self.all_labels.get(pid, {})
            patient_labels: Dict[str, Dict] = {}
            for task_name, task_cfg in self.all_task_configs.items():
                raw = raw_labels.get(task_name)
                if not raw:
                    patient_labels[task_name] = {"label": None, "mask": 0}
                    continue

                if task_cfg["type"] == "single":
                    label_text = raw[0]
                    if label_text in task_cfg["choices"]:
                        idx = task_cfg["choices"].index(label_text)
                        patient_labels[task_name] = {"label": idx, "mask": 1}
                    else:
                        patient_labels[task_name] = {"label": None, "mask": 0}
                elif task_cfg["type"] == "multi":
                    multi_hot = np.zeros(task_cfg["num_classes"], dtype=np.float32)
                    for label_text in raw:
                        if label_text in task_cfg["choices"]:
                            multi_hot[task_cfg["choices"].index(label_text)] = 1.0
                    if multi_hot.sum() > 0:
                        patient_labels[task_name] = {"label": multi_hot, "mask": 1}
                    else:
                        patient_labels[task_name] = {"label": None, "mask": 0}
                else:
                    raise ValueError(f"Unknown task type: {task_cfg['type']}")
            label_map[pid] = patient_labels
        return label_map

    def _required_views(self, sequence: str) -> Optional[set]:
        """Collect normalized views needed by all tasks of one sequence.

        Returns None if any task asks for "all slices" (empty views list).
        """
        required: set = set()
        for task_name in self.sequence_tasks[sequence].keys():
            views = self.all_task_configs[task_name].get("views", [])
            if not views:
                return None
            for v in views:
                view_part = v.split("_", 1)[1] if "_" in v else v
                required.add(normalize_view_name(view_part, sequence))
        return required

    def _augment_slices(self, slices: torch.Tensor) -> torch.Tensor:
        if self.augmentation is None:
            return slices
        return torch.stack(
            [self.augmentation(slices[i]) for i in range(slices.shape[0])], dim=0
        )

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.patient_ids[idx]
        original_pid = self.index[pid]["original_pid"]
        available_sequences = self.index[pid]["sequences"]

        with open(self.cache_dir / original_pid / "metadata.json", "r") as f:
            all_metadata = json.load(f)

        all_images: Dict[str, torch.Tensor] = {}
        for seq in available_sequences:
            npy_path = self.cache_dir / original_pid / f"{seq}.npy"
            if not npy_path.exists():
                continue

            slices = np.load(npy_path)  # [N, H, W]
            slice_metadata = all_metadata.get(seq, [])
            required_views = self._required_views(seq)

            if required_views is None:
                # Tasks request all slices (e.g., perfusion dynamic series).
                selected = slices
            else:
                idxs = [
                    i for i, meta in enumerate(slice_metadata)
                    if normalize_view_name(meta.get("view", ""), seq) in required_views
                ]
                selected = slices[idxs] if idxs else slices  # fallback to all slices

            if len(selected) > self.max_slices:
                pick = np.linspace(0, len(selected) - 1, self.max_slices, dtype=int)
                selected = selected[pick]

            tensor = torch.from_numpy(selected).float().unsqueeze(1)  # [N, 1, H, W]
            if self.use_augmentation:
                tensor = self._augment_slices(tensor)
            all_images[seq] = tensor

        # Assemble per-task labels and masks.
        all_labels: Dict[str, Dict[str, torch.Tensor]] = {}
        all_masks: Dict[str, Dict[str, int]] = {}
        patient_labels = self.label_map.get(pid, {})

        for seq in SEQUENCES:
            seq_labels: Dict[str, torch.Tensor] = {}
            seq_masks: Dict[str, int] = {}
            for task_name, task_cfg in self.sequence_tasks[seq].items():
                tl = patient_labels.get(task_name, {"label": None, "mask": 0})
                if tl["mask"] == 1:
                    if task_cfg["type"] == "single":
                        seq_labels[task_name] = torch.tensor(tl["label"], dtype=torch.long)
                    else:
                        seq_labels[task_name] = torch.from_numpy(tl["label"]).float()
                else:
                    if task_cfg["type"] == "single":
                        seq_labels[task_name] = torch.tensor(-1, dtype=torch.long)
                    else:
                        seq_labels[task_name] = torch.zeros(
                            task_cfg["num_classes"], dtype=torch.float32
                        )
                seq_masks[task_name] = tl["mask"]
            all_labels[seq] = seq_labels
            all_masks[seq] = seq_masks

        return {
            "patient_id": pid,
            "images": all_images,
            "labels": all_labels,
            "masks": all_masks,
            "available_sequences": available_sequences,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Single-bag batches (variable bag size). Larger batches are not supported."""
    if len(batch) == 1:
        return batch[0]
    raise NotImplementedError("Batch size > 1 is not supported.")
