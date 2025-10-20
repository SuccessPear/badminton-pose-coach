import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def pack_collate1(batch: list[tuple[torch.tensor, int]]) -> dict[str, any]:
    """
    Collate function sử dụng torch.nn.utils.rnn.pack_padded_sequence.
    - pose: (T,K,3) với T có thể khác nhau
    - Trả về PackedSequence để dùng cho RNN.

    Returns:
      packed: PackedSequence chứa (T, K*3)
      lengths: chiều dài thực tế từng sample
      labels: (B,)
    """
    poses, labels = zip(*batch)
    lengths = torch.tensor([p.shape[0] for p in poses], dtype=torch.long)
    K = poses[0].shape[1]

    # Flatten (T,K,3) -> (T, K*3)
    flat_poses = [p.reshape(p.shape[0], K*3) for p in poses]
    padded = torch.nn.utils.rnn.pad_sequence(flat_poses, batch_first=True)

    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return {"packed": packed, "lengths": lengths, "labels": labels_t}


def pack_collate(batch: list[tuple[torch.Tensor, int]]) -> dict[str, any]:
    """
    Collate cho 2 định dạng pose:
      - RNN:    (T, V, 3)  -> PackedSequence (flatten K*3)
      - ST-GCN: (C, T, V, M) -> pad theo T -> (B, C, T_max, V, M) + mask

    Trả về:
      Nếu RNN:
        {"packed": PackedSequence, "lengths": (B,), "labels": (B,), "mode": "rnn"}
      Nếu ST-GCN:
        {"data": (B,C,T_max,V,M), "lengths": (B,), "labels": (B,), "mask": (B,T_max), "mode": "stgcn"}
    """
    xs, ys = zip(*batch)
    first = xs[0]

    # ---------- Case A: RNN pose (T,V,3) ----------
    if first.ndim == 3 and first.shape[-1] in (2, 3):
        lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
        V, Ccoord = first.shape[1], first.shape[2]

        # (T,V,C) -> (T,V*C)
        flat = [x.reshape(x.shape[0], V * Ccoord) for x in xs]
        padded = pad_sequence(flat, batch_first=True)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        labels = torch.tensor(ys, dtype=torch.long)

        return {
            "packed": packed,
            "lengths": lengths,
            "labels": labels,
            "mode": "rnn",
        }

    # ---------- Case B: ST-GCN pose (C,T,V,M) ----------
    if first.ndim == 4:
        lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)
        T_max = int(lengths.max())
        C, _, V, M = first.shape
        B = len(xs)

        data = torch.zeros((B, C, T_max, V, M), dtype=first.dtype)
        mask = torch.zeros((B, T_max), dtype=torch.bool)

        for i, x in enumerate(xs):
            T = x.shape[1]
            data[i, :, :T, :, :] = x
            mask[i, :T] = True

        labels = torch.tensor(ys, dtype=torch.long)
        return {
            "packed": data,
            "lengths": lengths,
            "labels": labels,
            "mask": mask,
            "mode": "stgcn",
        }

    raise ValueError(f"Unsupported sample shape {tuple(first.shape)}. Expect (T,V,3) or (C,T,V,M).")
