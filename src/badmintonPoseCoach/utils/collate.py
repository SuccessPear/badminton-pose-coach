import torch
from torch.nn.utils.rnn import pack_padded_sequence
def pack_collate(batch: list[tuple[torch.tensor, int]]) -> dict[str, any]:
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