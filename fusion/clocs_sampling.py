"""Training-only candidate selection; never sample evaluation candidates."""
import hashlib
import numpy as np

PILOT_IDS = frozenset('000000 000149 001970 000972 001020 004842 002485 002610 006788 002106 006830 003079 006707 004838 002496 003013'.split())


def split_frames(ids, seed=20260928, dev_count=32):
    ids = sorted(ids)
    if len(ids) != 128 or len(set(ids)) != 128:
        raise ValueError('Require exactly 128 unique frames')
    if not PILOT_IDS.issubset(ids):
        raise ValueError('Expected earlier pilot frames are missing')
    eligible = set(ids) - PILOT_IDS
    ranked = sorted(eligible, key=lambda fid: hashlib.sha256(f'{seed}:{fid}'.encode()).hexdigest())
    if not 0 < dev_count < len(ranked):
        raise ValueError('Invalid development size')
    dev = sorted(ranked[:dev_count])
    return sorted(set(ids) - set(dev)), dev


def select_candidates(labels, scores, ratio, seed, epoch, frame_id, background_cap=256):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError('Candidate labels and scores must be matching vectors')
    if not np.isin(labels, [-1, 0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid labels or scores')
    if ratio not in (0, 4, 8, 16) or background_cap < 1:
        raise ValueError('Ratio must be 0, 4, 8 or 16; background cap positive')
    if ratio == 0:
        return np.flatnonzero(labels >= 0)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    count = min(len(neg), ratio * len(pos) if len(pos) else background_cap)
    ranked = neg[np.argsort(-scores[neg], kind='stable')]
    hard_count = count // 2
    hard = ranked[:hard_count]
    digest = hashlib.sha256(f'{seed}:{epoch}:{frame_id}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    random_part = rng.choice(ranked[hard_count:], count - hard_count, replace=False)
    return np.sort(np.concatenate((pos, hard, random_part)))
