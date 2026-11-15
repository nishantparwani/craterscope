"""Deduplicate, then make ONE grouped 70/15/15 split and freeze it (roadmap rule 1).

  1. pHash every image; pairs with Hamming <= 8 AND SSIM > 0.9 are duplicates.
     Each duplicate cluster keeps one image (the one with most boxes).
  2. Near-similar survivors (Hamming <= --group-hamming) are grouped so a scene
     never lands in two splits (group size capped to avoid chaining).
  3. --neg-holdout pothole-free images go to 'neg_test' (FPPI set, rule 3).
  4. Remaining groups are split 70/15/15, stratified pothole / pothole-free.
  5. Writes splits.json (commit it, never change it) and the Ultralytics layout:
       datasets/craterscope/images/{train,val,test,neg_test}, labels/..., data.yaml

    python prep_split.py --raw datasets/raw                      # create (Colab)
    python prep_split.py --raw datasets/raw --eval-sources rdd_india  # all countries train, India val/test
    python prep_split.py --raw datasets/raw --from-splits splits.json   # rebuild same split (Windows)
    python prep_split.py --raw datasets/raw --from-splits splits.json --build val test neg_test
"""
import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from classical import load_bgr
from config import DATASET, ROOT
from data_utils import hamming_matrix_rows, list_images, phash64, save_json, write_data_yaml


class UnionFind:
    def __init__(self, n, cap=None):
        self.p, self.size, self.cap = list(range(n)), [1] * n, cap

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb or (self.cap and self.size[ra] + self.size[rb] > self.cap):
            return
        self.p[rb] = ra
        self.size[ra] += self.size[rb]


def collect(raw: Path):
    """All raw images as dicts {key, img, lbl, n_boxes}; key = '<source>/<file>'."""
    items = []
    for src in sorted(p for p in raw.iterdir() if (p / "images").is_dir()):
        for img in list_images(src / "images"):
            lbl = src / "labels" / f"{img.stem}.txt"
            n = sum(1 for ln in lbl.read_text().splitlines() if ln.strip()) if lbl.exists() else 0
            items.append({"key": f"{src.name}/{img.name}", "img": img, "lbl": lbl, "n_boxes": n})
    return items


def thumbs_and_hashes(items):
    thumbs, hashes = [], []
    for i, it in enumerate(items):
        g = cv2.cvtColor(load_bgr(it["img"]), cv2.COLOR_BGR2GRAY)
        t = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
        thumbs.append(t)
        hashes.append(phash64(t))
        if i % 1000 == 0:
            print(f"  hashed {i}/{len(items)}")
    return thumbs, np.array(hashes, dtype=np.uint64)


def find_pairs(hashes, max_dist):
    pairs = []
    for s in range(0, len(hashes), 512):
        d = hamming_matrix_rows(hashes, s, min(s + 512, len(hashes)))
        a, b = np.nonzero(d <= max_dist)
        a = a + s
        keep = a < b
        pairs.extend(zip(a[keep].tolist(), b[keep].tolist(), d[a[keep] - s, b[keep]].tolist()))
    return pairs


def assign(groups, ratios, rng):
    """Greedy grouped split: shuffle groups, fill the split furthest below target."""
    order = rng.permutation(len(groups))
    total = sum(len(g) for g in groups)
    counts = {k: 0 for k in ratios}
    out = {k: [] for k in ratios}
    for gi in order:
        need = {k: ratios[k] * total - counts[k] for k in ratios}
        k = max(need, key=need.get)
        out[k].extend(groups[gi])
        counts[k] += len(groups[gi])
    return out


def materialise(items_by_key, split_map, out_root: Path):
    if out_root.exists():
        shutil.rmtree(out_root)
    for split, keys in split_map.items():
        for key in keys:
            it = items_by_key[key]
            name = key.replace("/", "__")
            dst = out_root / "images" / split / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(it["img"], dst)
            lbl_dst = out_root / "labels" / split / (Path(name).stem + ".txt")
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            if it["lbl"].exists():
                shutil.copy2(it["lbl"], lbl_dst)
            else:
                lbl_dst.write_text("")
    write_data_yaml(out_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/raw")
    ap.add_argument("--out", default=str(DATASET))
    ap.add_argument("--splits", default=str(ROOT / "splits.json"))
    ap.add_argument("--from-splits", default=None, help="rebuild an existing frozen split")
    ap.add_argument("--dup-hamming", type=int, default=8)
    ap.add_argument("--dup-ssim", type=float, default=0.9)
    ap.add_argument("--group-hamming", type=int, default=12)
    ap.add_argument("--max-group", type=int, default=50)
    ap.add_argument("--neg-holdout", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-sources", nargs="+", default=None,
                    help="raw sources allowed in val/test/neg_test, e.g. rdd_india (default: all)")
    ap.add_argument("--extra-neg-ratio", type=float, default=1.0,
                    help="train-only sources: keep at most ratio x positives pothole-free images")
    ap.add_argument("--build", nargs="+", default=["train", "val", "test", "neg_test"],
                    help="with --from-splits: which splits to rebuild (laptop: val test neg_test)")
    args = ap.parse_args()

    items = collect(Path(args.raw))
    by_key = {it["key"]: it for it in items}
    print(f"{len(items)} raw images")

    if args.from_splits:
        frozen = json.loads(Path(args.from_splits).read_text())
        split_map = {k: frozen[k] for k in args.build}
        missing = [k for keys in split_map.values() for k in keys if k not in by_key]
        if missing:
            raise SystemExit(f"{len(missing)} images in splits.json not found under {args.raw}, e.g. {missing[:3]}")
        materialise(by_key, split_map, Path(args.out))
        print({k: len(v) for k, v in split_map.items()}, "->", args.out)
        return

    t0 = time.time()
    thumbs, hashes = thumbs_and_hashes(items)
    pairs = find_pairs(hashes, max(args.dup_hamming, args.group_hamming))

    # 1) duplicates: Hamming <= 8 confirmed by SSIM > 0.9
    dup = UnionFind(len(items))
    n_dup_pairs = 0
    for a, b, d in pairs:
        if d <= args.dup_hamming and structural_similarity(thumbs[a], thumbs[b]) > args.dup_ssim:
            dup.union(a, b)
            n_dup_pairs += 1
    clusters = defaultdict(list)
    for i in range(len(items)):
        clusters[dup.find(i)].append(i)
    keep, removed = [], []
    for members in clusters.values():
        best = max(members, key=lambda i: items[i]["n_boxes"])
        keep.append(best)
        removed.extend(items[i]["key"] for i in members if i != best)
    keep_set = set(keep)
    print(f"duplicate pairs: {n_dup_pairs}, removed {len(removed)} images")

    # 2) grouping of near-similar survivors (capped to stop chaining)
    grp = UnionFind(len(items), cap=args.max_group)
    for a, b, d in pairs:
        if a in keep_set and b in keep_set and d <= args.group_hamming:
            grp.union(a, b)
    groups = defaultdict(list)
    for i in keep:
        groups[grp.find(i)].append(i)
    groups = list(groups.values())
    # 3) FPPI hold-out + 4) stratified grouped split, on the EVAL sources only.
    #    Train-only sources (other countries) go to train; their negatives are capped.
    source = [it["key"].split("/")[0] for it in items]
    eval_src = set(args.eval_sources) if args.eval_sources else set(source)
    rng = np.random.default_rng(args.seed)
    eval_groups = [g for g in groups if any(source[i] in eval_src for i in g)]
    extra_groups = [g for g in groups if not any(source[i] in eval_src for i in g)]

    pos_groups = [g for g in eval_groups if any(items[i]["n_boxes"] for i in g)]
    neg_groups = [g for g in eval_groups if not any(items[i]["n_boxes"] for i in g)]
    neg_groups = [neg_groups[i] for i in rng.permutation(len(neg_groups))]
    neg_test, rest_neg = [], []
    for g in neg_groups:
        (neg_test if len(neg_test) + len(g) <= args.neg_holdout else rest_neg).append(g)
    neg_test = [i for g in neg_test for i in g]
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_idx = {k: [] for k in ratios}
    for part in (pos_groups, rest_neg):
        for k, v in assign(part, ratios, rng).items():
            split_idx[k].extend(v)

    # other-source images that were grouped with an eval image: keep them out of val/test
    dropped = [i for k in ("val", "test") for i in split_idx[k] if source[i] not in eval_src]
    dropped += [i for i in neg_test if source[i] not in eval_src]
    for k in ("val", "test"):
        split_idx[k] = [i for i in split_idx[k] if source[i] in eval_src]
    neg_test = [i for i in neg_test if source[i] in eval_src]

    extra = [i for g in extra_groups for i in g]
    for src_name in sorted({source[i] for i in extra}):
        pos = [i for i in extra if source[i] == src_name and items[i]["n_boxes"]]
        neg = [i for i in extra if source[i] == src_name and not items[i]["n_boxes"]]
        cap = int(args.extra_neg_ratio * len(pos))
        neg = [neg[j] for j in rng.permutation(len(neg))[:cap]]
        split_idx["train"].extend(pos + neg)
        print(f"  train-only source {src_name}: {len(pos)} pothole images + {len(neg)} negatives")

    split_map = {k: sorted(items[i]["key"] for i in v) for k, v in split_idx.items()}
    split_map["neg_test"] = sorted(items[i]["key"] for i in neg_test)
    print(f"eval sources: {sorted(eval_src)}; cross-source group members kept out of val/test: {len(dropped)}")

    frozen = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": args.seed,
              "rules": {"dup_hamming": args.dup_hamming, "dup_ssim": args.dup_ssim,
                        "group_hamming": args.group_hamming, "max_group": args.max_group,
                        "eval_sources": sorted(eval_src), "extra_neg_ratio": args.extra_neg_ratio},
              "counts": {k: len(v) for k, v in split_map.items()},
              "positives": {k: sum(by_key[x]["n_boxes"] > 0 for x in v) for k, v in split_map.items()},
              "removed_duplicates": sorted(removed), **split_map}
    if Path(args.splits).exists():
        raise SystemExit(f"{args.splits} exists - the split is frozen. Delete it only if you mean to start over.")
    save_json(Path(args.splits), frozen)
    materialise(by_key, split_map, Path(args.out))
    print(frozen["counts"], "positives:", frozen["positives"], f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
