"""Shortcut kept for the earlier instructions: India only.

    python prep_rdd_india.py --src downloads/rdd2022
    ==  python prep_rdd2022.py --src downloads/rdd2022 --countries India
Output: datasets/raw/rdd_india/{images,labels}
"""
import argparse
import sys

import prep_rdd2022

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="downloads/rdd2022")
    ap.add_argument("--out", default="datasets/raw/rdd_india", help="kept for compatibility")
    a = ap.parse_args()
    raw = str(__import__("pathlib").Path(a.out).parent)
    sys.argv = [sys.argv[0], "--src", a.src, "--raw", raw, "--countries", "India"]
    prep_rdd2022.main()
