"""Shortcut kept for the earlier instructions: India only.

    python get_rdd_india.py      ==   python get_rdd2022.py --countries India
"""
import sys

import get_rdd2022

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--countries", "India", *sys.argv[1:]]
    get_rdd2022.main()
