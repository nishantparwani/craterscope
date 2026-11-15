"""Download RDD2022 from the official Figshare record and extract chosen countries.

The old per-country S3 links (bigdatacup.s3...RDD2022_<Country>.zip) return
403 Access Denied. The official copy is Figshare article 21431547: one
~12.4 GB archive with one nested zip per country (RDD2022/India.zip, ...).

    python get_rdd2022.py --countries India                 # India only (~0.5 GB kept)
    python get_rdd2022.py --countries all                   # every country
    python get_rdd2022.py --countries India Japan Czech United_States China_MotorBike
    python get_rdd2022.py --zip D:\\Downloads\\big.zip --countries all   # browser-downloaded archive
    python get_rdd2022.py --list                            # just show what is inside

Result: downloads/rdd2022/<Country>/train/{images,annotations/xmls}
"""
import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ARTICLE_API = "https://api.figshare.com/v2/articles/21431547"
UA = {"User-Agent": "craterscope-dataset-fetch/1.0"}


def figshare_file():
    with urllib.request.urlopen(urllib.request.Request(ARTICLE_API, headers=UA), timeout=60) as r:
        meta = json.load(r)
    files = sorted(meta["files"], key=lambda f: -f["size"])
    for f in files:
        print(f"  figshare file: {f['name']}  {f['size'] / 1e9:.2f} GB  {f['download_url']}")
    return files[0]


def _fetch(url: str, dst: Path, size: int) -> None:
    """One HTTP attempt, appending from the current file size (HTTP Range)."""
    done = dst.stat().st_size if dst.exists() else 0
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={done}-"} if done else UA)
    with urllib.request.urlopen(req, timeout=120) as r, dst.open("ab" if done else "wb") as f:
        if done and r.status != 206:              # server ignored Range -> start over
            f.seek(0)
            f.truncate()
            done = 0
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if size:
                sys.stdout.write(f"\r  {done / 1e9:6.2f} / {size / 1e9:.2f} GB ({100 * done / size:5.1f}%)")
                sys.stdout.flush()


def download(url: str, dst: Path, size: int, retries: int = 50) -> None:
    """Resumable download that retries dropped connections until the file is complete.
    Re-running the script also continues from the partial file."""
    for attempt in range(1, retries + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if size and have == size:
            print(f"\n  complete: {dst} ({have / 1e9:.2f} GB)")
            return
        if size and have > size:                   # corrupt partial file
            dst.unlink()
            have = 0
        if have:
            print(f"\n  resuming at {have / 1e9:.2f} GB (attempt {attempt})")
        try:
            _fetch(url, dst, size)
        except Exception as e:                      # dropped / reset / timed-out connection
            print(f"\n  connection problem: {e!r}; retrying in 10 s")
            time.sleep(10)
            continue
        if not size:
            return
    have = dst.stat().st_size if dst.exists() else 0
    raise SystemExit(f"Download incomplete ({have / 1e9:.2f} of {size / 1e9:.2f} GB). "
                     f"Run the same command again - it resumes from where it stopped.")


def country_zips(z: zipfile.ZipFile) -> dict:
    """{'India': 'RDD2022/India.zip', ...} for every nested country archive."""
    return {PurePosixPath(n).stem: n for n in z.namelist()
            if n.lower().endswith(".zip") and not n.endswith("/")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=["India"], help="names or 'all'")
    ap.add_argument("--zip", default=None, help="already-downloaded Figshare archive")
    ap.add_argument("--downloads", default="downloads")
    ap.add_argument("--out", default=None, help="default: <downloads>/rdd2022")
    ap.add_argument("--keep-big", action="store_true", help="keep the 12 GB archive afterwards")
    ap.add_argument("--keep-country-zips", action="store_true")
    ap.add_argument("--list", action="store_true", help="only list the countries inside")
    args = ap.parse_args()

    dl = Path(args.downloads)
    dl.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else dl / "rdd2022"
    big = Path(args.zip) if args.zip else dl / "RDD2022_all.zip"
    if not args.zip:
        f = figshare_file()
        print(f"downloading {f['name']} -> {big}")
        download(f["download_url"], big, f["size"])
    if not zipfile.is_zipfile(big):
        raise SystemExit(f"{big} is not a complete zip ({big.stat().st_size / 1e9:.2f} GB). "
                         "If you downloaded it in a browser, let the download finish; otherwise run "
                         "this script again without --zip and it will resume.")

    with zipfile.ZipFile(big) as z:
        inner = country_zips(z)
        print("countries in archive:", ", ".join(sorted(inner)) or "(none - not a nested archive)")
        if args.list:
            return
        wanted = sorted(inner) if args.countries == ["all"] else args.countries
        missing = [c for c in wanted if c not in inner]
        if missing:
            raise SystemExit(f"not in archive: {missing}; available: {sorted(inner)}")
        for c in wanted:
            czip = dl / f"RDD2022_{c}.zip"
            print(f"extracting {inner[c]} -> {czip}")
            with z.open(inner[c]) as src, czip.open("wb") as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            with zipfile.ZipFile(czip) as cz:
                cz.extractall(out)
            n_xml = sum(1 for _ in (out / c).rglob("*.xml")) if (out / c).exists() else -1
            print(f"  {c}: {n_xml} XML annotations under {out / c}")
            if not args.keep_country_zips:
                czip.unlink()
    if not args.keep_big and not args.zip:
        big.unlink()
        print(f"deleted {big} (use --keep-big to keep it)")


if __name__ == "__main__":
    main()
