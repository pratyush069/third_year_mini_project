"""Download the EXPY-TKY dataset (Tokyo expressway speeds, Oct-Dec 2021).

Source: MegaCRN repository (Jiang et al., AAAI 2023), MIT license.
Usage:  python -m src.data.download
"""
import os
import shutil
import ssl
import urllib.request

import certifi
import yaml

# Use certifi's certificate list so HTTPS works on macOS python.org installs too.
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

FILES = [
    "expy-tky_202110.csv.gz",               # October speeds
    "expy-tky_202111.csv.gz",               # November speeds
    "expy-tky_202112.csv.gz",               # December speeds (test month)
    "expy-tky_adj01.npy",                   # 0/1 adjacency: are two links connected?
    "expy-tky_adjdis.npy",                  # distance-based adjacency
    "expy-tky_link_info_add_end_latlon.csv",  # road name + start/end lat/lon per link
    "tokyo_link_idx.csv",                   # indices of the 1843-link subset
]


def download(config_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(config_path))["data"]
    os.makedirs(cfg["raw_dir"], exist_ok=True)

    for name in FILES:
        dest = os.path.join(cfg["raw_dir"], name)
        if os.path.exists(dest):
            print(f"[skip] {name} already downloaded")
            continue
        url = f"{cfg['base_url']}/{name}"
        print(f"[get ] {name} ...", flush=True)
        tmp = dest + ".part"  # write to .part first so a broken download never looks finished
        with urllib.request.urlopen(url, context=SSL_CTX) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.rename(tmp, dest)
        print(f"       {os.path.getsize(dest) / 1e6:.1f} MB")


if __name__ == "__main__":
    download()
