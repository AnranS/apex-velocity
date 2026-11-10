"""Download CC0 source files (Poly Haven). Each asset goes into its own fresh directory under _work/dl/.
   python3 fetch_cc0.py textures pine_bark:Diffuse,nor_gl jolcham_oak_bark_01:Diffuse,nor_gl ...
Files are data only; process them with `python3 -I`."""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "_work", "dl")
UA = {"User-Agent": "asset-fetch/1.0"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_model(asset, res):
    d = os.path.join(DL, asset)
    if os.path.isdir(d) and os.listdir(d):
        print("skip (exists):", asset)
        return
    os.makedirs(d, exist_ok=True)
    files = json.loads(get(f"https://api.polyhaven.com/files/{asset}"))
    entry = files["gltf"][res]["gltf"]
    todo = {os.path.basename(entry["url"]): entry["url"]}
    for rel, inc in entry.get("include", {}).items():
        rel = rel.replace("\\", "/")
        if rel.startswith("/") or ".." in rel.split("/") or ":" in rel:
            raise SystemExit(f"unsafe include path: {rel!r}")
        todo[rel] = inc["url"]
    for rel, url in todo.items():
        dst = os.path.join(d, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        data = get(url)
        with open(dst, "wb") as f:
            f.write(data)
        print("got", asset, rel, len(data) // 1024, "KB")
    with open(os.path.join(d, "SOURCE.txt"), "w") as f:
        f.write(f"https://polyhaven.com/a/{asset}\nLicense: CC0 1.0\n")


def main():
    res = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("1k", "2k") else "1k"
    if len(sys.argv) > 2 and sys.argv[2] == "model":
        for asset in sys.argv[3:]:
            fetch_model(asset, res)
        return
    for spec in sys.argv[2:]:
        asset, maps = spec.split(":")
        d = os.path.join(DL, asset)
        if os.path.isdir(d) and os.listdir(d):
            print("skip (exists):", asset)
            continue
        os.makedirs(d, exist_ok=True)
        files = json.loads(get(f"https://api.polyhaven.com/files/{asset}"))
        for m in maps.split(","):
            info = files[m][res]["jpg"]
            data = get(info["url"])
            name = os.path.basename(info["url"])
            with open(os.path.join(d, name), "wb") as f:
                f.write(data)
            print("got", asset, name, len(data) // 1024, "KB")
        with open(os.path.join(d, "SOURCE.txt"), "w") as f:
            f.write(f"https://polyhaven.com/a/{asset}\nLicense: CC0 1.0\n")


if __name__ == "__main__":
    main()
