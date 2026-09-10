#!/usr/bin/env python3
"""One checkpoint filename per arm, and every place that names it must agree.

    python .github/scripts/check_release_names.py

Four files name the released checkpoints: the four `configs/predict/*.py`, the README's model
table and its `hf_hub_download` block, and the `ARMS` dict in each of the two demo notebooks.
Nothing joins them at run time, so a rename that misses one of them is silent until a user hits
a 404. This compares them.

Standard library only, and it reads the files as text rather than importing them, so it runs in a
bare CI job with no torch and no weights.

Exit 0 if every source agrees, 1 otherwise, with the disagreement printed.
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The release naming decision: prepibind_<arm>_s<seed>_f<fold>[_fp16].pt
CKPT = re.compile(r"prepibind_(?P<arm>[a-z0-9_]+?)_s(?P<seed>\d+)_f(?P<fold>\d+)(?P<fp16>_fp16)?\.pt")
ARMS = ("qualitative", "ms", "ic50_500", "ic50_1000")

problems = []


def fail(msg):
    problems.append(msg)


def read(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


def from_predict_configs():
    """config basename -> checkpoint basename, as the four predict configs set `chkp_path`."""
    out = {}
    d = os.path.join(REPO, "configs", "predict")
    for name in sorted(n for n in os.listdir(d) if n.endswith(".py")):
        text = read("configs", "predict", name)
        hits = sorted({m.group(0) for m in CKPT.finditer(text)})
        if len(hits) != 1:
            fail(f"configs/predict/{name}: expected exactly one checkpoint filename, found {hits}")
            continue
        out[name] = hits[0]
    return out


def from_readme_table(readme):
    """config path -> checkpoint, from the `Available Models` table."""
    out = {}
    for line in readme.splitlines():
        if not line.startswith("|"):
            continue
        cfg = re.search(r"configs/predict/(config_[a-z0-9_]+\.py)", line)
        ck = CKPT.search(line)
        if cfg and ck:
            out[cfg.group(1)] = ck.group(0)
    return out


def from_readme_downloads(readme):
    """checkpoint -> repo_id, from the hf_hub_download block."""
    out = {}
    for m in re.finditer(r"""hf_hub_download\(\s*repo_id\s*=\s*["']([^"']+)["']\s*,\s*"""
                         r"""filename\s*=\s*["']([^"']+)["']""", readme):
        repo, fn = m.groups()
        if CKPT.fullmatch(fn):
            out[fn] = repo
    return out


def from_notebook(path):
    """config basename -> checkpoint, from the notebook's ARMS dict."""
    nb = json.loads(read(path))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    out = {}
    for m in re.finditer(r"""["'](config_[a-z0-9_]+\.py)["']\s*,\s*["'](prepibind_[^"']+\.pt)["']""",
                         src):
        out[m.group(1)] = m.group(2)
    return out


def main():
    readme = read("README.md")

    configs = from_predict_configs()
    if len(configs) != 4:
        fail(f"expected 4 predict configs, found {sorted(configs)}")

    # Every config ships a float16 checkpoint, and the arm in the filename must be a known arm.
    for name, ck in sorted(configs.items()):
        m = CKPT.fullmatch(ck)
        if not m.group("fp16"):
            fail(f"configs/predict/{name} loads {ck}, which is not a _fp16 file")
        if m.group("arm") not in ARMS:
            fail(f"configs/predict/{name} loads {ck}: unknown arm {m.group('arm')!r}")
    if len(set(configs.values())) != len(configs):
        fail(f"two predict configs load the same checkpoint: {sorted(configs.values())}")

    table = from_readme_table(readme)
    if table != configs:
        for k in sorted(set(table) | set(configs)):
            if table.get(k) != configs.get(k):
                fail(f"README model table says {k} -> {table.get(k)!r}; "
                     f"configs/predict/{k} loads {configs.get(k)!r}")

    downloads = from_readme_downloads(readme)
    if set(downloads) != set(configs.values()):
        fail("README's hf_hub_download block lists "
             f"{sorted(downloads)}, the predict configs load {sorted(configs.values())}")
    for fn, repo in sorted(downloads.items()):
        if repo != "daylight-00/prepibind-demo":
            fail(f"README downloads the float16 {fn} from {repo!r}, "
                 "not daylight-00/prepibind-demo")

    for nb in ("demo/run.ipynb", "demo/run_colab.ipynb"):
        arms = from_notebook(nb)
        if arms != configs:
            for k in sorted(set(arms) | set(configs)):
                if arms.get(k) != configs.get(k):
                    fail(f"{nb} ARMS maps {k} -> {arms.get(k)!r}; "
                         f"configs/predict/{k} loads {configs.get(k)!r}")

    # Anything README names as a checkpoint must be one of the eight released files: the four
    # float16 ones above and the four float32 research files with the same stems.
    released = set(configs.values()) | {c.replace("_fp16.pt", ".pt") for c in configs.values()}
    for m in CKPT.finditer(readme):
        if m.group(0) not in released:
            fail(f"README names {m.group(0)}, which no predict config loads")

    print(f"predict configs : {len(configs)}")
    for name, ck in sorted(configs.items()):
        print(f"  {name:24s} -> {ck}")
    print(f"README table    : {'agrees' if table == configs else 'DISAGREES'}")
    print(f"README downloads: {len(downloads)} float16 files")
    print(f"notebooks       : run.ipynb, run_colab.ipynb")

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nall sources agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
