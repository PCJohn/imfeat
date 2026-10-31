"""Latency of every code commit of a branch: build each, run the benchmark suite on it, keep
the logs, then tabulate and chart the numbers. The bundle reports (docs/oct-28-2026-
optimization-bundle.md, docs/oct-30-2026-optimization-bundle.md) were made with it.

    python scripts/latency_by_commit.py sweep  [--base REV] [--head REV] [--out DIR] \
                                               [--include REV ...] [--cores 1,2,3,4] \
                                               [--k EXPR] [--tag SUFFIX]
    python scripts/latency_by_commit.py report [--out DIR] [--tag SUFFIX[,SUFFIX...]]

`sweep` lists the commits base..head that touch src/ or CMakeLists.txt (the base itself
first, after any `--include` commits, which are timed as given: the original base of an
earlier bundle, say), exports each with `git archive`, builds it with CMake into one reused build
directory (only core.cpp recompiles between commits), assembles an importable package and
runs `pytest -m bench -s tests/test_bench.py` of the CURRENT checkout against it, pinned to
`--cores` when given (taskset; pick idle physical cores). One log per commit lands in
DIR/logs. `report` parses the logs into DIR/latency_by_commit.csv (the minimum of every
latency the suite prints: one row per number, one column per commit) and draws
DIR/latency_by_commit.png: the target scenario (test_target_pyramid) at 1, 2 and 4 threads
and the 1024x1024x3 finest-64 stride-1 headline, commit by commit.

Environment: IMFEAT_HIGHWAY_SRC may name a checked-out Highway source to build against
(else CMake fetches it); the build uses the Python that runs this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ("src/", "CMakeLists.txt")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def code_commits(base: str, head: str, include: list[str] = ()) -> list[tuple[str, str]]:
    """(hash, subject) of the included commits, base, then every commit base..head that
    changes code, oldest first."""
    out = [(git("rev-parse", "--short", c), git("log", "-1", "--format=%s", c)) for c in include]
    out.append((git("rev-parse", "--short", base), git("log", "-1", "--format=%s", base)))
    for line in git("log", "--reverse", "--format=%h %s", f"{base}..{head}").splitlines():
        h, subject = line.split(" ", 1)
        files = git("show", "--format=", "--name-only", h).split()
        if any(f.startswith(CODE) for f in files):
            out.append((h, subject))
    return out


def export(commit: str, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    data = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dst)
    now = time.time()
    for f in dst.rglob("*"):  # the archive's mtimes are the commit's; the reused build
        os.utime(f, (now, now))  # directory must see them as newer than its objects


def module_digest(pkg: Path) -> str:
    so = next(f for f in (pkg / "imfeat").iterdir() if f.suffix in (".so", ".pyd"))
    return hashlib.md5(so.read_bytes()).hexdigest()


def build(src: Path, bld: Path, pkg: Path) -> None:
    cfg = [
        "cmake", "-S", str(src), "-B", str(bld), "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython_EXECUTABLE={sys.executable}",
        "-Dnanobind_DIR=" + subprocess.check_output(
            [sys.executable, "-m", "nanobind", "--cmake_dir"], text=True).strip(),
    ]
    hwy = os.environ.get("IMFEAT_HIGHWAY_SRC")
    if hwy:
        cfg.append(f"-DFETCHCONTENT_SOURCE_DIR_HIGHWAY={hwy}")
    subprocess.run(cfg, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", str(bld), "-j", str(os.cpu_count() or 4)], check=True,
                   stdout=subprocess.DEVNULL)
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "imfeat").mkdir(parents=True)
    for f in (src / "src" / "imfeat").iterdir():
        if f.suffix in (".py", ".typed"):
            shutil.copy(f, pkg / "imfeat" / f.name)
    so = [f for f in bld.iterdir() if (f.name.startswith("imfeat_core") and ".so" in f.name)
          or f.suffix == ".pyd"]
    if not so:
        raise SystemExit(f"no module built in {bld}")
    shutil.copy(so[0], pkg / "imfeat" / so[0].name)


def sweep(args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    work = out / "work"
    commits = code_commits(args.base, args.head, args.include)
    with open(out / "commits.txt", "w") as f:
        for i, (h, s) in enumerate(commits):
            f.write(f"{i:02d} {h} {s}\n")
    print(f"{len(commits)} commits -> {out}")
    for i, (h, subject) in enumerate(commits):
        log = out / "logs" / f"{i:02d}_{h}{args.tag}.txt"
        if log.exists() and args.resume:
            continue
        print(f"[{i:02d}/{len(commits) - 1}] {h} {subject[:60]}", flush=True)
        src, pkg = work / "src", work / "pkgs" / f"{i:02d}_{h}"
        if not (pkg / "imfeat").exists():  # a package built earlier (another tag) is reused
            export(h, src)
            build(src, work / "build", pkg)
        digest = module_digest(pkg)
        if i and digest == module_digest(work / "pkgs" / f"{i - 1:02d}_{commits[i - 1][0]}"):
            raise SystemExit(f"{h}: module byte-identical to the previous commit's; not timed")
        cmd = [sys.executable, "-m", "pytest", "-q", "-m", "bench", "-s", "-p", "no:cacheprovider",
               "tests/test_bench.py"]
        if args.k:
            cmd += ["-k", args.k]
        if args.cores:
            cmd = ["taskset", "-c", args.cores, *cmd]
        env = {**os.environ, "PYTHONPATH": str(pkg)}
        with open(log, "w") as f:
            f.write(f"# commit {h}: {subject}\n")
            f.flush()
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)


# --- the log format of tests/test_bench.py ---------------------------------------------
STATS = ("min", "mean", "std", "p50", "p90", "p99")
NUM = r"-?\d+\.\d+"
FLOAT = re.compile(rf"^{NUM}$")
THREAD_ROW = re.compile(
    rf"^\s*(threads=\d+)\s+({NUM})\s+{NUM}x\s+({NUM}(?:\s+{NUM}){{5}})\s+{NUM}x\s*$"
)
STRIDE_ROW = re.compile(rf"^\s*(stride=\d\*?)\s+({NUM})\s+{NUM}x")
CHAN_ROW = re.compile(rf"^\s*C=\s*(\d+):\s+({NUM}) ms")
LABEL_ROW = re.compile(r"^\s*(C=\d+)\s+ms\s+speedup")


def parse(log: Path) -> list[tuple[str, str, str, float]]:
    """(section, row, stat, ms) for every latency the log reports."""
    rows: list[tuple[str, str, str, float]] = []
    section, pending, sub, in_head = "", "", "", False
    title: list[str] = []
    for ln in log.read_text().splitlines():
        if ln.startswith("=" * 20):  # a head(): its title between two rules
            if in_head:
                section, sub = title[0].strip() if title else "", ""
            else:
                title = []
            in_head = not in_head
            continue
        if in_head:
            title.append(ln)
            continue
        if " | " in ln and "finest-" in ln:  # test_latency: the config line, stats two below
            pending = ln.split("|", 1)[1].strip()
            continue
        if ln.strip().startswith("target:"):
            pending = ln.strip()
            continue
        if ln.strip().startswith("ms:"):
            continue
        m = THREAD_ROW.match(ln)
        if m:
            rows.append((section, m.group(1), "raw_min", float(m.group(2))))
            for st, v in zip(STATS, m.group(3).split()):
                rows.append((section, m.group(1), st, float(v)))
            continue
        m = LABEL_ROW.match(ln)
        if m:
            sub = m.group(1) + " "
            continue
        m = STRIDE_ROW.match(ln)
        if m:
            rows.append((section, sub + m.group(1), "min", float(m.group(2))))
            continue
        m = CHAN_ROW.match(ln)
        if m:
            rows.append((section, f"C={m.group(1)}", "min", float(m.group(2))))
            continue
        toks = ln.split()
        first = next((i for i, t in enumerate(toks) if FLOAT.match(t)), None)
        if first is None:
            continue
        nums = []
        for t in toks[first:]:
            if not FLOAT.match(t):
                break
            nums.append(float(t))
        if len(nums) < 6:
            continue
        label = " ".join(toks[:first])
        if pending and not label:  # test_latency / test_target_pyramid: stats under "ms:"
            label, section_ = pending, "target" if "target" in pending else "latency"
            pending = ""
        elif label:
            section_ = section
        else:
            continue
        for st, v in zip(STATS, nums):
            rows.append((section_, label, st, v))
    return rows


TARGET = "target: 1024x1024x3 finest-64 to 1x1 (7 levels) stride=1 threads="
KEY = {  # the chart's series: (section, row label) -> legend
    ("target", TARGET + "1"): "target, 1 thread",
    ("target", TARGET + "2"): "target, 2 threads",
    ("target", TARGET + "4"): "target, 4 threads",
    ("latency", "(1024, 1024, 3) finest-64 stride=1"): (
        "1024x1024x3 finest-64 s1, 1 thread (4 levels, no conversion)"
    ),
}


def report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    commits = [ln.split(" ", 2) for ln in (out / "commits.txt").read_text().splitlines()]
    table: dict[str, list[float | None]] = {k: [] for k in KEY.values()}
    tags = args.tag.split(",")  # several runs: the minimum per number over them
    tag = args.name or (tags[0] if len(tags) == 1 else "")
    wide: dict[tuple[str, str, str], list[str]] = {}  # (section, row, stat) -> a cell per commit
    for n, (i, h, _) in enumerate(commits):
        best: dict[tuple[str, str, str], float] = {}
        for t in tags:
            log = out / "logs" / f"{i}_{h}{t}.txt"
            for sec, row, st, v in parse(log) if log.exists() else []:
                best[sec, row, st] = min(v, best.get((sec, row, st), v))
        for (sec, row, st), v in best.items():
            if st in ("min", "raw_min"):  # the statistic the method reads; the rest is run noise
                wide.setdefault((sec, row, st), [""] * len(commits))[n] = f"{v:.3f}"
        got = {(sec, row): v for (sec, row, st), v in best.items() if st == "min"}
        for key, name in KEY.items():
            table[name].append(got.get(key))
    with open(out / f"latency_by_commit{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)  # one row per number, one column per commit (subjects: commits.txt)
        w.writerow(["section", "row", "stat", *(f"{i}_{h}" for i, h, _ in commits)])
        for (sec, row, st), cells in wide.items():
            w.writerow([sec, row, st, *cells])
    with open(out / f"latency_by_commit{tag}.md", "w") as f:
        f.write("| # | commit | " + " | ".join(table) + " | subject |\n")
        f.write("|---|---|" + "---|" * len(table) + "---|\n")
        for n, (i, h, subject) in enumerate(commits):
            cells = " | ".join("" if table[k][n] is None else f"{table[k][n]:.3f}" for k in table)
            f.write(f"| {i} | {h} | {cells} | {subject} |\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: csv and table written, no chart")
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = list(range(len(commits)))
    for name, ys in table.items():
        ax.plot(x, [y if y is not None else float("nan") for y in ys], marker="o", ms=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([h for _, h, _ in commits], rotation=90, fontsize=7)
    ax.set_ylabel("min latency, ms")
    ax.set_xlabel("code commit, oldest first (0 = the base)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("imfeat latency by commit (pinned minima; tests/test_bench.py)")
    fig.tight_layout()
    fig.savefig(out / f"latency_by_commit{tag}.png", dpi=130)
    print("wrote", out / f"latency_by_commit{tag}.csv", out / f"latency_by_commit{tag}.md",
          out / f"latency_by_commit{tag}.png")
    if args.zoom_from is None:
        return
    # the same series from one commit on, each on its own axis: a bundle of small steps
    # is invisible on the axis the original build sets
    z = args.zoom_from
    fig, axes = plt.subplots(len(table), 1, figsize=(11, 2.2 * len(table)), sharex=True)
    for ax, (name, ys) in zip(axes, table.items()):
        ax.plot(x[z:], [y if y is not None else float("nan") for y in ys[z:]], marker="o", ms=4)
        ax.set_ylabel("ms", fontsize=8)
        ax.set_title(name, fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xticks(x[z:])
    axes[-1].set_xticklabels([h for _, h, _ in commits[z:]], rotation=90, fontsize=7)
    axes[-1].set_xlabel(f"code commit, oldest first (from {commits[z][1]})")
    fig.suptitle("imfeat latency by commit, the bundle alone (pinned minima)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"latency_by_commit{tag}_zoom.png", dpi=130)
    print("wrote", out / f"latency_by_commit{tag}_zoom.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--base", default=None, help="base commit (default: merge-base with main)")
    s.add_argument("--head", default="HEAD")
    s.add_argument("--include", action="append", default=[],
                   help="a commit timed before the base, as given (repeatable)")
    s.add_argument("--out", default="docs/oct-28-2026-optimization-bundle")
    s.add_argument("--cores", default=None, help="taskset core list, e.g. 1,2,3,4")
    s.add_argument("--resume", action="store_true", help="skip commits whose log exists")
    s.add_argument("--k", default=None, help="pytest -k expression: a subset of the bench")
    s.add_argument("--tag", default="", help="suffix of the log names, e.g. _run2")
    r = sub.add_parser("report")
    r.add_argument("--out", default="docs/oct-28-2026-optimization-bundle")
    r.add_argument("--tag", default="",
                   help="which logs to report (their name suffix); several, comma-separated, "
                        "report the minimum per number over those runs")
    r.add_argument("--name", default="", help="output suffix when several tags are given")
    r.add_argument("--zoom-from", type=int, default=None,
                   help="also chart the series from this commit index on, one axis each")
    args = ap.parse_args()
    if args.cmd == "sweep":
        args.base = args.base or git("merge-base", "main", args.head)
        sweep(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
