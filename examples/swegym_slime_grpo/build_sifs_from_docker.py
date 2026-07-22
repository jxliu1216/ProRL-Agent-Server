#!/usr/bin/env python3
"""Build Apptainer SIF images from locally-loaded Docker images for 16 SWE-Gym tasks.

Unlike prepare_apptainer_images.py which pulls from a remote registry, this script
converts Docker images that have already been loaded via ``docker load`` into
Apptainer SIF format via ``apptainer build ... docker-daemon://``.

The output SIF files are named ``{instance_id}.sif`` — exactly what the Polar
runtime and the rest of the launch_e2e pipeline expect.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from sample_tasks import registry_image_for_instance_id

# ---- The 64 instance IDs matching the locally-loaded Docker images ----
INSTANCE_IDS = [
    # conan-io (3 tasks)
    "conan-io__conan-13403",
    "conan-io__conan-15422",
    "conan-io__conan-15699",
    # dask (6 tasks)
    "dask__dask-10972",
    "dask__dask-7191",
    "dask__dask-8686",
    "dask__dask-9213",
    "dask__dask-9378",
    "dask__dask-9627",
    # getmoto/moto (15 tasks)
    "getmoto__moto-4950",
    "getmoto__moto-4986",
    "getmoto__moto-5020",
    "getmoto__moto-5134",
    "getmoto__moto-5582",
    "getmoto__moto-5587",
    "getmoto__moto-5865",
    "getmoto__moto-5959",
    "getmoto__moto-6114",
    "getmoto__moto-6178",
    "getmoto__moto-6299",
    "getmoto__moto-6469",
    "getmoto__moto-7111",
    "getmoto__moto-7167",
    "getmoto__moto-7212",
    # iterative/dvc (5 tasks)
    "iterative__dvc-1651",
    "iterative__dvc-2231",
    "iterative__dvc-4778",
    "iterative__dvc-4785",
    "iterative__dvc-5839",
    # pandas-dev (13 tasks)
    "pandas-dev__pandas-49118",
    "pandas-dev__pandas-49766",
    "pandas-dev__pandas-50713",
    "pandas-dev__pandas-51605",
    "pandas-dev__pandas-51936",
    "pandas-dev__pandas-52076",
    "pandas-dev__pandas-52077",
    "pandas-dev__pandas-52516",
    "pandas-dev__pandas-53856",
    "pandas-dev__pandas-57058",
    "pandas-dev__pandas-57089",
    "pandas-dev__pandas-57173",
    "pandas-dev__pandas-57957",
    # Project-MONAI (12 tasks)
    "Project-MONAI__MONAI-1571",
    "Project-MONAI__MONAI-2238",
    "Project-MONAI__MONAI-2696",
    "Project-MONAI__MONAI-3403",
    "Project-MONAI__MONAI-4109",
    "Project-MONAI__MONAI-4583",
    "Project-MONAI__MONAI-5183",
    "Project-MONAI__MONAI-5423",
    "Project-MONAI__MONAI-5543",
    "Project-MONAI__MONAI-5640",
    "Project-MONAI__MONAI-6560",
    "Project-MONAI__MONAI-6756",
    # pydantic (4 tasks)
    "pydantic__pydantic-8004",
    "pydantic__pydantic-8072",
    "pydantic__pydantic-8316",
    "pydantic__pydantic-8511",
    # python/mypy (6 tasks)
    "python__mypy-11135",
    "python__mypy-11567",
    "python__mypy-12943",
    "python__mypy-15184",
    "python__mypy-16869",
    "python__mypy-5617",
]

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "tmp" / "swegym_apptainer_images"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "tmp" / "apptainer_cache"
DEFAULT_TMP_DIR = PROJECT_ROOT / "tmp" / "apptainer_tmp"
DEFAULT_AGENT_CLI_DIR = PROJECT_ROOT / "tmp" / "swegym_agent_cli" / "opt_node"
NODE_VERSION = "22.11.0"
NODE_DIST_URL = (
    f"https://nodejs.org/dist/v{NODE_VERSION}/"
    f"node-v{NODE_VERSION}-linux-x64.tar.xz"
)
DEFAULT_NODE_DIST = PROJECT_ROOT / "required_packages" / f"node-v{NODE_VERSION}-linux-x64.tar.xz"
AGENT_CLI_PACKAGES = (
    "@openai/codex@0.144.5",  # pinned to match DEFAULT_CODEX_VERSION in polar/agent/presets/codex.py
    # "@anthropic-ai/claude-code@latest",
    # "@qwen-code/qwen-code@latest",
    # "opencode-ai@latest",
    # "@mariozechner/pi-coding-agent@latest",
)
NPM_REGISTRY = os.environ.get("NPM_REGISTRY", "https://registry.npmmirror.com")
REQUIRED_AGENT_BINS = (
    "node",
    "npm",
    "npx",
    "codex",
    # "claude",
    # "qwen",
    # "opencode",
    # "pi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory for prepared .sif images.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Apptainer cache directory.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help="Apptainer temporary build directory.",
    )
    parser.add_argument(
        "--agent-cli-dir",
        type=Path,
        default=DEFAULT_AGENT_CLI_DIR,
        help="Host directory mounted as /opt/node in task containers.",
    )
    parser.add_argument(
        "--node-dist",
        type=Path,
        default=DEFAULT_NODE_DIST,
        help="Path to a local node-v%s-linux-x64.tar.xz tarball. "
             "When the file exists it is extracted directly; "
             "otherwise falls back to downloading from nodejs.org." % NODE_VERSION,
    )
    parser.add_argument(
        "--skip-cli",
        action="store_true",
        help="Only prepare SIF images; do not prepare the shared CLI directory.",
    )
    parser.add_argument(
        "--force-cli",
        action="store_true",
        help="Rebuild and re-extract the shared Node/agent CLI directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild SIF images even when the output file exists.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Number of concurrent apptainer build jobs.",
    )
    return parser.parse_args()


def sif_path_for_instance(instance_id: str, image_dir: Path) -> Path:
    if "/" in instance_id or "\0" in instance_id:
        raise ValueError(f"instance_id is not safe for a filename: {instance_id!r}")
    return image_dir / f"{instance_id}.sif"


def image_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _apptainer_binary() -> str:
    override = os.environ.get("POLAR_APPTAINER_BIN")
    if override:
        return override
    import shutil
    for candidate in ("/usr/bin/apptainer", "/bin/apptainer"):
        if Path(candidate).is_file():
            return candidate
    resolved = shutil.which("apptainer")
    if resolved:
        return resolved
    return "apptainer"


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, check=True, env=env)


def _agent_cli_missing_bins(agent_cli_dir: Path) -> list[str]:
    return [
        name for name in REQUIRED_AGENT_BINS
        if not (agent_cli_dir / "bin" / name).is_file()
    ]


def ensure_agent_cli_dir(agent_cli_dir: Path, *, force: bool, node_dist: Path | None = None) -> None:
    missing_bins = _agent_cli_missing_bins(agent_cli_dir)
    if not missing_bins and not force:
        print(f"Shared agent CLI directory already exists: {agent_cli_dir}")
        return

    agent_cli_dir = agent_cli_dir.resolve()
    agent_cli_dir.parent.mkdir(parents=True, exist_ok=True)
    if agent_cli_dir.exists():
        shutil.rmtree(agent_cli_dir)
    agent_cli_dir.mkdir(parents=True)

    if node_dist is not None and node_dist.is_file():
        print(f"Using local Node.js tarball: {node_dist}")
        run_command([
            "tar", "-xJ",
            "--strip-components=1",
            "-f", str(node_dist),
            "-C", str(agent_cli_dir),
        ])
    else:
        print(f"Downloading Node.js from {NODE_DIST_URL}")
        run_command([
            "bash",
            "-c",
            (
                f"curl -fsSL {shlex.quote(NODE_DIST_URL)} | "
                f"tar -xJ --strip-components=1 -C {shlex.quote(str(agent_cli_dir))}"
            ),
        ])

    npm_bin = agent_cli_dir / "bin" / "npm"
    env = {
        **os.environ,
        "PATH": f"{agent_cli_dir / 'bin'}:{os.environ.get('PATH', '')}",
    }
    run_command(
        [
            str(npm_bin),
            "install",
            "-g",
            "--no-audit",
            "--no-fund",
            f"--prefix={agent_cli_dir}",
            f"--registry={NPM_REGISTRY}",
            *AGENT_CLI_PACKAGES,
        ],
        env=env,
    )

    missing_bins = _agent_cli_missing_bins(agent_cli_dir)
    if missing_bins:
        raise RuntimeError(
            "agent CLI setup did not create expected executable(s): "
            + ", ".join(missing_bins)
        )
    print(f"Prepared shared agent CLI directory: {agent_cli_dir}")


def ensure_sif(
    instance_id: str,
    *,
    image_dir: Path,
    force: bool,
    env: dict[str, str],
) -> tuple[str, str]:
    target = sif_path_for_instance(instance_id, image_dir)
    if image_ready(target) and not force:
        return ("skipped", str(target))

    # Compute the Docker image name from instance_id (same mapping as prepare_apptainer_images.py)
    docker_image = registry_image_for_instance_id(instance_id)
    # Strip docker:// prefix if present — we build from docker-daemon instead
    if docker_image.startswith("docker://"):
        docker_image = docker_image[len("docker://"):]
    # Strip the :latest tag for the daemon reference; apptainer resolves it correctly
    daemon_ref = f"docker-daemon://{docker_image}"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()

    command = [
        _apptainer_binary(),
        "build",
        "--force",
        str(tmp_path),
        daemon_ref,
    ]
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    try:
        subprocess.run(command, check=True, env=env)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return ("built", str(target))


def main() -> int:
    args = parse_args()
    if shutil.which(_apptainer_binary()) is None and not Path(_apptainer_binary()).is_file():
        raise SystemExit("apptainer not found — is it installed or loaded as a module?")

    if not args.skip_cli:
        ensure_agent_cli_dir(args.agent_cli_dir, force=args.force_cli, node_dist=args.node_dist)

    env = {
        **os.environ,
        "APPTAINER_CACHEDIR": str(args.cache_dir.resolve()),
        "APPTAINER_TMPDIR": str(args.tmp_dir.resolve()),
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Building {len(INSTANCE_IDS)} Apptainer SIF image(s) from local Docker "
        f"with {max(args.jobs, 1)} job(s).",
        flush=True,
    )
    print(f"Image dir: {args.image_dir.resolve()}", flush=True)
    print(f"Cache dir: {args.cache_dir.resolve()}", flush=True)
    print(f"Tmp dir:  {args.tmp_dir.resolve()}", flush=True)

    if args.jobs <= 1:
        for iid in INSTANCE_IDS:
            status, path = ensure_sif(
                iid, image_dir=args.image_dir, force=args.force, env=env
            )
            print(f"{status}: {path}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(
                    ensure_sif, iid, image_dir=args.image_dir, force=args.force, env=env
                )
                for iid in INSTANCE_IDS
            ]
            for future in as_completed(futures):
                status, path = future.result()
                print(f"{status}: {path}", flush=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
