"""Monkey-patch swegym evaluator to prefer a local environment / requirements cache.

Before calling ``requests.get()`` on raw.githubusercontent.com, this patch
scans ``required_packages/swegym_env_cache/{repo_slug}/`` for a file whose
name starts with ``{commit}.`` — regardless of the path suffix. If found,
the file content is returned directly, bypassing the proxy and avoiding
SSL verification failures.

Covers both:
  - ``get_environment_yml_by_commit``  (conda environment YAML)
  - ``get_requirements_by_commit``     (pip requirements)

Usage (one-shot, called from launch_e2e.sh after swegym is installed)::

    python3 -c "from polar.patches.swegym_env_cache import apply; apply()"
"""

from __future__ import annotations

from pathlib import Path

CACHE_DIR = Path(__file__).parents[3] / "required_packages" / "swegym_env_cache"

_originals: dict[str, object] = {}


def _lookup_cache(repo: str, commit: str) -> str | None:
    """Scan ``{repo_slug}/{commit}.*`` in the cache directory.

    Returns the file content on the first match, or ``None`` on miss.
    """
    slug = repo.replace("/", "__")
    cache_dir = CACHE_DIR / slug
    if not cache_dir.is_dir():
        return None
    prefix = f"{commit}."
    for entry in cache_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.stat().st_size == 0:
            continue
        if entry.name.startswith(prefix):
            return entry.read_text()
    return None


def _rename_env(content: str, env_name: str) -> str:
    """Replace the ``name:`` line in a conda environment YAML."""
    lines = content.split("\n")
    cleaned: list[str] = []
    for line in lines:
        if line.startswith("name:"):
            cleaned.append(f"name: {env_name}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def apply() -> None:
    """Replace swegym harness functions with cache-aware versions."""
    from swegym.harness import utils

    # —— get_environment_yml_by_commit(repo, commit, env_name) ——
    _env_original = utils.get_environment_yml_by_commit

    def _env_patched(repo: str, commit: str, env_name: str) -> str:
        cached = _lookup_cache(repo, commit)
        if cached is not None:
            return _rename_env(cached, env_name)
        return _env_original(repo, commit, env_name)

    utils.get_environment_yml_by_commit = _env_patched
    _originals["get_environment_yml_by_commit"] = _env_original

    # —— get_requirements_by_commit(repo, commit) ——
    _reqs_original = utils.get_requirements_by_commit

    def _reqs_patched(repo: str, commit: str) -> str:
        cached = _lookup_cache(repo, commit)
        if cached is not None:
            return cached
        return _reqs_original(repo, commit)

    utils.get_requirements_by_commit = _reqs_patched
    _originals["get_requirements_by_commit"] = _reqs_original


def revert() -> None:
    """Restore the original swegym functions (used for testing only)."""
    if not _originals:
        return
    from swegym.harness import utils
    for name, func in _originals.items():
        setattr(utils, name, func)
    _originals.clear()
