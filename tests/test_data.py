"""Tests of the data shipped under ``data/`` (CEC 2017 definitions, engineering problems)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from edeco.benchmarks import cec2017
from edeco.benchmarks.engineering import engineering_definitions

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_shipped_files_match_their_checksums():
    lines = (DATA_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    readme = (DATA_DIR / "README.md").read_text(encoding="utf-8")
    assert len(lines) == 5
    for line in lines:
        digest, name = line.split()
        assert hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == digest, name
        assert digest in readme, f"checksum of {name} missing from data/README.md"


def test_repository_data_is_the_default_cec2017_source(monkeypatch):
    monkeypatch.delenv(cec2017.DATA_ENV_VAR, raising=False)
    assert cec2017.find_data_dir() == DATA_DIR / "cec2017"


@pytest.mark.parametrize("dim", cec2017.SUPPORTED_DIMS)
def test_cec2017_archive_layout(dim):
    with np.load(DATA_DIR / "cec2017" / f"cec2017_D{dim}.npz") as archive:
        keys = set(archive.files)
        assert {f"M_{k}" for k in cec2017.FUNCTION_IDS} <= keys
        assert {f"shift_{k}" for k in cec2017.FUNCTION_IDS} <= keys
        shuffled = {k for k in cec2017.FUNCTION_IDS if 11 <= k <= 20 or k >= 29}
        assert {key for key in keys if key.startswith("shuffle_")} == {
            f"shuffle_{k}" for k in shuffled
        }
        assert archive["M_1"].shape == (1, dim, dim)
        assert archive["M_21"].shape == (10, dim, dim)
        assert archive["shift_30"].shape == (10, dim)
        # Transformation matrices are regular and shuffles are permutations of 1..D.
        assert np.linalg.matrix_rank(archive["M_5"][0]) == dim
        assert sorted(archive["shuffle_11"][0]) == list(range(1, dim + 1))
        assert np.all(np.abs(archive["shift_1"]) <= 80.0)


def test_official_text_files_are_read_identically(tmp_path, monkeypatch):
    """A directory with the organisers' text files gives the same functions."""
    dim = 10
    with np.load(DATA_DIR / "cec2017" / f"cec2017_D{dim}.npz") as archive:
        for func_num in (1, 11, 21):
            rotation = archive[f"M_{func_num}"].reshape(-1, dim)
            np.savetxt(tmp_path / f"M_{func_num}_D{dim}.txt", rotation, fmt="%.17g")
            shift = np.zeros((archive[f"shift_{func_num}"].shape[0], 100))
            shift[:, :dim] = archive[f"shift_{func_num}"]
            np.savetxt(tmp_path / f"shift_data_{func_num}.txt", shift, fmt="%.17g")
        np.savetxt(tmp_path / f"shuffle_data_11_D{dim}.txt", archive["shuffle_11"], fmt="%d")
    x = np.linspace(-80.0, 80.0, dim)
    for func_num in (1, 11, 21):
        from_text = cec2017.CEC2017Function(func_num, dim, tmp_path)
        assert from_text(x) == cec2017.make_function(func_num, dim)(x)
    # The environment variable selects the directory as well.
    monkeypatch.setenv(cec2017.DATA_ENV_VAR, str(tmp_path))
    assert cec2017.find_data_dir() == tmp_path


def test_engineering_definitions_file_matches_the_code():
    shipped = json.loads((DATA_DIR / "engineering" / "problems.json").read_text(encoding="utf-8"))
    assert shipped["problems"] == engineering_definitions()
    assert [p["dimension"] for p in shipped["problems"]] == [3, 4, 2, 4, 7, 4, 10, 5, 5, 5]


def test_readme_links_and_paper_figures_resolve():
    """Every relative link and image of the README points to an existing file."""
    import re

    root = DATA_DIR.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r'src="([^"]+)"', readme) + re.findall(r"\]\(([^)#]+)\)", readme)
    relative = [t for t in targets if not t.startswith(("http://", "https://"))]
    assert len(relative) >= 15
    for target in relative:
        assert (root / target).exists(), f"README link does not resolve: {target}"
    figures = sorted((root / "assets" / "paper_figures").glob("fig*.png"))
    assert [path.name for path in figures] == [f"fig{i:02d}.png" for i in range(1, 9)]
    listing = (root / "assets" / "paper_figures" / "README.md").read_text(encoding="utf-8")
    assert all(path.name in listing for path in figures)
    assert all(path.stat().st_size < 1_500_000 for path in figures)


def test_cli_data_dir_option(tmp_path, monkeypatch, capsys):
    """`--data-dir` selects the CEC 2017 data and an empty directory is reported."""
    from edeco.cli import main

    # Registering the variable with monkeypatch restores its original state afterwards,
    # although main() sets it directly from --data-dir.
    monkeypatch.setenv(cec2017.DATA_ENV_VAR, str(DATA_DIR / "cec2017"))
    arguments = ["solve", "--suite", "cec2017", "--function", "1", "--dim", "10"]
    arguments += ["--pop-size", "10", "--max-fes", "200"]
    assert main([*arguments, "--data-dir", str(DATA_DIR / "cec2017")]) == 0
    assert "cec2017 / F1" in capsys.readouterr().out
    cec2017._cached.cache_clear()
    with pytest.raises(cec2017.CEC2017DataError):
        main([*arguments, "--data-dir", str(tmp_path)])
    cec2017._cached.cache_clear()
