from pathlib import Path

from reflexlm.cli.run_phase2bs_live_text_runtime_baselines import _sha256


def test_phase2bs_manifest_hash_is_content_based(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"same": true}', encoding="utf-8")
    second.write_text('{"same": true}', encoding="utf-8")

    assert _sha256(first) == _sha256(second)

    second.write_text('{"same": false}', encoding="utf-8")
    assert _sha256(first) != _sha256(second)
