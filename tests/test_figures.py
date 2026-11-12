"""The README's figures come from results by script, and a missing input skips its figure."""

import json

from bench import figures


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def results_tree(root):
    write(root / "profile/before/rates.json", json.dumps({"prefill_tokens_per_s": {"0": 100, "8192": 50}}))
    write(root / "profile/after2/rates.json", json.dumps({"prefill_tokens_per_s": {"0": 150, "8192": 90}}))
    write(root / "profile/vllm/rates_quick.json", json.dumps({"prefill_tokens_per_s": {"0": 200}}))
    header = "layout,batch,tokens,unique_pct\n"
    write(root / "decode_bandwidth/decode_bandwidth.csv", header + "fragmented,1,1024,15\nfragmented,1,4096,16\nfragmented,32,1024,64\ncontiguous,1,1024,14\n")
    crossover = "tokens,swap_s,recompute_s\n512,0.006,0.09\n1024,0.012,0.19\n"
    write(root / "kv_crossover/kv_crossover.csv", crossover)
    write(root / "kv_crossover_native_bf16/kv_crossover.csv", crossover)
    for system, jct in (("layover-combined", 600.0), ("vllm", 900.0)):
        for run in range(2):
            record = {"system": system, "status": "ok", "result": {"metrics": {"mean_jct_s": jct + run}}}
            write(root / f"matrix/runs/w/{system}/load100/run-{run}.json", json.dumps(record))
    write(root / "matrix/runs/w/vllm/load100/run-9.json", json.dumps({"system": "vllm", "status": "failed"}))


def test_every_figure_is_drawn_from_its_results(tmp_path, capsys):
    results_tree(tmp_path / "r")
    assert figures.main(["--results", str(tmp_path / "r"), "--out", str(tmp_path / "o")]) == 0
    names = {path.name for path in (tmp_path / "o").iterdir()}
    assert names == {"matrix_jct.png", "prefill.png", "decode_bandwidth.png", "crossover.png"}
    assert all((tmp_path / "o" / name).read_bytes()[:4] == b"\x89PNG" for name in names)


def test_only_finished_runs_count():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results_tree(root)
        assert figures.matrix_runs(root) == {"layover-combined": [600.0, 601.0], "vllm": [900.0, 901.0]}


def test_missing_results_skip_their_figures(tmp_path, capsys):
    assert figures.main(["--results", str(tmp_path / "empty"), "--out", str(tmp_path / "o")]) == 0
    assert capsys.readouterr().out.count("skipped, no results") == len(figures.FIGURES)
    assert not (tmp_path / "o").exists()
