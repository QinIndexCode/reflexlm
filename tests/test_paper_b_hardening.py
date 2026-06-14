import json
import re
from pathlib import Path

from reflexlm.paper_baseline_audit import build_baseline_zero_audit, write_baseline_zero_audit
from reflexlm.paper_figures import (
    audit_drawio_layout,
    audit_static_label_layout,
    build_ai_prototype_prompt,
    parse_figure_source,
    render_figure,
    write_ai_prototype_prompts,
    write_drawio,
)
from reflexlm.paper_tables import (
    build_all_paper_tables,
    build_homeostasis_mechanism_evidence_table,
    table_to_latex,
    write_paper_tables,
)


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_zero_audit_classifies_current_zero_controls() -> None:
    report = build_baseline_zero_audit(ROOT)

    assert report["passed"] is True
    assert report["unexplained_zero_count"] == 0
    assert report["interpretation"]["zero_controls_are_not_automatic_stronger_evidence"] is True
    assert any(phase["native_and_no_nsi_both_zero"] for phase in report["phases"])
    assert any(phase["full_zero"] for phase in report["phases"])

    zero_categories = {
        row["zero_category"]
        for phase in report["phases"]
        for row in phase["rows"]
        if row["zero_category"]
    }
    assert "expected_zero_due_to_missing_capability" in zero_categories
    assert "valid_zero_failure" in zero_categories
    assert "suspicious_zero_requires_redesign" not in zero_categories


def test_baseline_zero_audit_writes_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / "baseline_zero.json"
    output_md = tmp_path / "baseline_zero.md"

    report = write_baseline_zero_audit(output_json=output_json, output_md=output_md, root=ROOT)

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert saved["audit_family"] == "paper_b_baseline_zero_interpretability"
    assert saved["passed"] == report["passed"]
    assert "Row classifications" in markdown
    assert "suspicious_zero_requires_redesign" in markdown


def test_editable_figure_sources_parse_and_render(tmp_path: Path) -> None:
    source = ROOT / "docs" / "figures" / "src" / "nsi_architecture.mmd"
    spec = parse_figure_source(source)

    assert spec.title == "Native Nervous Interface Architecture"
    assert spec.subtitle == "Bounded command selection only; not production autonomy or open-ended repair"
    assert any(lane.label == "Observation layer" for lane in spec.lanes)
    assert "receptors" in spec.nodes
    assert any(edge.source == "receptors" and edge.target == "latent" for edge in spec.edges)

    outputs = render_figure(spec, tmp_path)
    assert {path.suffix for path in outputs} == {".svg", ".pdf", ".png"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_paper_b_drawio_sources_are_editable_and_non_overlapping(tmp_path: Path) -> None:
    source_dir = ROOT / "docs" / "figures" / "src"

    for source in sorted([*source_dir.glob("*.mmd"), *source_dir.glob("*.dot")]):
        spec = parse_figure_source(source)
        audit = audit_drawio_layout(spec)
        label_audit = audit_static_label_layout(spec)
        drawio_path = write_drawio(spec, tmp_path)
        drawio_xml = drawio_path.read_text(encoding="utf-8")

        assert audit["passed"] is True, (source.name, audit)
        assert label_audit["passed"] is True, (source.name, label_audit)
        assert label_audit["off_route_labels"] == [], (source.name, label_audit["off_route_labels"])
        assert label_audit["foreign_route_overlaps"] == [], (source.name, label_audit["foreign_route_overlaps"])
        assert label_audit["low_clearance_labels"] == [], (source.name, label_audit["low_clearance_labels"])
        assert audit["lane_title_intrusions"] == [], (source.name, audit["lane_title_intrusions"])
        assert audit["unconnected_nodes"] == [], (source.name, audit["unconnected_nodes"])
        assert audit["edge_node_crossings"] == [], (source.name, audit["edge_node_crossings"])
        assert audit["edge_route_conflicts"] == [], (source.name, audit["edge_route_conflicts"])
        assert "<mxfile" in drawio_xml
        assert "orthogonalEdgeStyle" in drawio_xml
        assert "gridSize=\"10\"" in drawio_xml
        if source.name == "nsi_architecture.mmd":
            assert audit["lane_count"] == 5
            assert "lane_observation" in drawio_xml
        assert "connection_label_legend" in drawio_xml
        assert "Connection labels" in drawio_xml
        assert "value=\"E1\"" in drawio_xml
        assert "as=\"offset\"" in drawio_xml
        assert drawio_path.name.endswith(".drawio")


def test_key_figures_have_visual_hierarchy_metadata() -> None:
    source_dir = ROOT / "docs" / "figures" / "src"

    for source in sorted([*source_dir.glob("*.mmd"), *source_dir.glob("*.dot")]):
        spec = parse_figure_source(source)
        assert spec.subtitle, source.name
        assert len(spec.lanes) >= 3, source.name


def test_ai_prototype_prompts_preserve_editable_source_authority(tmp_path: Path) -> None:
    source = ROOT / "docs" / "figures" / "src" / "nsi_architecture.mmd"
    spec = parse_figure_source(source)
    prototype = build_ai_prototype_prompt(spec)

    assert prototype.model_family == "gpt-image-2"
    assert "visual prototype" in prototype.boundary_note
    assert "Mermaid/DOT/draw.io" in prototype.boundary_note
    assert "bounded command selection only" in prototype.prompt
    assert "Observation layer" in prototype.prompt
    assert "Figure subtitle:" in prototype.prompt
    assert "No new architecture modules" in prototype.negative_prompt
    assert "candidate_0" in prototype.negative_prompt
    assert "Structured receptors" in prototype.prompt
    assert "Debug Cortex route" in prototype.prompt

    manifest = write_ai_prototype_prompts(
        ROOT / "docs" / "figures" / "src",
        tmp_path / "ai_prototypes",
    )
    prompt_path = Path(manifest["figures"]["nsi_architecture.mmd"]["prompt_path"])
    prompt_text = prompt_path.read_text(encoding="utf-8")

    assert manifest["ai_prototype_pipeline"] == "figure_spec_to_gpt_image_prompt_to_editable_vector_redraw"
    assert prompt_path.exists()
    assert "Status: visual prototype only" in prompt_text
    assert "No sealed-evaluation feedback loop" in prompt_text


def test_paper_b_latex_uses_generated_figures_not_ascii_diagrams() -> None:
    tex = (ROOT / "docs" / "paper_b" / "main.tex").read_text(encoding="utf-8")
    figure_refs = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", tex)

    assert "\\includegraphics" in tex
    assert "figures/nsi-architecture.pdf" in tex
    assert len(figure_refs) >= 6
    assert all((ROOT / "docs" / "paper_b" / figure).exists() for figure in figure_refs)
    assert "```" not in tex


def test_paper_b_tables_are_artifact_backed_and_escape_latex(tmp_path: Path) -> None:
    manifest = write_paper_tables(output_dir=tmp_path / "tables", manifest_json=tmp_path / "manifest.json", root=ROOT)
    table_slugs = set(manifest["tables"])

    assert {
        "claim_boundary",
        "positive_evidence_matrix",
        "negative_evidence",
        "baseline_zero_summary",
        "homeostasis_mechanism_evidence",
    } <= table_slugs
    assert all(Path(table["path"]).exists() for table in manifest["tables"].values())

    tables = {table.slug: table for table in build_all_paper_tables(ROOT)}
    positive = table_to_latex(tables["positive_evidence_matrix"])
    negative = table_to_latex(tables["negative_evidence"])
    baseline = table_to_latex(tables["baseline_zero_summary"])
    assert "Phase2R" in positive
    assert "dynamic pytest traces" in positive
    assert "Phase2L" in negative
    assert "sealed transfer failure" in negative
    assert "expected missing capability" in baseline
    assert "valid zero failure" in baseline
    assert "negative sealed-transfer evidence only" in baseline


def test_homeostasis_mechanism_table_is_artifact_backed() -> None:
    table = build_homeostasis_mechanism_evidence_table(ROOT)
    latex = table_to_latex(table)

    assert "HMAC-authenticated state artifacts" in latex
    assert "missing-key control fails closed" in latex
    assert "0.008929" in latex
    assert "external-machine replay not yet run" in latex


def test_paper_b_latex_references_generated_tables() -> None:
    tex = (ROOT / "docs" / "paper_b" / "main.tex").read_text(encoding="utf-8")
    table_refs = re.findall(r"\\input\{(tables/[^}]+\.tex)\}", tex)

    assert len(table_refs) >= 5
    assert all((ROOT / "docs" / "paper_b" / table).exists() for table in table_refs)
    assert "tab:positive-evidence" in tex
    assert "tab:negative-evidence" in tex
    assert "tab:homeostasis-mechanism-evidence" in tex


def test_architecture_iteration_review_preserves_claim_boundary() -> None:
    review = (ROOT / "docs" / "spec" / "architecture_iteration_review_2026-05-22.md").read_text(
        encoding="utf-8"
    )
    normalized = re.sub(r"\s+", " ", review)

    assert "does not yet prove production autonomy" in normalized
    assert "epoch-making architecture status" in normalized
    assert "No hardcoded test names" in normalized
    assert "Phase2S should be treated as a falsification benchmark" in normalized
