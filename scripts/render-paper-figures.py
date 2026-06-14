from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflexlm.paper_figures import render_figure_sources
from reflexlm.paper_figures import write_ai_prototype_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Render editable Paper B figure sources.")
    parser.add_argument("--source-dir", default="docs/figures/src")
    parser.add_argument("--export-dir", default="docs/figures/export")
    parser.add_argument("--drawio-dir", default="docs/figures/drawio")
    parser.add_argument("--paper-figures-dir", default="docs/paper_b/figures")
    parser.add_argument("--paper-drawio-dir", default="docs/paper_b/figures")
    parser.add_argument("--manifest-json", default="docs/figures/export/figure_manifest.json")
    parser.add_argument("--ai-prototype-dir", default="docs/figures/ai_prototypes")
    parser.add_argument("--ai-prototype-model", default="gpt-image-2")
    args = parser.parse_args()

    ai_prototypes = write_ai_prototype_prompts(
        source_dir=args.source_dir,
        output_dir=args.ai_prototype_dir,
        model_family=args.ai_prototype_model,
    )
    exported = render_figure_sources(
        source_dir=args.source_dir,
        export_dir=args.export_dir,
        paper_figures_dir=args.paper_figures_dir,
        drawio_dir=args.drawio_dir,
        paper_drawio_dir=args.paper_drawio_dir,
    )
    manifest = {
        "figure_pipeline": "editable_source_to_drawio_svg_pdf_png",
        "drawio_workflow": "Agents365-ai/drawio-skill compatible editable XML; CLI export is optional.",
        "source_dir": args.source_dir,
        "export_dir": args.export_dir,
        "drawio_dir": args.drawio_dir,
        "paper_figures_dir": args.paper_figures_dir,
        "paper_drawio_dir": args.paper_drawio_dir,
        "ai_prototype_dir": args.ai_prototype_dir,
        "ai_prototype_workflow": (
            "GPT image prototypes are visual references only; update editable "
            "Mermaid/DOT/draw.io sources before using a figure in the paper."
        ),
        "ai_prototypes": ai_prototypes,
        "figures": exported,
    }
    manifest_path = Path(args.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
