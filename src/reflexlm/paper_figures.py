from __future__ import annotations

import json
import math
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class FigureNode:
    node_id: str
    x: float
    y: float
    label: str
    role: str = "default"


@dataclass(frozen=True)
class FigureEdge:
    source: str
    target: str
    label: str = ""


@dataclass(frozen=True)
class FigureLane:
    lane_id: str
    y0: float
    y1: float
    label: str
    role: str = "default"


@dataclass
class FigureSpec:
    slug: str
    title: str = ""
    subtitle: str = ""
    width: float = 12.0
    height: float = 7.0
    lanes: list[FigureLane] = field(default_factory=list)
    nodes: dict[str, FigureNode] = field(default_factory=dict)
    edges: list[FigureEdge] = field(default_factory=list)


@dataclass(frozen=True)
class PlotRect:
    x0: float
    y0: float
    x1: float
    y1: float

    def expanded(self, margin: float) -> "PlotRect":
        return PlotRect(self.x0 - margin, self.y0 - margin, self.x1 + margin, self.y1 + margin)

    def overlaps(self, other: "PlotRect") -> bool:
        return self.x0 < other.x1 and self.x1 > other.x0 and self.y0 < other.y1 and self.y1 > other.y0


@dataclass(frozen=True)
class EdgeLabelPlacement:
    edge_index: int
    x: float
    y: float
    rect: PlotRect
    angle_degrees: float = 0.0
    segment_index: int = -1
    segment_length: float = 0.0
    foreign_route_clearance: float = math.inf
    foreign_route_overlap_count: int = 0
    collision_score: int = 0


@dataclass(frozen=True)
class EdgeRoute:
    edge_index: int
    points: tuple[tuple[float, float], ...]
    crossing_nodes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeLabelCandidate:
    x: float
    y: float
    angle_degrees: float
    segment_index: int
    segment_length: float


@dataclass(frozen=True)
class AiPrototypePrompt:
    slug: str
    model_family: str
    prompt: str
    negative_prompt: str
    boundary_note: str


ROLE_COLORS: dict[str, tuple[str, str]] = {
    "input": ("#e8f3ff", "#1f77b4"),
    "state": ("#edf7ed", "#2ca02c"),
    "model": ("#fff3df", "#ff7f0e"),
    "control": ("#f2ecff", "#7f3fbf"),
    "evidence": ("#fff0f0", "#d62728"),
    "output": ("#ecfbfb", "#17becf"),
    "boundary": ("#f7f7f7", "#555555"),
    "default": ("#ffffff", "#333333"),
}

DRAWIO_SCALE = 170
DRAWIO_MARGIN = 100
DRAWIO_GRID = 10

ROLE_DRAWIO_STYLES: dict[str, str] = {
    "input": "rounded=1;whiteSpace=wrap;html=1;fillColor=#e8f3ff;strokeColor=#1f77b4;strokeWidth=2;fontSize=13;",
    "state": "rounded=1;whiteSpace=wrap;html=1;fillColor=#edf7ed;strokeColor=#2ca02c;strokeWidth=2;fontSize=13;",
    "model": "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff3df;strokeColor=#ff7f0e;strokeWidth=2;fontSize=13;",
    "control": "rounded=1;whiteSpace=wrap;html=1;fillColor=#f2ecff;strokeColor=#7f3fbf;strokeWidth=2;fontSize=13;",
    "evidence": "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff0f0;strokeColor=#d62728;strokeWidth=2;fontSize=13;",
    "output": "rounded=1;whiteSpace=wrap;html=1;fillColor=#ecfbfb;strokeColor=#17becf;strokeWidth=2;fontSize=13;",
    "boundary": "rounded=1;whiteSpace=wrap;html=1;fillColor=#f7f7f7;strokeColor=#555555;strokeWidth=2;fontSize=13;dashed=1;",
    "default": "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;strokeWidth=2;fontSize=13;",
}

DRAWIO_EDGE_STYLE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
    "html=1;strokeWidth=2;strokeColor=#4a4a4a;fontSize=12;"
    "labelBackgroundColor=#ffffff;labelBorderColor=#dddddd;"
    "labelPosition=center;verticalLabelPosition=middle;align=center;verticalAlign=middle;"
    "endArrow=block;endFill=1;"
)

RENDER_STATIC_EDGE_LABELS = False
LANE_LABEL_DIVIDER_X = 2.25
LANE_CONTENT_MIN_X = 2.72


def _strip_comment_prefix(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("%%"):
        return stripped[2:].strip()
    if stripped.startswith("//"):
        return stripped[2:].strip()
    return None


def _parse_node(payload: str) -> FigureNode:
    parts = [part.strip() for part in payload.split("|")]
    if len(parts) < 4:
        raise ValueError(f"Invalid node directive: {payload!r}")
    node_id, x_text, y_text, label = parts[:4]
    role = parts[4] if len(parts) > 4 and parts[4] else "default"
    return FigureNode(
        node_id=node_id,
        x=float(x_text),
        y=float(y_text),
        label=label.replace("\\n", "\n"),
        role=role,
    )


def _parse_edge(payload: str) -> FigureEdge:
    edge_text, _, label = payload.partition("|")
    source, sep, target = edge_text.partition("->")
    if not sep:
        raise ValueError(f"Invalid edge directive: {payload!r}")
    return FigureEdge(source=source.strip(), target=target.strip(), label=label.strip())


def _parse_lane(payload: str) -> FigureLane:
    parts = [part.strip() for part in payload.split("|")]
    if len(parts) < 4:
        raise ValueError(f"Invalid lane directive: {payload!r}")
    lane_id, y0_text, y1_text, label = parts[:4]
    role = parts[4] if len(parts) > 4 and parts[4] else "default"
    return FigureLane(
        lane_id=lane_id,
        y0=float(y0_text),
        y1=float(y1_text),
        label=label.replace("\\n", "\n"),
        role=role,
    )


def parse_figure_source(path: str | Path) -> FigureSpec:
    source_path = Path(path)
    spec = FigureSpec(slug=source_path.stem.replace("_", "-"))
    for line in source_path.read_text(encoding="utf-8").splitlines():
        payload = _strip_comment_prefix(line)
        if payload is None:
            continue
        if payload.startswith("title:"):
            spec.title = payload.split(":", 1)[1].strip()
        elif payload.startswith("subtitle:"):
            spec.subtitle = payload.split(":", 1)[1].strip()
        elif payload.startswith("size:"):
            width_text, _, height_text = payload.split(":", 1)[1].partition("|")
            spec.width = float(width_text.strip())
            spec.height = float(height_text.strip())
        elif payload.startswith("lane:"):
            spec.lanes.append(_parse_lane(payload.split(":", 1)[1].strip()))
        elif payload.startswith("node:"):
            node = _parse_node(payload.split(":", 1)[1].strip())
            spec.nodes[node.node_id] = node
        elif payload.startswith("edge:"):
            spec.edges.append(_parse_edge(payload.split(":", 1)[1].strip()))
    if not spec.nodes:
        raise ValueError(f"No figure node directives found in {source_path}")
    return spec


def _wrap_label(label: str, width: int = 22) -> str:
    lines: list[str] = []
    for part in label.splitlines():
        wrapped = textwrap.wrap(part, width=width) or [part]
        lines.extend(wrapped)
    return "\n".join(lines)


def _snap(value: float, grid: int = DRAWIO_GRID) -> int:
    return int(round(value / grid) * grid)


def _drawio_label(label: str) -> str:
    return "<br>".join(_wrap_label(label, width=28).splitlines())


def _plot_to_drawio_point(spec: FigureSpec, x: float, y: float) -> tuple[float, float]:
    return (
        DRAWIO_MARGIN + x * DRAWIO_SCALE,
        DRAWIO_MARGIN + (spec.height - y) * DRAWIO_SCALE,
    )


def _edge_tag(index: int) -> str:
    return f"E{index}"


def _edge_legend_lines(spec: FigureSpec) -> list[str]:
    return [
        f"{_edge_tag(index)}: {edge.label}"
        for index, edge in enumerate(spec.edges, start=1)
        if edge.label
    ]


def _legend_columns(lines: list[str], rows_per_column: int = 5) -> list[list[str]]:
    if not lines:
        return []
    return [lines[index : index + rows_per_column] for index in range(0, len(lines), rows_per_column)]


def _drawio_connection_legend_value(spec: FigureSpec) -> str:
    lines = _edge_legend_lines(spec)
    if not lines:
        return ""
    return "<b>Connection labels</b><br>" + "<br>".join(_wrap_label(line, width=54).replace("\n", " ") for line in lines)


def _static_connection_legend_text(spec: FigureSpec) -> str:
    lines = _edge_legend_lines(spec)
    columns = _legend_columns(lines)
    if not columns:
        return ""
    column_width = 54
    rows = max(len(column) for column in columns)
    rendered: list[str] = ["Connection labels"]
    for row_index in range(rows):
        row_parts = []
        for column in columns:
            value = column[row_index] if row_index < len(column) else ""
            row_parts.append(value.ljust(column_width))
        rendered.append("  ".join(row_parts).rstrip())
    return "\n".join(rendered)


def _prototype_node_list(spec: FigureSpec) -> str:
    return "\n".join(
        f"- {node.node_id}: {node.label.replace(chr(10), '; ')} [{node.role}]"
        for node in spec.nodes.values()
    )


def _prototype_lane_list(spec: FigureSpec) -> str:
    return "\n".join(
        f"- {lane.lane_id}: {lane.label} from y={lane.y0:g} to y={lane.y1:g} [{lane.role}]"
        for lane in spec.lanes
    )


def _prototype_edge_list(spec: FigureSpec) -> str:
    return "\n".join(
        f"- {edge.source} -> {edge.target}: {edge.label or 'unlabeled relation'}"
        for edge in spec.edges
    )


def build_ai_prototype_prompt(spec: FigureSpec, model_family: str = "gpt-image-2") -> AiPrototypePrompt:
    """Build a visual-design prompt without changing mechanism semantics.

    AI-generated images are treated as visual prototypes only. The authoritative
    figure remains the editable Mermaid/DOT/draw.io source generated from the
    same `FigureSpec`.
    """

    prompt = f"""Create a clean editorial architecture-diagram prototype for an academic paper.

Target model: {model_family}.
Figure title: {spec.title}.
Figure subtitle: {spec.subtitle or 'none'}.
Canvas: wide landscape, high resolution, white background, vector-like shapes.
Visual style: Nature/ACM paper-ready systems diagram, restrained color palette, high contrast, spacious grouping, no decorative sci-fi effects.

Layer/lane structure:
{_prototype_lane_list(spec) or '- no explicit lanes'}

Required nodes and roles:
{_prototype_node_list(spec)}

Required directed relations:
{_prototype_edge_list(spec)}

Design requirements:
- Preserve the exact mechanism boundary: bounded command selection only, not production autonomy or open-ended repair.
- Use grouped lanes or layered regions if useful: observation/receptors, latent state, routing/native heads, controls/ablations, claim boundary.
- Prefer semantic icons only when they clarify the node role; do not add extra system components.
- Use short callouts for relations, or edge-number tags with a compact legend, but keep all labels clearly separated from nodes and lines.
- Avoid overlapping labels, crossing-heavy wiring, and ambiguous label ownership.
- Leave enough whitespace for later Draw.io/vector recreation.
- The image is a visual prototype; final paper artwork will be manually/vector recreated from editable source files.
"""
    negative_prompt = (
        "No new architecture modules. No sealed-evaluation feedback loop. "
        "No candidate_0/candidate_1 markers, gold labels, hidden hints, shell autonomy, "
        "robot mascots, screenshots, code blocks, dense tiny text, or unverifiable performance claims."
    )
    boundary_note = (
        "AI raster output is a visual prototype only and is non-authoritative. "
        "Use it only to choose composition, visual hierarchy, and spacing before "
        "updating editable Mermaid/DOT/draw.io sources."
    )
    return AiPrototypePrompt(
        slug=spec.slug,
        model_family=model_family,
        prompt=prompt.strip() + "\n",
        negative_prompt=negative_prompt,
        boundary_note=boundary_note,
    )


def write_ai_prototype_prompts(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    model_family: str = "gpt-image-2",
) -> dict[str, object]:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    prompt_dir = output_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    prompts: dict[str, dict[str, str]] = {}
    for path in sorted([*source_path.glob("*.mmd"), *source_path.glob("*.dot")]):
        spec = parse_figure_source(path)
        prototype = build_ai_prototype_prompt(spec, model_family=model_family)
        prompt_path = prompt_dir / f"{spec.slug}.md"
        prompt_path.write_text(
            "\n".join(
                [
                    f"# AI Prototype Prompt: {spec.title}",
                    "",
                    f"- Source figure: `{path.as_posix()}`",
                    f"- Intended model family: `{prototype.model_family}`",
                    "- Status: visual prototype only; not final evidence artwork.",
                    "",
                    "## Prompt",
                    "",
                    prototype.prompt.rstrip(),
                    "",
                    "## Negative Prompt",
                    "",
                    prototype.negative_prompt,
                    "",
                    "## Boundary Note",
                    "",
                    prototype.boundary_note,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        prompts[path.name] = {
            "prompt_path": str(prompt_path),
            "slug": prototype.slug,
            "model_family": prototype.model_family,
        }

    manifest = {
        "ai_prototype_pipeline": "figure_spec_to_gpt_image_prompt_to_editable_vector_redraw",
        "authority": "Mermaid/DOT/draw.io source remains authoritative; AI image output is visual reference only.",
        "model_family": model_family,
        "prompt_dir": str(prompt_dir),
        "figures": prompts,
    }
    manifest_path = output_path / "ai_prototype_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _strip_svg_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _drawio_node_geometry(spec: FigureSpec, node: FigureNode) -> dict[str, int]:
    lines = _wrap_label(node.label, width=28).splitlines()
    max_line = max((len(line) for line in lines), default=1)
    width = _snap(min(max(220, max_line * 7.4 + 70), 380))
    height = _snap(min(max(80, len(lines) * 23 + 36), 160))
    center_x = DRAWIO_MARGIN + node.x * DRAWIO_SCALE
    center_y = DRAWIO_MARGIN + (spec.height - node.y) * DRAWIO_SCALE
    return {
        "x": _snap(center_x - width / 2),
        "y": _snap(center_y - height / 2),
        "width": int(width),
        "height": int(height),
    }


def _node_rects(spec: FigureSpec) -> dict[str, dict[str, int]]:
    return {node_id: _drawio_node_geometry(spec, node) for node_id, node in spec.nodes.items()}


def _drawio_lane_geometry(spec: FigureSpec, lane: FigureLane) -> dict[str, int]:
    x = 20
    y_top = DRAWIO_MARGIN + (spec.height - lane.y1) * DRAWIO_SCALE
    height = (lane.y1 - lane.y0) * DRAWIO_SCALE
    width = spec.width * DRAWIO_SCALE + DRAWIO_MARGIN * 2 - 40
    return {
        "x": _snap(x),
        "y": _snap(y_top),
        "width": _snap(width),
        "height": _snap(height),
    }


def _rect_overlap(left: dict[str, int], right: dict[str, int]) -> bool:
    return (
        left["x"] < right["x"] + right["width"]
        and left["x"] + left["width"] > right["x"]
        and left["y"] < right["y"] + right["height"]
        and left["y"] + left["height"] > right["y"]
    )


def _rect_gap(left: dict[str, int], right: dict[str, int]) -> float:
    dx = max(left["x"] - (right["x"] + right["width"]), right["x"] - (left["x"] + left["width"]), 0)
    dy = max(left["y"] - (right["y"] + right["height"]), right["y"] - (left["y"] + left["height"]), 0)
    return math.hypot(dx, dy)


def _render_node_rect(node: FigureNode) -> PlotRect:
    box_width = min(max(1.75, 0.118 * max(len(line) for line in node.label.splitlines())), 2.85)
    box_height = min(max(0.78, 0.27 * len(_wrap_label(node.label).splitlines()) + 0.36), 1.55)
    return PlotRect(
        node.x - box_width / 2,
        node.y - box_height / 2,
        node.x + box_width / 2,
        node.y + box_height / 2,
    )


def _point_inside_rect(point: tuple[float, float], rect: PlotRect) -> bool:
    x, y = point
    return rect.x0 <= x <= rect.x1 and rect.y0 <= y <= rect.y1


def _segment_intersects_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: PlotRect,
) -> bool:
    """Return true when a line segment crosses a rectangle.

    This uses Liang-Barsky clipping rather than a sampled approximation, so the
    audit catches the exact failure mode where a relation arrow visually passes
    through an unrelated card.
    """

    if _point_inside_rect(start, rect) or _point_inside_rect(end, rect):
        return True

    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    p_values = (-dx, dx, -dy, dy)
    q_values = (x0 - rect.x0, rect.x1 - x0, y0 - rect.y0, rect.y1 - y0)
    u0 = 0.0
    u1 = 1.0
    for p, q in zip(p_values, q_values):
        if abs(p) < 1e-9:
            if q < 0:
                return False
            continue
        ratio = q / p
        if p < 0:
            if ratio > u1:
                return False
            u0 = max(u0, ratio)
        else:
            if ratio < u0:
                return False
            u1 = min(u1, ratio)
    return u0 <= u1


def _segment_length(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _route_length(points: list[tuple[float, float]]) -> float:
    return sum(_segment_length(left, right) for left, right in zip(points, points[1:]))


def _route_crossing_nodes(
    *,
    points: list[tuple[float, float]],
    rects: dict[str, PlotRect],
    ignore_node_ids: set[str],
    margin: float = 0.08,
) -> list[str]:
    crossings: list[str] = []
    for node_id, rect in rects.items():
        if node_id in ignore_node_ids:
            continue
        expanded = rect.expanded(margin)
        if any(_segment_intersects_rect(start, end, expanded) for start, end in zip(points, points[1:])):
            crossings.append(node_id)
    return sorted(crossings)


def _axis_aligned_overlap_length(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
    *,
    tolerance: float = 0.05,
) -> float:
    first_horizontal = abs(first_start[1] - first_end[1]) <= tolerance
    second_horizontal = abs(second_start[1] - second_end[1]) <= tolerance
    first_vertical = abs(first_start[0] - first_end[0]) <= tolerance
    second_vertical = abs(second_start[0] - second_end[0]) <= tolerance
    if first_horizontal and second_horizontal and abs(first_start[1] - second_start[1]) <= tolerance:
        first_min, first_max = sorted((first_start[0], first_end[0]))
        second_min, second_max = sorted((second_start[0], second_end[0]))
        return max(0.0, min(first_max, second_max) - max(first_min, second_min))
    if first_vertical and second_vertical and abs(first_start[0] - second_start[0]) <= tolerance:
        first_min, first_max = sorted((first_start[1], first_end[1]))
        second_min, second_max = sorted((second_start[1], second_end[1]))
        return max(0.0, min(first_max, second_max) - max(first_min, second_min))
    return 0.0


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
    *,
    tolerance: float = 1e-8,
) -> bool:
    def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return (
            min(a[0], c[0]) - tolerance <= b[0] <= max(a[0], c[0]) + tolerance
            and min(a[1], c[1]) - tolerance <= b[1] <= max(a[1], c[1]) + tolerance
        )

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    if o1 * o2 < -tolerance and o3 * o4 < -tolerance:
        return True
    if abs(o1) <= tolerance and on_segment(first_start, second_start, first_end):
        return True
    if abs(o2) <= tolerance and on_segment(first_start, second_end, first_end):
        return True
    if abs(o3) <= tolerance and on_segment(second_start, first_start, second_end):
        return True
    if abs(o4) <= tolerance and on_segment(second_start, first_end, second_end):
        return True
    return False


def _edge_route_relation_penalty(
    candidate: list[tuple[float, float]],
    placed_routes: list[EdgeRoute],
) -> float:
    penalty = 0.0
    for start, end in zip(candidate, candidate[1:]):
        for route in placed_routes:
            for other_start, other_end in zip(route.points, route.points[1:]):
                shared_endpoint = any(
                    _segment_length(left, right) <= 0.05
                    for left in (start, end)
                    for right in (other_start, other_end)
                )
                overlap = _axis_aligned_overlap_length(start, end, other_start, other_end)
                if overlap > 0.12:
                    penalty += overlap * 90
                elif not shared_endpoint and _segments_intersect(start, end, other_start, other_end):
                    penalty += 22
    return penalty


def _edge_route_relation_conflicts(routes: list[EdgeRoute]) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    for left_index, left in enumerate(routes):
        for right in routes[left_index + 1 :]:
            for left_start, left_end in zip(left.points, left.points[1:]):
                for right_start, right_end in zip(right.points, right.points[1:]):
                    shared_endpoint = any(
                        _segment_length(a, b) <= 0.05
                        for a in (left_start, left_end)
                        for b in (right_start, right_end)
                    )
                    overlap = _axis_aligned_overlap_length(left_start, left_end, right_start, right_end)
                    if overlap > 0.18:
                        conflicts.append(
                            {
                                "kind": "overlap",
                                "edges": [_edge_tag(left.edge_index), _edge_tag(right.edge_index)],
                                "length": round(overlap, 3),
                            }
                        )
                    elif not shared_endpoint and _segments_intersect(left_start, left_end, right_start, right_end):
                        conflicts.append(
                            {
                                "kind": "crossing",
                                "edges": [_edge_tag(left.edge_index), _edge_tag(right.edge_index)],
                            }
                        )
    return conflicts


def _dedupe_adjacent_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or _segment_length(deduped[-1], point) > 0.02:
            deduped.append(point)
    return deduped


def _anchor_point_on_rect(
    *,
    rect: PlotRect,
    center: tuple[float, float],
    toward: tuple[float, float],
) -> tuple[float, float]:
    dx = toward[0] - center[0]
    dy = toward[1] - center[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return center

    candidates: list[tuple[float, float, float]] = []
    if abs(dx) > 1e-9:
        for x in (rect.x0, rect.x1):
            scale = (x - center[0]) / dx
            y = center[1] + dy * scale
            if scale >= 0 and rect.y0 - 1e-6 <= y <= rect.y1 + 1e-6:
                candidates.append((scale, x, y))
    if abs(dy) > 1e-9:
        for y in (rect.y0, rect.y1):
            scale = (y - center[1]) / dy
            x = center[0] + dx * scale
            if scale >= 0 and rect.x0 - 1e-6 <= x <= rect.x1 + 1e-6:
                candidates.append((scale, x, y))
    if not candidates:
        return center
    _, x, y = min(candidates, key=lambda item: item[0])
    return (x, y)


def _edge_route_candidates(spec: FigureSpec, edge: FigureEdge) -> list[list[tuple[float, float]]]:
    source = spec.nodes[edge.source]
    target = spec.nodes[edge.target]
    start = (source.x, source.y)
    end = (target.x, target.y)
    mid_x = (source.x + target.x) / 2
    mid_y = (source.y + target.y) / 2

    y_offsets = [mid_y, source.y, target.y]
    for offset in (0.85, 1.25, 1.7, 2.15):
        y_offsets.extend([min(spec.height - 0.45, max(0.35, mid_y + offset)), min(spec.height - 0.45, max(0.35, mid_y - offset))])

    x_offsets = [mid_x, source.x, target.x]
    for offset in (1.1, 1.7, 2.35, 3.0):
        x_offsets.extend([min(spec.width - 0.45, max(0.45, mid_x + offset)), min(spec.width - 0.45, max(0.45, mid_x - offset))])

    candidates: list[list[tuple[float, float]]] = [
        [start, end],
        [start, (target.x, source.y), end],
        [start, (source.x, target.y), end],
    ]
    for y in y_offsets:
        candidates.append([start, (source.x, y), (target.x, y), end])
    for x in x_offsets:
        candidates.append([start, (x, source.y), (x, target.y), end])
    return [_dedupe_adjacent_points(candidate) for candidate in candidates]


def _anchor_route_endpoints(spec: FigureSpec, edge: FigureEdge, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    rects = {node_id: _render_node_rect(node) for node_id, node in spec.nodes.items()}
    source = spec.nodes[edge.source]
    target = spec.nodes[edge.target]
    anchored = list(points)
    anchored[0] = _anchor_point_on_rect(
        rect=rects[edge.source],
        center=(source.x, source.y),
        toward=anchored[1],
    )
    anchored[-1] = _anchor_point_on_rect(
        rect=rects[edge.target],
        center=(target.x, target.y),
        toward=anchored[-2],
    )
    return _dedupe_adjacent_points(anchored)


def compute_edge_routes(spec: FigureSpec) -> list[EdgeRoute]:
    rects = {node_id: _render_node_rect(node) for node_id, node in spec.nodes.items()}
    routes: list[EdgeRoute] = []
    placed_routes: list[EdgeRoute] = []

    for index, edge in enumerate(spec.edges, start=1):
        ranked: list[tuple[float, list[tuple[float, float]], list[str]]] = []
        for candidate in _edge_route_candidates(spec, edge):
            anchored = _anchor_route_endpoints(spec, edge, candidate)
            crossings = _route_crossing_nodes(
                points=anchored,
                rects=rects,
                ignore_node_ids={edge.source, edge.target},
            )
            orthogonal_penalty = sum(
                0.35
                for start, end in zip(anchored, anchored[1:])
                if abs(start[0] - end[0]) > 0.05 and abs(start[1] - end[1]) > 0.05
            )
            relation_penalty = _edge_route_relation_penalty(anchored, placed_routes)
            score = len(crossings) * 1_000 + relation_penalty + len(anchored) * 2.0 + _route_length(anchored) + orthogonal_penalty
            ranked.append((score, anchored, crossings))
        ranked.sort(key=lambda item: item[0])
        _, best_points, best_crossings = ranked[0]
        route = EdgeRoute(index, tuple(best_points), tuple(best_crossings))
        routes.append(route)
        placed_routes.append(route)
    return routes


def _route_midpoint(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    if len(points) == 1:
        return points[0]
    total = _route_length(list(points))
    if total <= 1e-9:
        return points[0]
    remaining = total / 2
    for start, end in zip(points, points[1:]):
        length = _segment_length(start, end)
        if remaining <= length:
            ratio = remaining / length if length > 1e-9 else 0.0
            return (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio)
        remaining -= length
    return points[-1]


def _normalize_label_angle(dx: float, dy: float) -> float:
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def _distance_point_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - sx, py - sy)
    ratio = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    closest_x = sx + ratio * dx
    closest_y = sy + ratio * dy
    return math.hypot(px - closest_x, py - closest_y)


def _distance_point_to_route(point: tuple[float, float], route: EdgeRoute) -> float:
    return min(
        _distance_point_to_segment(point, start, end)
        for start, end in zip(route.points, route.points[1:])
    )


def _distance_rect_to_segment(rect: PlotRect, start: tuple[float, float], end: tuple[float, float]) -> float:
    if _segment_intersects_rect(start, end, rect):
        return 0.0
    corners = [
        (rect.x0, rect.y0),
        (rect.x0, rect.y1),
        (rect.x1, rect.y0),
        (rect.x1, rect.y1),
    ]
    corner_distance = min(_distance_point_to_segment(corner, start, end) for corner in corners)
    endpoint_distance = min(
        max(
            rect.x0 - point[0],
            point[0] - rect.x1,
            0.0,
        )
        if rect.y0 <= point[1] <= rect.y1
        else math.hypot(
            max(rect.x0 - point[0], 0.0, point[0] - rect.x1),
            max(rect.y0 - point[1], 0.0, point[1] - rect.y1),
        )
        for point in (start, end)
    )
    return min(corner_distance, endpoint_distance)


def _label_foreign_route_metrics(
    *,
    rect: PlotRect,
    point: tuple[float, float],
    edge_index: int,
    routes: dict[int, EdgeRoute],
) -> tuple[int, float]:
    overlap_count = 0
    min_clearance = math.inf
    expanded = rect.expanded(0.03)
    for other_edge_index, route in routes.items():
        if other_edge_index == edge_index:
            continue
        for start, end in zip(route.points, route.points[1:]):
            if _segment_intersects_rect(start, end, expanded):
                overlap_count += 1
            min_clearance = min(
                min_clearance,
                _distance_rect_to_segment(expanded, start, end),
                _distance_point_to_segment(point, start, end),
            )
    return overlap_count, min_clearance


def _edge_label_rect(x: float, y: float, text: str) -> PlotRect:
    width = max(0.44, 0.18 + len(text) * 0.14)
    height = 0.34
    return PlotRect(x - width / 2, y - height / 2, x + width / 2, y + height / 2)


def _candidate_edge_label_positions(
    *,
    edge: FigureEdge,
    spec: FigureSpec,
    route: EdgeRoute | None = None,
) -> list[EdgeLabelCandidate]:
    if route is not None and len(route.points) >= 2:
        positions: list[EdgeLabelCandidate] = []
        along_values = [0.50, 0.36, 0.64, 0.24, 0.76]
        for segment_index, (start, end) in enumerate(zip(route.points, route.points[1:])):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length < 0.22:
                continue
            angle = _normalize_label_angle(dx, dy)
            for along in along_values:
                x = start[0] + dx * along
                y = start[1] + dy * along
                if 0.25 <= x <= spec.width - 0.25 and 0.25 <= y <= spec.height - 0.25:
                    positions.append(
                        EdgeLabelCandidate(
                            x=x,
                            y=y,
                            angle_degrees=angle,
                            segment_index=segment_index,
                            segment_length=segment_length,
                        )
                    )
        if positions:
            return positions

    source = spec.nodes[edge.source]
    target = spec.nodes[edge.target]
    dx = target.x - source.x
    dy = target.y - source.y
    angle = _normalize_label_angle(dx, dy)
    segment_length = math.hypot(dx, dy)
    along_values = [0.50, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74]

    positions: list[EdgeLabelCandidate] = []
    for along in along_values:
        x = source.x + dx * along
        y = source.y + dy * along
        if 0.25 <= x <= spec.width - 0.25 and 0.25 <= y <= spec.height - 0.25:
            positions.append(
                EdgeLabelCandidate(
                    x=x,
                    y=y,
                    angle_degrees=angle,
                    segment_index=0,
                    segment_length=segment_length,
                )
            )
    return positions


def _edge_label_collision_score(
    *,
    rect: PlotRect,
    node_rects: list[PlotRect],
    placed_rects: list[PlotRect],
) -> int:
    score = 0
    for node_rect in node_rects:
        if rect.overlaps(node_rect.expanded(0.12)):
            score += 100
    for placed_rect in placed_rects:
        if rect.overlaps(placed_rect.expanded(0.10)):
            score += 50
    return score


def compute_edge_label_placements(spec: FigureSpec) -> list[EdgeLabelPlacement]:
    node_rects = [_render_node_rect(node) for node in spec.nodes.values()]
    placed_rects: list[PlotRect] = []
    placements: list[EdgeLabelPlacement] = []
    routes = {route.edge_index: route for route in compute_edge_routes(spec)}

    for index, edge in enumerate(spec.edges, start=1):
        if not edge.label:
            continue
        text = _edge_tag(index)
        route = routes.get(index)
        candidates = _candidate_edge_label_positions(edge=edge, spec=spec, route=route)
        if not candidates:
            source = spec.nodes[edge.source]
            target = spec.nodes[edge.target]
            fallback_dx = target.x - source.x
            fallback_dy = target.y - source.y
            candidates = [
                EdgeLabelCandidate(
                    x=(source.x + target.x) / 2,
                    y=(source.y + target.y) / 2,
                    angle_degrees=_normalize_label_angle(fallback_dx, fallback_dy),
                    segment_index=0,
                    segment_length=math.hypot(fallback_dx, fallback_dy),
                )
            ]

        ranked: list[tuple[int, float, float, float, int, EdgeLabelPlacement]] = []
        for candidate_index, candidate in enumerate(candidates):
            rect = _edge_label_rect(candidate.x, candidate.y, text)
            collision_score = _edge_label_collision_score(
                rect=rect,
                node_rects=node_rects,
                placed_rects=placed_rects,
            )
            foreign_overlap_count, foreign_clearance = _label_foreign_route_metrics(
                rect=rect,
                point=(candidate.x, candidate.y),
                edge_index=index,
                routes=routes,
            )
            clearance_penalty = 0 if foreign_clearance >= 0.18 else int(round((0.18 - foreign_clearance) * 300))
            score = collision_score + foreign_overlap_count * 200 + clearance_penalty
            # Stable tie-breaker keeps tags close to the routed edge midpoint,
            # not the raw source-target chord, so fan-in routes stay readable.
            midpoint = _route_midpoint(route.points) if route is not None else (
                (spec.nodes[edge.source].x + spec.nodes[edge.target].x) / 2,
                (spec.nodes[edge.source].y + spec.nodes[edge.target].y) / 2,
            )
            midpoint_distance = math.hypot(candidate.x - midpoint[0], candidate.y - midpoint[1])
            ranked.append(
                (
                    score,
                    0.0 if math.isinf(foreign_clearance) else -foreign_clearance,
                    midpoint_distance,
                    -candidate.segment_length,
                    candidate_index,
                    EdgeLabelPlacement(
                        edge_index=index,
                        x=candidate.x,
                        y=candidate.y,
                        rect=rect,
                        angle_degrees=candidate.angle_degrees,
                        segment_index=candidate.segment_index,
                        segment_length=candidate.segment_length,
                        foreign_route_clearance=foreign_clearance,
                        foreign_route_overlap_count=foreign_overlap_count,
                        collision_score=score,
                    ),
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        placement = ranked[0][5]
        placed_rects.append(placement.rect)
        placements.append(placement)
    return placements


def audit_static_label_layout(spec: FigureSpec) -> dict[str, object]:
    placements = compute_edge_label_placements(spec)
    node_rects = [_render_node_rect(node).expanded(0.12) for node in spec.nodes.values()]
    routes = {route.edge_index: route for route in compute_edge_routes(spec)}
    node_collisions: list[str] = []
    label_collisions: list[list[str]] = []
    off_route_labels: list[dict[str, object]] = []
    foreign_route_overlaps: list[dict[str, object]] = []
    low_clearance_labels: list[dict[str, object]] = []

    for placement in placements:
        if any(placement.rect.overlaps(node_rect) for node_rect in node_rects):
            node_collisions.append(_edge_tag(placement.edge_index))
        route = routes.get(placement.edge_index)
        if route is not None:
            distance = _distance_point_to_route((placement.x, placement.y), route)
            if distance > 0.035:
                off_route_labels.append(
                    {
                        "tag": _edge_tag(placement.edge_index),
                        "distance": round(distance, 4),
                    }
                )
        if placement.foreign_route_overlap_count > 0:
            foreign_route_overlaps.append(
                {
                    "tag": _edge_tag(placement.edge_index),
                    "overlap_count": placement.foreign_route_overlap_count,
                }
            )
        if placement.foreign_route_clearance < 0.18:
            low_clearance_labels.append(
                {
                    "tag": _edge_tag(placement.edge_index),
                    "clearance": round(placement.foreign_route_clearance, 4),
                }
            )

    for left_index, left in enumerate(placements):
        for right in placements[left_index + 1 :]:
            if left.rect.overlaps(right.rect.expanded(0.10)):
                label_collisions.append([_edge_tag(left.edge_index), _edge_tag(right.edge_index)])

    return {
        "passed": (
            not node_collisions
            and not label_collisions
            and not off_route_labels
            and not foreign_route_overlaps
            and not low_clearance_labels
        ),
        "label_count": len(placements),
        "node_label_collisions": node_collisions,
        "label_label_collisions": label_collisions,
        "off_route_labels": off_route_labels,
        "foreign_route_overlaps": foreign_route_overlaps,
        "low_clearance_labels": low_clearance_labels,
        "placements": [
            {
                "tag": _edge_tag(placement.edge_index),
                "x": round(placement.x, 3),
                "y": round(placement.y, 3),
                "angle_degrees": round(placement.angle_degrees, 2),
                "segment_index": placement.segment_index,
                "segment_length": round(placement.segment_length, 3),
                "foreign_route_clearance": None
                if math.isinf(placement.foreign_route_clearance)
                else round(placement.foreign_route_clearance, 4),
                "foreign_route_overlap_count": placement.foreign_route_overlap_count,
                "collision_score": placement.collision_score,
            }
            for placement in placements
        ],
    }


def _lane_title_intrusions(spec: FigureSpec) -> list[dict[str, object]]:
    if not spec.lanes:
        return []
    intrusions: list[dict[str, object]] = []
    for node_id, node in spec.nodes.items():
        rect = _render_node_rect(node)
        if rect.x0 < LANE_CONTENT_MIN_X:
            intrusions.append(
                {
                    "node": node_id,
                    "left_x": round(rect.x0, 3),
                    "required_min_x": LANE_CONTENT_MIN_X,
                }
            )
    return intrusions


def _unconnected_nodes(spec: FigureSpec) -> list[str]:
    connected_ids = {edge.source for edge in spec.edges} | {edge.target for edge in spec.edges}
    return sorted(set(spec.nodes) - connected_ids)


def write_drawio(spec: FigureSpec, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / f"{spec.slug}.drawio"

    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "agent": "reflexlm-paper-figures;drawio-skill-compatible",
            "version": "26.0.0",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": spec.slug, "name": spec.title or spec.slug})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(_snap(spec.width * DRAWIO_SCALE + DRAWIO_MARGIN * 2)),
            "dy": str(_snap(spec.height * DRAWIO_SCALE + DRAWIO_MARGIN * 2)),
            "grid": "1",
            "gridSize": str(DRAWIO_GRID),
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(_snap(spec.width * DRAWIO_SCALE + DRAWIO_MARGIN * 2)),
            "pageHeight": str(_snap(spec.height * DRAWIO_SCALE + DRAWIO_MARGIN * 2)),
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    if spec.subtitle:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": "figure_subtitle",
                "value": _drawio_label(spec.subtitle),
                "style": (
                    "text;html=1;strokeColor=none;fillColor=none;fontSize=14;"
                    "fontColor=#333333;align=center;verticalAlign=middle;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(DRAWIO_MARGIN),
                "y": "40",
                "width": str(_snap(spec.width * DRAWIO_SCALE)),
                "height": "40",
                "as": "geometry",
            },
        )

    for lane in spec.lanes:
        face, edge_color = ROLE_COLORS.get(lane.role, ROLE_COLORS["default"])
        geometry = _drawio_lane_geometry(spec, lane)
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"lane_{lane.lane_id}",
                "value": _drawio_label(lane.label),
                "style": (
                    "rounded=0;whiteSpace=wrap;html=1;"
                    f"fillColor={face};strokeColor={edge_color};opacity=18;"
                    "dashed=1;strokeWidth=1;fontSize=14;fontStyle=1;"
                    "align=left;verticalAlign=middle;spacingLeft=16;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(geometry["x"]),
                "y": str(geometry["y"]),
                "width": str(geometry["width"]),
                "height": str(geometry["height"]),
                "as": "geometry",
            },
        )

    for node_id, node in spec.nodes.items():
        geometry = _drawio_node_geometry(spec, node)
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": node_id,
                "value": _drawio_label(node.label),
                "style": ROLE_DRAWIO_STYLES.get(node.role, ROLE_DRAWIO_STYLES["default"]),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(geometry["x"]),
                "y": str(geometry["y"]),
                "width": str(geometry["width"]),
                "height": str(geometry["height"]),
                "as": "geometry",
            },
        )

    placements = {placement.edge_index: placement for placement in compute_edge_label_placements(spec)}
    routes = {route.edge_index: route for route in compute_edge_routes(spec)}
    for index, edge in enumerate(spec.edges, start=1):
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"edge_{index}_{edge.source}_{edge.target}",
                "value": _edge_tag(index) if edge.label else "",
                "style": DRAWIO_EDGE_STYLE,
                "edge": "1",
                "parent": "1",
                "source": edge.source,
                "target": edge.target,
            },
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "x": "0", "y": "0", "as": "geometry"})
        route = routes.get(index)
        if route is not None and len(route.points) > 2:
            waypoints = ET.SubElement(geometry, "Array", {"as": "points"})
            for point in route.points[1:-1]:
                waypoint_x, waypoint_y = _plot_to_drawio_point(spec, point[0], point[1])
                ET.SubElement(
                    waypoints,
                    "mxPoint",
                    {
                        "x": str(_snap(waypoint_x)),
                        "y": str(_snap(waypoint_y)),
                    },
                )
        placement = placements.get(index)
        if placement is not None:
            source_node = spec.nodes[edge.source]
            target_node = spec.nodes[edge.target]
            midpoint_x, midpoint_y = _plot_to_drawio_point(
                spec,
                (source_node.x + target_node.x) / 2,
                (source_node.y + target_node.y) / 2,
            )
            label_x, label_y = _plot_to_drawio_point(spec, placement.x, placement.y)
            ET.SubElement(
                geometry,
                "mxPoint",
                {
                    "x": str(_snap(label_x - midpoint_x)),
                    "y": str(_snap(label_y - midpoint_y)),
                    "as": "offset",
                },
            )

    legend_value = _drawio_connection_legend_value(spec)
    if legend_value:
        legend_height = _snap(60 + len(_edge_legend_lines(spec)) * 24)
        legend_width = _snap(min(max(760, spec.width * DRAWIO_SCALE * 0.72), 1680))
        legend_x = DRAWIO_MARGIN
        legend_y = _snap(spec.height * DRAWIO_SCALE + DRAWIO_MARGIN * 2 - legend_height - 30)
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": "connection_label_legend",
                "value": legend_value,
                "style": (
                    "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;"
                    "strokeColor=#777777;strokeWidth=1;fontSize=12;align=left;"
                    "spacingLeft=10;spacingRight=10;spacingTop=8;spacingBottom=8;"
                ),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(_snap(legend_x)),
                "y": str(legend_y),
                "width": str(legend_width),
                "height": str(legend_height),
                "as": "geometry",
            },
        )

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return target


def audit_drawio_layout(spec: FigureSpec) -> dict[str, object]:
    rects = _node_rects(spec)
    overlap_pairs: list[list[str]] = []
    min_gap: float | None = None
    node_ids = sorted(rects)
    for left_index, left_id in enumerate(node_ids):
        for right_id in node_ids[left_index + 1 :]:
            left = rects[left_id]
            right = rects[right_id]
            if _rect_overlap(left, right):
                overlap_pairs.append([left_id, right_id])
            else:
                gap = _rect_gap(left, right)
                min_gap = gap if min_gap is None else min(min_gap, gap)

    referenced_ids = {edge.source for edge in spec.edges} | {edge.target for edge in spec.edges}
    missing_ids = sorted(referenced_ids - set(spec.nodes))
    grid_aligned = all(
        value % DRAWIO_GRID == 0
        for geometry in rects.values()
        for value in (geometry["x"], geometry["y"], geometry["width"], geometry["height"])
    )
    lanes_valid = all(lane.y0 < lane.y1 and 0 <= lane.y0 <= spec.height and 0 <= lane.y1 <= spec.height for lane in spec.lanes)
    label_audit = audit_static_label_layout(spec)
    lane_title_intrusions = _lane_title_intrusions(spec)
    unconnected_nodes = _unconnected_nodes(spec)
    edge_routes = compute_edge_routes(spec)
    edge_node_crossings = [
        {
            "edge": _edge_tag(route.edge_index),
            "crossing_nodes": list(route.crossing_nodes),
        }
        for route in edge_routes
        if route.crossing_nodes
    ]
    edge_route_conflicts = _edge_route_relation_conflicts(edge_routes)
    return {
        "passed": (
            not overlap_pairs
            and not missing_ids
            and grid_aligned
            and lanes_valid
            and not lane_title_intrusions
            and not unconnected_nodes
            and not edge_node_crossings
            and not edge_route_conflicts
        ),
        "node_count": len(spec.nodes),
        "edge_count": len(spec.edges),
        "lane_count": len(spec.lanes),
        "missing_edge_node_ids": missing_ids,
        "overlap_pairs": overlap_pairs,
        "edge_node_crossings": edge_node_crossings,
        "edge_route_conflicts": edge_route_conflicts,
        "grid_aligned": grid_aligned,
        "lanes_valid": lanes_valid,
        "lane_title_intrusions": lane_title_intrusions,
        "unconnected_nodes": unconnected_nodes,
        "min_node_gap_px": None if min_gap is None else round(min_gap, 2),
        "static_label_layout": label_audit,
    }


def render_figure(spec: FigureSpec, output_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(spec.width, spec.height))
    ax.set_title(spec.title, fontsize=16, weight="bold", pad=24 if spec.subtitle else 14)
    if spec.subtitle:
        ax.text(
            spec.width / 2,
            spec.height + 0.42,
            spec.subtitle,
            ha="center",
            va="center",
            fontsize=10.4,
            weight="normal",
            color="#303846",
            zorder=6,
        )
    ax.set_xlim(0, spec.width)
    ax.set_ylim(-1.35, spec.height + (0.62 if spec.subtitle else 0.15))
    ax.axis("off")

    placements = {placement.edge_index: placement for placement in compute_edge_label_placements(spec)}
    routes = {route.edge_index: route for route in compute_edge_routes(spec)}

    for lane in spec.lanes:
        face, edge_color = ROLE_COLORS.get(lane.role, ROLE_COLORS["default"])
        ax.axhspan(lane.y0, lane.y1, facecolor=face, edgecolor=edge_color, linewidth=0.9, alpha=0.16, linestyle="--", zorder=0)
        ax.text(
            0.18,
            (lane.y0 + lane.y1) / 2,
            _wrap_label(lane.label, width=18),
            ha="left",
            va="center",
            fontsize=8.6,
            weight="bold",
            color=edge_color,
            zorder=0.5,
        )
    if spec.lanes:
        ax.axvline(LANE_LABEL_DIVIDER_X, ymin=0.11, ymax=0.97, color="#b8c5d6", linewidth=0.8, zorder=0.6)

    for edge_index, edge in enumerate(spec.edges):
        source = spec.nodes[edge.source]
        target = spec.nodes[edge.target]
        route = routes[edge_index + 1]
        path = MplPath(
            route.points,
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(route.points) - 1),
        )
        arrow = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color="#4a4a4a",
            zorder=1,
        )
        ax.add_patch(arrow)
        if edge.label:
            placement = placements[edge_index + 1]
            ax.text(
                placement.x,
                placement.y,
                _edge_tag(edge_index + 1) if not RENDER_STATIC_EDGE_LABELS else _wrap_label(edge.label, width=20),
                ha="center",
                va="center",
                fontsize=8.4,
                color="#333333",
                rotation=placement.angle_degrees,
                rotation_mode="anchor",
                bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#dddddd"},
                zorder=2,
            )

    for node in spec.nodes.values():
        face, edge_color = ROLE_COLORS.get(node.role, ROLE_COLORS["default"])
        box_width = min(max(1.75, 0.118 * max(len(line) for line in node.label.splitlines())), 2.85)
        box_height = min(max(0.78, 0.27 * len(_wrap_label(node.label).splitlines()) + 0.36), 1.55)
        patch = FancyBboxPatch(
            (node.x - box_width / 2, node.y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.7,
            facecolor=face,
            edgecolor=edge_color,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            node.x,
            node.y,
            _wrap_label(node.label),
            ha="center",
            va="center",
            fontsize=9.2,
            color="#111111",
            zorder=4,
        )

    legend_text = _static_connection_legend_text(spec)
    if legend_text:
        ax.text(
            0.16,
            -1.3,
            legend_text,
            ha="left",
            va="bottom",
            fontsize=7.8,
            family="monospace",
            color="#202020",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#ffffff", "ec": "#999999"},
            zorder=5,
        )

    outputs = [
        output_path / f"{spec.slug}.svg",
        output_path / f"{spec.slug}.pdf",
        output_path / f"{spec.slug}.png",
    ]
    for target in outputs:
        fig.savefig(target, bbox_inches="tight", dpi=220)
        if target.suffix == ".svg":
            _strip_svg_trailing_whitespace(target)
    plt.close(fig)
    return outputs


def render_figure_sources(
    source_dir: str | Path,
    export_dir: str | Path,
    paper_figures_dir: str | Path | None = None,
    drawio_dir: str | Path | None = None,
    paper_drawio_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    source_path = Path(source_dir)
    exported: dict[str, list[str]] = {}
    for path in sorted([*source_path.glob("*.mmd"), *source_path.glob("*.dot")]):
        spec = parse_figure_source(path)
        outputs = render_figure(spec, export_dir)
        if drawio_dir is not None:
            outputs.append(write_drawio(spec, drawio_dir))
        exported[path.name] = [str(output) for output in outputs]
        if paper_figures_dir is not None:
            paper_path = Path(paper_figures_dir)
            paper_path.mkdir(parents=True, exist_ok=True)
            for output in outputs:
                if output.suffix != ".drawio":
                    shutil.copy2(output, paper_path / output.name)
        if paper_drawio_dir is not None and drawio_dir is not None:
            paper_drawio_path = Path(paper_drawio_dir)
            paper_drawio_path.mkdir(parents=True, exist_ok=True)
            for output in outputs:
                if output.suffix == ".drawio":
                    shutil.copy2(output, paper_drawio_path / output.name)
    return exported
