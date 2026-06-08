"""
CaneCestry Dash/Flask app

Fixed version generated from the uploaded app:
- DAG/tree output is SVG instead of stretched PNG, so variety labels stay crisp.
- Matrix generation is one page with one Generate button and one pedigree-expansion slider.
- Removed orphaned subset-pedigree page workflow and dummy/temp matrix callbacks.
- Removed duplicate data-store and duplicate /download route.
- Fixed uploaded pedigree persistence path: pedigree/user_input/pedigree.txt.
- Add-pedigree upload now appends valid rows and recalculates the active filtered dataframe.
- Family-tree male/female lineage highlighting now combines styles instead of overwriting them.
- Added matrix PCA/MDS ordination, group/era insights, founder-origin summaries, ancestor contribution plots, Cytoscape interactive viewer, a pedigree fan-chart generation ring layout, direct-line ancestor lookup, high-resolution visualization exports, fan-chart sex-colored labels, and flippable family-tree orientation.
"""

from __future__ import annotations

import base64
import os
import re
import time
import urllib.parse
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import dash
import dash_bootstrap_components as dbc
import flask
import graphviz
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from flask import Flask, send_file
from matplotlib.colors import Normalize
from numba import njit

try:
    import dash_cytoscape as cyto
    HAS_CYTOSCAPE = True
except Exception:
    cyto = None
    HAS_CYTOSCAPE = False

# =============================================================================
# Configuration
# =============================================================================

MAX_CLUSTER_SIZE = 300
REQUIRED_PEDIGREE_COLUMNS = ["LineName", "FemaleParent", "MaleParent"]
PATTERN_POLY_P = re.compile(r"^\d*[pP]\d*$")

BASE_DIR = Path(__file__).resolve().parent
PEDIGREE_DIR = BASE_DIR / "pedigree"
USER_INPUT_DIR = PEDIGREE_DIR / "user_input"
USER_INPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FILE = PEDIGREE_DIR / "Pedigree_Subset.txt"
USER_FILE = USER_INPUT_DIR / "pedigree.txt"

OUTPUT_DIR = BASE_DIR / "matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

server = Flask(__name__)
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SANDSTONE,
        "https://use.fontawesome.com/releases/v5.10.2/css/all.css",
    ],
    server=server,
    suppress_callback_exceptions=True,
)

CUSTOM_CSS = {
    "container": {"padding": "20px", "marginTop": "20px", "backgroundColor": "#f9f9f9"},
    "header": {
        "textAlign": "center",
        "padding": "10px",
        "color": "#2c3e50",
        "fontFamily": "Arial",
        "fontWeight": "bold",
        "fontSize": "50px",
    },
    "button": {
        "margin": "10px",
        "fontWeight": "bold",
        "backgroundColor": "#2980b9",
        "color": "white",
    },
    "table_wrap": {
        "overflowX": "auto",
        "marginBottom": "20px",
        "border": "1px solid #ccc",
        "maxHeight": "500px",
        "overflowY": "auto",
    },
    "tree_image": {"width": "100%", "height": "auto", "padding": "10px"},
    "dropdown": {"fontWeight": "bold", "color": "#2980b9"},
    "round_button": {
        "borderRadius": "50%",
        "width": "85px",
        "height": "85px",
        "fontSize": "42px",
        "textAlign": "center",
        "lineHeight": "65px",
        "backgroundColor": "#2980b9",
        "color": "white",
        "border": "none",
    },
    "left_button_group": {
        "display": "flex",
        "alignItems": "center",
        "gap": "10px",
        "marginTop": "0px",
    },
    "right_button_group": {
        "display": "flex",
        "justifyContent": "flex-end",
        "alignItems": "center",
        "gap": "10px",
        "marginTop": "0px",
    },
}

HIGHRES_GRAPH_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "canecestry_visualization_highres",
        "height": 2200,
        "width": 2200,
        "scale": 3,
    },
}

DOWNLOAD_BUTTON_STYLE = {
    "margin": "6px",
    "fontWeight": "bold",
}

# =============================================================================
# Data loading and active dataframe management
# =============================================================================


def clean_parent_value(value) -> Optional[str]:
    """Convert blank/placeholder parent values to None and strip valid names."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.lower() in {"", ".", "0", "na", "nan", "none", "unknown"}:
        return None
    return value


def normalize_pedigree_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a pedigree dataframe."""
    missing = [col for col in REQUIRED_PEDIGREE_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(
            "Pedigree file must contain columns: "
            + ", ".join(REQUIRED_PEDIGREE_COLUMNS)
            + f". Missing: {', '.join(missing)}"
        )

    out = raw_df[REQUIRED_PEDIGREE_COLUMNS].copy()
    out["LineName"] = out["LineName"].astype(str).str.strip()
    out = out[out["LineName"].ne("")].copy()
    out = out.drop_duplicates(subset=["LineName"], keep="first")

    for col in ["FemaleParent", "MaleParent"]:
        out[col] = out[col].map(lambda x: clean_parent_value(x) or "")

    return out.reset_index(drop=True)


def read_pedigree_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pedigree file not found: {path}")
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raw = pd.read_csv(path, sep="\t")
    return normalize_pedigree_df(raw)


def load_current_pedigree() -> pd.DataFrame:
    if USER_FILE.exists():
        print(f"Loading USER pedigree: {USER_FILE}")
        return read_pedigree_file(USER_FILE)
    print(f"Loading DEFAULT pedigree: {DEFAULT_FILE}")
    return read_pedigree_file(DEFAULT_FILE)


def compute_filtered_df(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows that are informative for pedigree functions.

    This preserves all entries that either have at least one parent or are used as a
    parent elsewhere. Completely isolated founder-like rows are dropped from tree
    dropdowns to avoid clutter, matching the behavior of the original app.
    """
    if source_df.empty:
        return source_df.copy()

    parent_values = pd.concat(
        [source_df["MaleParent"], source_df["FemaleParent"]], ignore_index=True
    ).map(clean_parent_value)
    parents_set = set(parent_values.dropna().tolist())

    has_no_parents = source_df["MaleParent"].map(clean_parent_value).isna() & source_df[
        "FemaleParent"
    ].map(clean_parent_value).isna()
    is_not_used_as_parent = ~source_df["LineName"].isin(parents_set)
    isolated_mask = has_no_parents & is_not_used_as_parent
    return source_df.loc[~isolated_mask].reset_index(drop=True)


try:
    default_df = read_pedigree_file(DEFAULT_FILE)
except Exception as exc:
    print(f"WARNING: Could not load default pedigree: {exc}")
    default_df = pd.DataFrame(columns=REQUIRED_PEDIGREE_COLUMNS)

try:
    df = load_current_pedigree()
except Exception as exc:
    print(f"WARNING: Could not load current pedigree: {exc}")
    df = default_df.copy()

filtered_df = compute_filtered_df(df)


def set_active_dataframe(new_df: pd.DataFrame, persist: bool = False) -> str:
    """Set global df/filtered_df and optionally save as the user pedigree file."""
    global df, filtered_df
    df = normalize_pedigree_df(new_df)
    filtered_df = compute_filtered_df(df)

    if persist:
        USER_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(USER_FILE, sep="\t", index=False)

    return f"Active pedigree now has {len(df):,} rows; {len(filtered_df):,} rows are shown in tree/search tools."


def line_options(source_df: Optional[pd.DataFrame] = None) -> list[dict[str, str]]:
    source_df = filtered_df if source_df is None else source_df
    names = sorted(source_df["LineName"].dropna().astype(str).unique().tolist())
    return [{"label": name, "value": name} for name in names]

# =============================================================================
# Graph and pedigree helpers
# =============================================================================


def safe_file_stem(value: str, fallback: str = "canecestry_export") -> str:
    """Make a short filesystem-safe stem for exported visualizations."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip("_")
    return value[:90] or fallback


def download_href(path: Path | str, file_type: str = "download") -> str:
    return f"/download?filename={urllib.parse.quote(str(path))}&type={file_type}"


def graphviz_to_svg_img(dot: graphviz.Digraph, max_height: str = "900px") -> html.Img:
    """Return a crisp SVG Graphviz image for Dash."""
    dot.attr("graph", bgcolor="transparent")
    svg_bytes = dot.pipe(format="svg")
    encoded_svg = base64.b64encode(svg_bytes).decode("utf-8")
    return html.Img(
        src=f"data:image/svg+xml;base64,{encoded_svg}",
        style={**CUSTOM_CSS["tree_image"], "maxHeight": max_height, "objectFit": "contain"},
    )


def graphviz_visualization_block(dot: graphviz.Digraph, stem: str, max_height: str = "900px") -> html.Div:
    """Render a tree/DAG plus high-resolution downloads.

    SVG/PDF are vector exports and are the best choices for huge pedigrees.
    The PNG export is rendered at high DPI for quick presentation use.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = safe_file_stem(stem)
    timestamp = int(time.time() * 1000)

    dot.attr("graph", bgcolor="transparent")
    svg_bytes = dot.pipe(format="svg")
    svg_path = OUTPUT_DIR / f"{stem}_{timestamp}.svg"
    svg_path.write_bytes(svg_bytes)

    export_links = [
        html.A("Download editable SVG", href=download_href(svg_path, "svg"), download=svg_path.name, className="btn btn-success", style=DOWNLOAD_BUTTON_STYLE),
    ]

    try:
        pdf_path = OUTPUT_DIR / f"{stem}_{timestamp}.pdf"
        pdf_path.write_bytes(dot.pipe(format="pdf"))
        export_links.append(html.A("Download PDF", href=download_href(pdf_path, "pdf"), download=pdf_path.name, className="btn btn-secondary", style=DOWNLOAD_BUTTON_STYLE))
    except Exception as exc:
        export_links.append(html.Span(f"PDF export unavailable: {exc}", style={"marginLeft": "8px", "color": "#777"}))

    try:
        # DPI affects raster PNG output only. SVG/PDF above stay vector/crisp regardless of zoom.
        dot.attr("graph", dpi="500")
        png_path = OUTPUT_DIR / f"{stem}_{timestamp}_large.png"
        png_path.write_bytes(dot.pipe(format="png"))
        export_links.append(html.A("Download large PNG", href=download_href(png_path, "png_download"), download=png_path.name, className="btn btn-warning", style=DOWNLOAD_BUTTON_STYLE))
    except Exception as exc:
        export_links.append(html.Span(f"PNG export unavailable: {exc}", style={"marginLeft": "8px", "color": "#777"}))

    encoded_svg = base64.b64encode(svg_bytes).decode("utf-8")
    preview = html.Img(
        src=f"data:image/svg+xml;base64,{encoded_svg}",
        style={**CUSTOM_CSS["tree_image"], "maxHeight": max_height, "objectFit": "contain"},
    )
    return html.Div([html.Div(export_links, style={"marginTop": "8px", "marginBottom": "8px"}), preview])


def save_plotly_html(fig: go.Figure, stem: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{safe_file_stem(stem)}_{int(time.time() * 1000)}.html"
    fig.write_html(str(path), include_plotlyjs=True, config=HIGHRES_GRAPH_CONFIG, full_html=True)
    return path


def normalize_style(*parts: str) -> str:
    bits: list[str] = []
    for part in parts:
        if not part:
            continue
        for piece in str(part).split(","):
            piece = piece.strip()
            if piece and piece not in bits:
                bits.append(piece)
    return ",".join(bits) if bits else "filled"


def build_parent_lookup(source_df: pd.DataFrame) -> dict[str, tuple[Optional[str], Optional[str]]]:
    lookup: dict[str, tuple[Optional[str], Optional[str]]] = {}
    for _, row in source_df.iterrows():
        lookup[str(row["LineName"])] = (
            clean_parent_value(row.get("MaleParent")),
            clean_parent_value(row.get("FemaleParent")),
        )
    return lookup


def build_child_lookup(source_df: pd.DataFrame) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for _, row in source_df.iterrows():
        child = str(row["LineName"])
        for parent in (clean_parent_value(row.get("MaleParent")), clean_parent_value(row.get("FemaleParent"))):
            if parent:
                children[parent].append(child)
    return children


def find_ancestors(line_name: str, source_df: pd.DataFrame):
    """Return ancestors, parent-child relationships, and generation bins."""
    parent_lookup = build_parent_lookup(source_df)
    ancestors: set[str] = set()
    relationships: list[tuple[str, str, str]] = []
    generations: dict[int, list[str]] = defaultdict(list)
    seen_edges: set[tuple[str, str, str]] = set()
    seen_nodes_at_depth: set[tuple[str, int]] = set()

    queue = deque([(line_name, 0)])
    while queue:
        child, depth = queue.popleft()
        if (child, depth) not in seen_nodes_at_depth:
            generations[depth].append(child)
            seen_nodes_at_depth.add((child, depth))

        if child not in parent_lookup:
            continue

        male_parent, female_parent = parent_lookup[child]
        for parent, role in [(male_parent, "male"), (female_parent, "female")]:
            if not parent:
                continue
            edge = (parent, child, role)
            if edge not in seen_edges:
                relationships.append(edge)
                seen_edges.add(edge)
            if parent not in ancestors:
                ancestors.add(parent)
                queue.append((parent, depth + 1))

    return ancestors, relationships, generations


def find_descendants(line_name: str, source_df: pd.DataFrame):
    """Return descendants, parent-child relationships, and generation bins."""
    child_lookup = build_child_lookup(source_df)
    parent_lookup = build_parent_lookup(source_df)
    descendants: set[str] = set()
    relationships: list[tuple[str, str, str]] = []
    generations: dict[int, list[str]] = defaultdict(list)
    seen_edges: set[tuple[str, str, str]] = set()

    queue = deque([(line_name, 0)])
    visited = {line_name}
    while queue:
        parent, depth = queue.popleft()
        generations[depth].append(parent)
        for child in child_lookup.get(parent, []):
            male_parent, female_parent = parent_lookup.get(child, (None, None))
            role = "male" if male_parent == parent else "female" if female_parent == parent else "descendant"
            edge = (parent, child, role)
            if edge not in seen_edges:
                relationships.append(edge)
                seen_edges.add(edge)
            if child not in visited:
                visited.add(child)
                descendants.add(child)
                queue.append((child, depth + 1))

    return descendants, relationships, generations


def get_direct_line(line_name: str, source_df: pd.DataFrame, parent_col: str) -> set[str]:
    """Follow the same-sex parental chain for lineage highlighting."""
    lookup = source_df.set_index("LineName", drop=False)
    current = line_name
    collected: set[str] = set()
    while current in lookup.index:
        row = lookup.loc[current]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        parent = clean_parent_value(row.get(parent_col))
        if not parent or parent in collected:
            break
        collected.add(parent)
        current = parent
    return collected


def parse_user_genotype_list(text: str) -> list[str]:
    """Parse a pasted genotype list separated by newlines, commas, semicolons, or tabs."""
    if not text:
        return []
    seen: set[str] = set()
    genotypes: list[str] = []
    for chunk in re.split(r"[,;\n\t]+", str(text)):
        name = chunk.strip()
        if name and name not in seen:
            seen.add(name)
            genotypes.append(name)
    return genotypes


def trace_direct_ancestor_chain(line_name: str, source_df: pd.DataFrame, parent_col: str) -> dict[str, object]:
    """
    Follow only one direct parental side until it terminates.

    For parent_col='MaleParent', this gives sire -> sire's sire -> sire's sire's sire, etc.
    For parent_col='FemaleParent', this gives dam -> dam's dam -> dam's dam's dam, etc.
    """
    lookup = source_df.set_index("LineName", drop=False)
    current = str(line_name).strip()
    path = [current]
    visited = {current}

    if current not in lookup.index:
        return {
            "FurthestAncestor": "",
            "Depth": 0,
            "Path": current,
            "Status": "Input genotype not found in pedigree",
        }

    status = "Reached recorded founder / missing parent"
    while True:
        row = lookup.loc[current]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        parent = clean_parent_value(row.get(parent_col))
        if not parent:
            status = "Reached recorded founder / missing parent"
            break
        if parent in visited:
            path.append(parent)
            status = "Stopped at cycle"
            break

        path.append(parent)
        visited.add(parent)
        if parent not in lookup.index:
            status = "Stopped at referenced ancestor not present as a LineName row"
            break
        current = parent

    if len(path) == 1:
        furthest = ""
        depth = 0
    else:
        furthest = path[-1]
        depth = len(path) - 1

    return {
        "FurthestAncestor": furthest,
        "Depth": depth,
        "Path": " → ".join(path),
        "Status": status,
    }


def direct_line_ancestor_table(genotypes: list[str], source_df: pd.DataFrame) -> pd.DataFrame:
    """Build a table of furthest direct male and female ancestors for pasted genotypes."""
    available = set(source_df["LineName"].astype(str).tolist())
    rows = []
    for genotype in genotypes:
        male = trace_direct_ancestor_chain(genotype, source_df, "MaleParent")
        female = trace_direct_ancestor_chain(genotype, source_df, "FemaleParent")
        rows.append(
            {
                "Genotype": genotype,
                "FoundInPedigree": genotype in available,
                "FurthestDirectMaleAncestor": male["FurthestAncestor"],
                "DirectMaleDepth": male["Depth"],
                "DirectMalePath": male["Path"],
                "DirectMaleStatus": male["Status"],
                "FurthestDirectFemaleAncestor": female["FurthestAncestor"],
                "DirectFemaleDepth": female["Depth"],
                "DirectFemalePath": female["Path"],
                "DirectFemaleStatus": female["Status"],
            }
        )
    return pd.DataFrame(rows)


def collect_selected_only(selected_lines: Iterable[str], source_df: pd.DataFrame) -> set[str]:
    valid = set(source_df["LineName"].astype(str).tolist())
    return {line for line in selected_lines if line in valid}


def collect_lines_with_ancestors(selected_lines: Iterable[str], source_df: pd.DataFrame) -> set[str]:
    selected = collect_selected_only(selected_lines, source_df)
    all_lines = set(selected)
    for line in selected:
        ancestors, _, _ = find_ancestors(line, source_df)
        all_lines.update(ancestors)
    return all_lines



def infer_year_from_name(name: str) -> Optional[int]:
    """
    Best-effort year inference from sugarcane-style names.

    Examples:
    - L01-299 -> 2001
    - HoCP96-540 -> 1996
    - LCP85-384 -> 1985
    - 2017410 -> 2017

    This is intentionally heuristic, so plots should be read as an era guide,
    not a strict release-year database.
    """
    if not name:
        return None
    s = str(name).strip()

    # Full 4-digit year anywhere in the name, common for numeric IDs.
    m4 = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", s)
    if m4:
        year = int(m4.group(1))
        if 1900 <= year <= 2099:
            return year

    # Numeric IDs such as 2017410 may begin with a 4-digit year.
    m4_prefix = re.match(r"^(19\d{2}|20\d{2})", s)
    if m4_prefix:
        year = int(m4_prefix.group(1))
        if 1900 <= year <= 2099:
            return year

    # Common sugarcane prefix + two-digit series year before a dash.
    # L01-299, HoCP96-540, LCP85-384, CP72-2086, etc.
    m2 = re.search(r"[A-Za-z]+\s*[-_]?\s*(\d{2})\s*[-_]", s)
    if m2:
        yy = int(m2.group(1))
        return 2000 + yy if yy <= 35 else 1900 + yy

    return None


def decade_label(year: Optional[int]) -> str:
    if year is None:
        return "Unknown"
    return f"{year // 10 * 10}s"


def split_user_patterns(pattern_text: str) -> list[str]:
    if not pattern_text:
        return []
    parts = []
    for chunk in re.split(r"[,\n;]+", pattern_text):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def match_lines_by_patterns(source_df: pd.DataFrame, pattern_text: str, match_mode: str = "contains") -> list[str]:
    """Return LineName values matching one or more user-supplied groups/patterns."""
    patterns = split_user_patterns(pattern_text)
    if not patterns:
        return []

    names = source_df["LineName"].dropna().astype(str).unique().tolist()
    matched: list[str] = []
    for name in names:
        hit = False
        for pat in patterns:
            if match_mode == "regex":
                try:
                    if re.search(pat, name, flags=re.IGNORECASE):
                        hit = True
                        break
                except re.error:
                    continue
            elif match_mode == "startswith":
                if name.lower().startswith(pat.lower()):
                    hit = True
                    break
            else:
                if pat.lower() in name.lower():
                    hit = True
                    break
        if hit:
            matched.append(name)
    return sorted(matched)


def dataframe_to_dash_table(table_df: pd.DataFrame, max_rows: int = 40):
    if table_df is None or table_df.empty:
        return html.Div("No rows to display.")
    preview = table_df.head(max_rows)
    header = html.Thead(html.Tr([html.Th(col) for col in preview.columns]))
    body = html.Tbody([
        html.Tr([html.Td(row.get(col, "")) for col in preview.columns])
        for _, row in preview.iterrows()
    ])
    note = html.Div(
        f"Showing {len(preview):,} of {len(table_df):,} rows.",
        style={"fontWeight": "bold", "marginBottom": "8px"},
    )
    return html.Div([note, html.Table([header, body], className="table table-striped table-sm")])


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical/Torgerson MDS with no sklearn dependency."""
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    if n == 0:
        return np.empty((0, n_components))
    if n == 1:
        return np.zeros((1, n_components))
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals = np.maximum(vals[:n_components], 0)
    coords = vecs[:, :n_components] * np.sqrt(vals)
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])), constant_values=0)
    return coords


def matrix_to_ordination(matrix_df: pd.DataFrame, method: str = "mds") -> pd.DataFrame:
    """Convert a relationship/kinship matrix to 2D plotting coordinates."""
    A = matrix_df.copy()
    A = A.loc[A.index, A.index]
    names = A.index.astype(str).tolist()
    X = A.to_numpy(dtype=float)
    n = len(names)

    if n == 0:
        return pd.DataFrame(columns=["LineName", "Dim1", "Dim2", "Year", "Era"])

    if method == "pca":
        Xc = X - X.mean(axis=0, keepdims=True)
        if n == 1:
            coords = np.zeros((1, 2))
        else:
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            coords = U[:, :2] * S[:2]
            if coords.shape[1] < 2:
                coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), constant_values=0)
        xlab, ylab = "PC1", "PC2"
    else:
        # Relationship-space distance. This behaves like an additive genetic
        # distance when the matrix is positive semidefinite: d² = aii + ajj - 2aij.
        diag = np.diag(X)
        D2 = diag[:, None] + diag[None, :] - 2 * X
        D = np.sqrt(np.maximum(D2, 0))
        coords = classical_mds(D, 2)
        xlab, ylab = "MDS1", "MDS2"

    years = [infer_year_from_name(nm) for nm in names]
    out = pd.DataFrame(
        {
            "LineName": names,
            "Dim1": coords[:, 0],
            "Dim2": coords[:, 1],
            "Year": years,
            "Era": [decade_label(y) for y in years],
            "XLabel": xlab,
            "YLabel": ylab,
        }
    )
    return out


def build_ordination_figure(ord_df: pd.DataFrame, method: str, show_labels: bool):
    if ord_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No matrix data available")
        return fig

    fig = go.Figure()
    eras = sorted(ord_df["Era"].fillna("Unknown").unique().tolist())
    for era in eras:
        sub = ord_df[ord_df["Era"] == era]
        fig.add_trace(
            go.Scatter(
                x=sub["Dim1"],
                y=sub["Dim2"],
                mode="markers+text" if show_labels else "markers",
                text=sub["LineName"] if show_labels else None,
                textposition="top center",
                marker={"size": 10, "line": {"width": 0.5, "color": "black"}},
                name=str(era),
                customdata=np.stack([sub["LineName"], sub["Year"].fillna("Unknown"), sub["Era"]], axis=-1),
                hovertemplate="Line: %{customdata[0]}<br>Year/series: %{customdata[1]}<br>Era: %{customdata[2]}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
            )
        )
    xlab = ord_df["XLabel"].iloc[0]
    ylab = ord_df["YLabel"].iloc[0]
    fig.update_layout(
        title="Kinship matrix ordination: " + ("PCA of matrix rows" if method == "pca" else "MDS of relationship distance"),
        xaxis_title=xlab,
        yaxis_title=ylab,
        hovermode="closest",
        height=720,
        legend_title="Inferred era",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
    )
    return fig


def compute_ancestor_contributions(line_name: str, source_df: pd.DataFrame, max_depth: int = 12) -> pd.DataFrame:
    """
    Estimate expected pedigree contribution for ancestors.

    Each parent receives half of the current contribution recursively. Duplicate
    appearances through multiple paths are accumulated.
    """
    parent_lookup = build_parent_lookup(source_df)
    contributions: dict[str, float] = defaultdict(float)
    path_counts: dict[str, int] = defaultdict(int)
    queue = deque([(line_name, 1.0, 0)])
    seen_guard = 0
    while queue and seen_guard < 100000:
        seen_guard += 1
        child, weight, depth = queue.popleft()
        if depth >= max_depth:
            continue
        male_parent, female_parent = parent_lookup.get(child, (None, None))
        for parent in (female_parent, male_parent):
            if not parent:
                continue
            child_weight = weight * 0.5
            contributions[parent] += child_weight
            path_counts[parent] += 1
            if parent in parent_lookup:
                queue.append((parent, child_weight, depth + 1))

    records = []
    for ancestor, contrib in contributions.items():
        male_parent, female_parent = parent_lookup.get(ancestor, (None, None))
        is_founder = not male_parent and not female_parent
        records.append(
            {
                "Ancestor": ancestor,
                "ExpectedContribution": contrib,
                "Percent": contrib * 100,
                "PathCount": path_counts[ancestor],
                "InferredYear": infer_year_from_name(ancestor),
                "Era": decade_label(infer_year_from_name(ancestor)),
                "FounderLike": is_founder,
            }
        )
    return pd.DataFrame(records).sort_values("ExpectedContribution", ascending=False).reset_index(drop=True)


def collect_founders_for_line(line_name: str, source_df: pd.DataFrame, max_depth: int = 12) -> set[str]:
    parent_lookup = build_parent_lookup(source_df)
    founders: set[str] = set()
    queue = deque([(line_name, 0)])
    visited_paths = 0
    while queue and visited_paths < 100000:
        visited_paths += 1
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        male_parent, female_parent = parent_lookup.get(node, (None, None))
        parents = [p for p in (male_parent, female_parent) if p]
        if not parents and node != line_name:
            founders.add(node)
        for parent in parents:
            pm, pf = parent_lookup.get(parent, (None, None))
            if not pm and not pf:
                founders.add(parent)
            else:
                queue.append((parent, depth + 1))
    return founders


def build_group_insights(pattern_text: str, match_mode: str, max_depth: int = 12):
    matched = match_lines_by_patterns(filtered_df, pattern_text, match_mode)
    matched_df = filtered_df[filtered_df["LineName"].isin(matched)].copy()
    if matched_df.empty:
        return matched, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    parent_records = []
    for _, row in matched_df.iterrows():
        for parent_col, role in [("FemaleParent", "Female"), ("MaleParent", "Male")]:
            parent = clean_parent_value(row.get(parent_col))
            if parent:
                parent_records.append({"Parent": parent, "Role": role, "Child": row["LineName"]})
    parent_counts = (
        pd.DataFrame(parent_records)
        .groupby(["Parent", "Role"], as_index=False)
        .agg(Uses=("Child", "nunique"))
        if parent_records
        else pd.DataFrame(columns=["Parent", "Role", "Uses"])
    )
    if not parent_counts.empty:
        parent_counts = parent_counts.sort_values("Uses", ascending=False)

    timeline = pd.DataFrame(
        {
            "LineName": matched,
            "InferredYear": [infer_year_from_name(x) for x in matched],
        }
    )
    timeline["Era"] = timeline["InferredYear"].map(decade_label)
    timeline_counts = timeline.groupby("Era", as_index=False).agg(Count=("LineName", "nunique")).sort_values("Era")

    founder_counts: dict[str, int] = defaultdict(int)
    for line in matched:
        for founder in collect_founders_for_line(line, filtered_df, max_depth=max_depth):
            founder_counts[founder] += 1
    founders_df = pd.DataFrame(
        [{"Founder": k, "MatchedLinesUsingFounder": v, "InferredYear": infer_year_from_name(k)} for k, v in founder_counts.items()]
    )
    if not founders_df.empty:
        founders_df["Era"] = founders_df["InferredYear"].map(decade_label)
        founders_df = founders_df.sort_values("MatchedLinesUsingFounder", ascending=False)

    matched_df["InferredYear"] = matched_df["LineName"].map(infer_year_from_name)
    matched_df["Era"] = matched_df["InferredYear"].map(decade_label)
    return matched, parent_counts, timeline_counts, founders_df, matched_df


def horizontal_bar_figure(table_df: pd.DataFrame, x_col: str, y_col: str, title: str, top_n: int = 25):
    fig = go.Figure()
    if table_df is None or table_df.empty:
        fig.update_layout(title=title + " — no data", height=420)
        return fig
    plot_df = table_df.head(top_n).iloc[::-1]
    fig.add_trace(go.Bar(x=plot_df[x_col], y=plot_df[y_col], orientation="h", text=plot_df[x_col], textposition="auto"))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title="", height=max(420, 24 * len(plot_df) + 120), margin={"l": 160, "r": 20, "t": 60, "b": 40})
    return fig


def timeline_figure(timeline_counts: pd.DataFrame):
    fig = go.Figure()
    if timeline_counts is None or timeline_counts.empty:
        fig.update_layout(title="Inferred era/timeline — no data", height=420)
        return fig
    fig.add_trace(go.Bar(x=timeline_counts["Era"], y=timeline_counts["Count"], text=timeline_counts["Count"], textposition="auto"))
    fig.update_layout(title="Inferred era/timeline distribution", xaxis_title="Inferred era", yaxis_title="Number of matched lines", height=460)
    return fig


def founder_cloud_figure(founders_df: pd.DataFrame, top_n: int = 50):
    fig = go.Figure()
    if founders_df is None or founders_df.empty:
        fig.update_layout(title="Founder origin cloud — no founder-like ancestors found", height=500)
        return fig
    plot_df = founders_df.head(top_n).copy()
    # Deterministic pseudo-cloud coordinates so the layout is stable between runs.
    angles = np.linspace(0, 4 * np.pi, len(plot_df), endpoint=False)
    radii = np.linspace(0.2, 1.0, len(plot_df))
    sizes = 12 + 42 * (plot_df["MatchedLinesUsingFounder"] / plot_df["MatchedLinesUsingFounder"].max())
    fig.add_trace(
        go.Scatter(
            x=radii * np.cos(angles),
            y=radii * np.sin(angles),
            mode="text",
            text=plot_df["Founder"],
            textfont={"size": sizes},
            customdata=np.stack([plot_df["MatchedLinesUsingFounder"], plot_df.get("Era", pd.Series(["Unknown"] * len(plot_df)))], axis=-1),
            hovertemplate="Founder: %{text}<br>Matched lines using founder: %{customdata[0]}<br>Era: %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Founder origin cloud", height=620, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return fig


def build_ancestor_subgraph(selected_line: str, source_df: pd.DataFrame, max_depth: int):
    ancestors, relationships, generations = find_ancestors(selected_line, source_df)
    nodes = set()
    for gen in range(max_depth + 1):
        nodes.update(generations.get(gen, []))
    nodes.add(selected_line)
    edges = [(p, c, r) for p, c, r in relationships if p in nodes and c in nodes]
    return nodes, edges, generations



def build_pedigree_fan_rows(selected_line: str, source_df: pd.DataFrame, max_depth: int) -> pd.DataFrame:
    """
    Build a slot-based pedigree fan chart table.

    Unlike a normal graph, this keeps each ancestral position as its own slot.
    That means repeated ancestors can appear in multiple sectors, which is exactly
    what pedigree wheels/fan charts need.
    """
    max_depth = int(max(1, min(max_depth or 5, 12)))
    parent_lookup = build_parent_lookup(source_df)

    rows: list[dict] = []
    queue = deque()

    root_value = float(2 ** max_depth)
    rows.append(
        {
            "id": "root",
            "label": selected_line,
            "line": selected_line,
            "parent": "",
            "generation": 0,
            "role": "selected",
            "root_branch": "Selected genotype",
            "path": selected_line,
            "value": root_value,
            "missing": False,
        }
    )
    queue.append((selected_line, "root", 0, selected_line, "Selected genotype"))

    role_specs = [
        ("female", "F", "Female parent", "Female ancestry branch"),
        ("male", "M", "Male parent", "Male ancestry branch"),
    ]

    while queue:
        line, parent_id, gen, path, root_branch = queue.popleft()
        if gen >= max_depth:
            continue

        male_parent, female_parent = parent_lookup.get(line, (None, None))
        role_to_parent = {"male": male_parent, "female": female_parent}

        for role, short_role, role_label, branch_label in role_specs:
            parent_line = role_to_parent.get(role)
            child_gen = gen + 1
            child_id = f"{parent_id}|{short_role}"
            child_path = f"{path} > {short_role}"
            next_root_branch = branch_label if gen == 0 else root_branch
            missing = parent_line is None
            label = parent_line if parent_line else f"Unknown {role_label.lower()}"

            rows.append(
                {
                    "id": child_id,
                    "label": label,
                    "line": parent_line or "Unknown",
                    "parent": parent_id,
                    "generation": child_gen,
                    "role": role_label,
                    "root_branch": next_root_branch,
                    "path": child_path,
                    "value": float(2 ** (max_depth - child_gen)),
                    "missing": missing,
                }
            )

            # Missing parents are intentionally not expanded. Their sector value
            # represents the whole unobserved branch beyond that point.
            if parent_line:
                queue.append((parent_line, child_id, child_gen, child_path, next_root_branch))

    fan_df = pd.DataFrame(rows)
    if fan_df.empty:
        return fan_df

    non_missing = fan_df.loc[~fan_df["missing"] & fan_df["line"].ne("Unknown"), "line"]
    repeated_counts = non_missing.value_counts().to_dict()
    fan_df["slot_contribution_pct"] = fan_df["generation"].map(lambda g: 100.0 / (2 ** int(g)) if int(g) > 0 else 100.0)
    fan_df["total_slots_for_line"] = fan_df["line"].map(lambda x: int(repeated_counts.get(x, 0)) if x != "Unknown" else 0)
    contribution_totals = (
        fan_df.loc[~fan_df["missing"] & fan_df["line"].ne("Unknown")]
        .groupby("line")["slot_contribution_pct"]
        .sum()
        .to_dict()
    )
    fan_df["total_contribution_pct"] = fan_df["line"].map(lambda x: float(contribution_totals.get(x, 0.0)))
    fan_df["inferred_year"] = fan_df["line"].map(lambda x: infer_year_from_name(x) if x != "Unknown" else None)
    fan_df["display_year"] = fan_df["inferred_year"].map(lambda x: int(x) if pd.notna(x) else "Unknown")
    return fan_df


def pedigree_fan_chart_figure(selected_line: str, max_depth: int, large_export: bool = False):
    """Circular pedigree wheel/fan chart, closer to published ancestry-ring visuals.

    The on-screen figure is allowed to hide tiny labels so it stays readable.
    The large export forces labels to show and uses a much larger canvas.
    """
    fan_df = build_pedigree_fan_rows(selected_line, filtered_df, max_depth)
    if fan_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No pedigree nodes found")
        return fig

    def sector_color(row):
        if bool(row["missing"]):
            return "#d9d9d9"
        if row["generation"] == 0:
            return "#31a354"
        # Repeated known ancestors get a purple tint because they indicate
        # duplicated ancestry/inbreeding loops in the fan.
        if int(row.get("total_slots_for_line", 0)) > 1 and row["generation"] > 1:
            return "#9e9ac8"
        if row["root_branch"] == "Female ancestry branch":
            return "#fdae6b"
        if row["root_branch"] == "Male ancestry branch":
            return "#9ecae1"
        return "#f0f0f0"

    colors = [sector_color(row) for _, row in fan_df.iterrows()]

    def label_color(row):
        # Text color indicates the sex/role of the parent at that sector.
        # Female parent slots are red; male parent slots are blue; root/unknown are neutral.
        role = str(row.get("role", ""))
        if role == "Female parent":
            return "#b30000"
        if role == "Male parent":
            return "#08519c"
        if bool(row.get("missing", False)):
            return "#666666"
        return "#111111"

    text_colors = [label_color(row) for _, row in fan_df.iterrows()]
    text_size = 14 if large_export else 10

    customdata = np.stack(
        [
            fan_df["generation"].astype(str),
            fan_df["role"].astype(str),
            fan_df["root_branch"].astype(str),
            fan_df["display_year"].astype(str),
            fan_df["path"].astype(str),
            fan_df["slot_contribution_pct"].round(3).astype(str),
            fan_df["total_contribution_pct"].round(3).astype(str),
            fan_df["total_slots_for_line"].astype(str),
        ],
        axis=-1,
    )

    fig = go.Figure(
        go.Sunburst(
            ids=fan_df["id"],
            labels=fan_df["label"],
            parents=fan_df["parent"],
            values=fan_df["value"],
            branchvalues="total",
            maxdepth=int(max_depth) + 1,
            insidetextorientation="radial",
            textinfo="label",
            textfont={"color": text_colors, "size": text_size},
            insidetextfont={"color": text_colors, "size": text_size},
            marker={"colors": colors, "line": {"color": "white", "width": 1.2}},
            customdata=customdata,
            hovertemplate=(
                "Line/slot: %{label}<br>"
                "Generation: %{customdata[0]}<br>"
                "Role: %{customdata[1]}<br>"
                "Major branch: %{customdata[2]}<br>"
                "Inferred year: %{customdata[3]}<br>"
                "Path: %{customdata[4]}<br>"
                "Expected contribution from this slot: %{customdata[5]}%<br>"
                "Total expected contribution for this line: %{customdata[6]}%<br>"
                "Number of repeated slots for this line: %{customdata[7]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Pedigree fan chart for {selected_line}" + (" — full-label export" if large_export else ""),
        height=max(1700, int(max_depth) * 260) if large_export else 850,
        width=max(1700, int(max_depth) * 260) if large_export else None,
        margin={"l": 20, "r": 20, "t": 95, "b": 55} if large_export else {"l": 10, "r": 10, "t": 75, "b": 35},
        uniformtext={"minsize": 10 if large_export else 8, "mode": "show" if large_export else "hide"},
        annotations=[
            {
                "text": "Warm = female-side branch • Cool = male-side branch • Purple = repeated ancestor • Gray = unknown branch • Red text = female parent slot • Blue text = male parent slot",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.02,
                "showarrow": False,
                "font": {"size": 12},
            }
        ],
    )
    return fig


def radial_node_ring_figure(selected_line: str, max_depth: int):
    nodes, edges, generations = build_ancestor_subgraph(selected_line, filtered_df, max_depth)
    if not nodes:
        fig = go.Figure()
        fig.update_layout(title="No pedigree nodes found")
        return fig

    coords: dict[str, tuple[float, float, int]] = {}
    for gen in range(max_depth + 1):
        gen_nodes = sorted(set(generations.get(gen, [])))
        if gen == 0:
            coords[selected_line] = (0.0, 0.0, 0)
            continue
        if not gen_nodes:
            continue
        for i, node in enumerate(gen_nodes):
            angle = 2 * np.pi * i / max(len(gen_nodes), 1)
            coords[node] = (gen * np.cos(angle), gen * np.sin(angle), gen)

    edge_x, edge_y = [], []
    for parent, child, role in edges:
        if parent in coords and child in coords:
            x0, y0, _ = coords[parent]
            x1, y1, _ = coords[child]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    node_df = pd.DataFrame(
        [{"LineName": n, "x": coords[n][0], "y": coords[n][1], "Generation": coords[n][2], "Year": infer_year_from_name(n)} for n in coords]
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 1}, hoverinfo="skip", name="Parent links"))
    fig.add_trace(
        go.Scatter(
            x=node_df["x"],
            y=node_df["y"],
            mode="markers+text",
            text=node_df["LineName"],
            textposition="top center",
            marker={"size": np.where(node_df["LineName"] == selected_line, 20, 12), "line": {"width": 1, "color": "black"}},
            customdata=np.stack([node_df["Generation"], node_df["Year"].fillna("Unknown")], axis=-1),
            hovertemplate="Line: %{text}<br>Ancestor generation: %{customdata[0]}<br>Inferred year: %{customdata[1]}<extra></extra>",
            name="Pedigree nodes",
        )
    )
    # Add faint ring guides.
    shapes = []
    for gen in range(1, max_depth + 1):
        shapes.append({"type": "circle", "xref": "x", "yref": "y", "x0": -gen, "y0": -gen, "x1": gen, "y1": gen, "line": {"width": 1, "dash": "dot"}})
    fig.update_layout(
        title=f"Generation ring layout for {selected_line}",
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        shapes=shapes,
        height=760,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def generation_ring_figure(selected_line: str, max_depth: int, style: str = "fan", large_export: bool = False):
    """Return either the new pedigree fan chart or the older node/link ring layout."""
    if style == "node-ring":
        fig = radial_node_ring_figure(selected_line, max_depth)
        if large_export:
            fig.update_layout(height=max(1400, int(max_depth) * 180), width=max(1400, int(max_depth) * 180))
        return fig
    return pedigree_fan_chart_figure(selected_line, max_depth, large_export=large_export)



def cytoscape_elements_for_line(selected_line: str, max_depth: int):
    nodes, edges, generations = build_ancestor_subgraph(selected_line, filtered_df, max_depth)
    parent_lookup = build_parent_lookup(filtered_df)
    elements = []
    for node in sorted(nodes):
        male_parent, female_parent = parent_lookup.get(node, (None, None))
        generation = None
        for gen, names in generations.items():
            if node in names:
                generation = gen
                break
        classes = "selected" if node == selected_line else "founder" if not male_parent and not female_parent else "normal"
        elements.append(
            {
                "data": {
                    "id": node,
                    "label": node,
                    "male_parent": male_parent or "",
                    "female_parent": female_parent or "",
                    "generation": generation if generation is not None else "",
                    "year": infer_year_from_name(node) or "",
                },
                "classes": classes,
            }
        )
    for parent, child, role in edges:
        elements.append({"data": {"source": parent, "target": child, "role": role}, "classes": role})
    return elements


# =============================================================================
# Matrix math
# =============================================================================


def sort_pedigree_df(pedigree_df: pd.DataFrame) -> pd.DataFrame:
    """Topologically sort pedigree so parents appear before children when possible."""
    if pedigree_df.empty:
        return pedigree_df.copy()

    work = normalize_pedigree_df(pedigree_df)
    line_set = set(work["LineName"])
    children: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {line: 0 for line in work["LineName"]}

    for _, row in work.iterrows():
        child = row["LineName"]
        for parent in (clean_parent_value(row["MaleParent"]), clean_parent_value(row["FemaleParent"])):
            if parent and parent in line_set:
                children[parent].append(child)
                indegree[child] += 1

    queue = deque([line for line in work["LineName"] if indegree[line] == 0])
    sorted_lines: list[str] = []
    visited: set[str] = set()

    while queue:
        line = queue.popleft()
        if line in visited:
            continue
        visited.add(line)
        sorted_lines.append(line)
        for child in children.get(line, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    remaining = [line for line in work["LineName"] if line not in visited]
    if remaining:
        print("WARNING: Cycles or unresolved ordering detected. Appending remaining lines:", remaining)
        sorted_lines.extend(remaining)

    return work.set_index("LineName").loc[sorted_lines].reset_index()


@njit
def _build_matrix_numba(n, sire_idxs, dam_idxs):
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        s_i = sire_idxs[i]
        d_i = dam_idxs[i]
        if s_i == -1 and d_i == -1:
            A[i, i] = 1.0
        elif s_i != -1 and d_i == -1:
            A[i, i] = 1.0
            A[i, :i] = 0.5 * A[s_i, :i]
            A[:i, i] = A[i, :i]
        elif s_i == -1 and d_i != -1:
            A[i, i] = 1.0
            A[i, :i] = 0.5 * A[d_i, :i]
            A[:i, i] = A[i, :i]
        else:
            A[i, i] = 1.0 + 0.5 * A[s_i, d_i]
            temp = 0.5 * (A[s_i, :i] + A[d_i, :i])
            A[i, :i] = temp
            A[:i, i] = temp
    return A


def compute_amatrix_diploid(pedigree_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = sort_pedigree_df(pedigree_df)
    individuals = sorted_df["LineName"].tolist()
    idx_map = {ind: i for i, ind in enumerate(individuals)}
    n = len(individuals)

    sire_idxs = np.full(n, -1, dtype=np.int64)
    dam_idxs = np.full(n, -1, dtype=np.int64)

    for i, row in sorted_df.iterrows():
        sire = clean_parent_value(row.get("MaleParent"))
        dam = clean_parent_value(row.get("FemaleParent"))
        if sire in idx_map:
            sire_idxs[i] = idx_map[sire]
        if dam in idx_map:
            dam_idxs[i] = idx_map[dam]

    A = _build_matrix_numba(n, sire_idxs, dam_idxs)
    return pd.DataFrame(A, index=individuals, columns=individuals)


def compute_amatrix_coancestry(pedigree_df: pd.DataFrame) -> pd.DataFrame:
    return compute_amatrix_diploid(pedigree_df) / 2.0


def compute_selected_matrix(pedigree_df: pd.DataFrame, method_choice: int) -> pd.DataFrame:
    if method_choice == 0:
        return compute_amatrix_diploid(pedigree_df)
    return compute_amatrix_coancestry(pedigree_df)

# =============================================================================
# Layout
# =============================================================================

app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(id="page-content")])


def base_layout(content):
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Button(
                                    html.I(className="fas fa-home"),
                                    id="home-button",
                                    style=CUSTOM_CSS["round_button"],
                                ),
                                html.Button(
                                    html.I(className="fas fa-table"),
                                    id="view-data-button",
                                    style={**CUSTOM_CSS["round_button"], "backgroundColor": "#27ae60"},
                                ),
                            ],
                            style=CUSTOM_CSS["left_button_group"],
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Button(
                                    html.I(className="fas fa-trash"),
                                    id="clear-data-button",
                                    style={**CUSTOM_CSS["round_button"], "backgroundColor": "#e74c3c"},
                                ),
                                html.Button("?", id="open-info-modal", style=CUSTOM_CSS["round_button"]),
                            ],
                            style=CUSTOM_CSS["right_button_group"],
                        ),
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    html.H1("CaneCestry", className="app-header", style={**CUSTOM_CSS["header"], "fontSize": "60px"})
                )
            ),
            dbc.Row(dbc.Col(html.Div(id="common-content", children=content))),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("CaneCestry 2.0 – User Reference")),
                    dbc.ModalBody(
                        html.Div(
                            [
                                html.H4("Overview"),
                                html.P(
                                    "CaneCestry supports pedigree management, pedigree visualization, "
                                    "ancestry queries, and generation of Henderson A-matrices or coancestry matrices."
                                ),
                                html.H4("Matrix Generation"),
                                html.P(
                                    "Use one page, one Generate button, and the pedigree-expansion slider to choose "
                                    "Selected lines only, Selected lines + ancestors, or Selected lines + ancestors + descendants."
                                ),
                                html.H4("Pedigree Explorer"),
                                html.Ul(
                                    [
                                        html.Li("Specific pairing lookup."),
                                        html.Li("Single-parent progeny lookup."),
                                        html.Li("SVG family trees with generation-depth control."),
                                        html.Li("Male/female lineage highlighting using blue/red border emphasis."),
                                        html.Li("Descendant, combined-family, and temporary-progeny trees."),
                                    ]
                                ),
                                html.H4("Add Pedigree Entries"),
                                html.P(
                                    "Upload rows with LineName, FemaleParent, and MaleParent. Valid rows are saved, "
                                    "and rows with unrecognized parents can be corrected before saving."
                                ),
                            ],
                            style={"fontSize": "15px"},
                        )
                    ),
                    dbc.ModalFooter(dbc.Button("Close", id="close-info-modal", className="ms-auto")),
                ],
                id="info-modal",
                size="lg",
                is_open=False,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Current Dataset")),
                    dbc.ModalBody(html.Div(id="current-data-table", style=CUSTOM_CSS["table_wrap"])),
                    dbc.ModalFooter(dbc.Button("Close", id="close-data-modal", className="ms-auto")),
                ],
                id="data-modal",
                size="xl",
                is_open=False,
            ),
        ],
        fluid=True,
        style=CUSTOM_CSS["container"],
    )


def splash_page_layout():
    content = [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Choose Your Data Source")),
                dbc.ModalBody(
                    [
                        dcc.RadioItems(
                            id="splash-data-choice",
                            options=[
                                {"label": "Use built-in sugarcane data", "value": "example"},
                                {"label": "Upload your own data", "value": "upload"},
                            ],
                            value="example",
                            labelStyle={"display": "block", "marginBottom": "10px"},
                            style={"marginBottom": "20px"},
                        ),
                        html.Div(
                            [
                                html.P(
                                    "Upload a .csv or tab-delimited .txt with columns: LineName, FemaleParent, MaleParent",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.Upload(
                                    id="splash-upload-data",
                                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="splash-upload-status", style={"marginTop": "10px", "color": "green"}),
                            ],
                            id="splash-upload-section",
                            style={"display": "none"},
                        ),
                    ]
                ),
                dbc.ModalFooter(dbc.Button("Proceed", id="splash-modal-proceed-btn", color="primary")),
            ],
            id="splash-data-modal",
            is_open=True,
            backdrop="static",
            keyboard=False,
        ),
        dbc.Row(dbc.Col(html.P("Please select an option:", style={"textAlign": "center", "fontSize": "24px"}))),
        big_nav_button("Generate Kinship Matrix", "/main-page"),
        big_nav_button("Pedigree Explorer", "/progeny-finder"),
        big_nav_button("Add Pedigree Entries", "/add-pedigree-entries"),
    ]
    return base_layout(content)


def big_nav_button(label: str, href: str):
    return dbc.Row(
        dbc.Col(
            html.A(
                label,
                href=href,
                className="btn",
                style={
                    "width": "320px",
                    "height": "80px",
                    "fontSize": "20px",
                    "marginBottom": "10px",
                    "backgroundColor": "#2980b9",
                    "color": "white",
                    "borderRadius": "10px",
                    "fontWeight": "bold",
                    "display": "flex",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "textDecoration": "none",
                },
            ),
            width="auto",
        ),
        justify="center",
    )


def method_slider(id_value="kinship-method-slider"):
    return dcc.Slider(
        id=id_value,
        min=0,
        max=1,
        step=1,
        marks={0: "A Matrix", 1: "Coancestry"},
        value=1,
        tooltip={"placement": "bottom", "always_visible": True},
    )


def main_page_layout():
    content = [
        dbc.Row(dbc.Col(html.H1("Generate Kinship Matrix", style=CUSTOM_CSS["header"]))),
        dbc.Row(dbc.Col(html.Label("Relationship calculation:", style={"fontWeight": "bold"}))),
        dbc.Row(dbc.Col(html.Div(method_slider(), style={"width": "360px"}), width="auto"), justify="center"),
        html.Br(),
        dbc.Row(dbc.Col(html.Label("Pedigree expansion:", style={"fontWeight": "bold"}))),
        dbc.Row(
            dbc.Col(
                html.Div(
                    dcc.Slider(
                        id="pedigree-expansion-slider",
                        min=0,
                        max=2,
                        step=1,
                        marks={
                            0: "Selected only",
                            1: "Selected + ancestors",
                            2: "Selected + ancestors + descendants",
                        },
                        value=1,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    style={"width": "720px"},
                ),
                width="auto",
            ),
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Select or paste line names:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="line-name-dropdown",
                            options=line_options(),
                            multi=True,
                            placeholder="Type to search line names...",
                            style=CUSTOM_CSS["dropdown"],
                        ),
                        dcc.Input(
                            id="paste-line-names",
                            type="text",
                            placeholder="Paste comma-separated line names here...",
                            style={"marginTop": "10px", "width": "100%"},
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        dbc.Row(dbc.Col(html.Ul(id="selected-line-names-list", children=[]))),
        dbc.Row(
            [
                dbc.Col(html.Button("Generate Kinship Matrix", id="generate-amatrix-button", className="btn btn-info", style=CUSTOM_CSS["button"]), width=4),
                dbc.Col(html.A("Download Full Matrix", id="download-full-link", href="", download="full_matrix.csv", className="btn btn-success", style=CUSTOM_CSS["button"]), width=4),
                dbc.Col(html.A("Download Subset Matrix", id="download-subset-link", href="", download="subset_matrix.csv", className="btn btn-warning", style=CUSTOM_CSS["button"]), width=4),
            ],
            style={"marginTop": "20px"},
            className="mb-4",
        ),
        dbc.Row(dbc.Col(html.Div(id="matrix-status", style={"fontWeight": "bold", "marginTop": "10px"}))),
        dcc.Loading(
            dbc.Row(dbc.Col(html.Img(id="heatmap-image", src="", style={"width": "100%", "padding": "10px"}))),
            type="default",
        ),
        html.Div(
            html.A(
                "Download high-resolution heatmap PNG",
                id="download-heatmap-link",
                href="",
                download="kinship_heatmap.png",
                className="btn btn-warning",
                style={**CUSTOM_CSS["button"], "display": "none"},
            )
        ),
        html.Hr(),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Matrix PCA / MDS ordination"),
                    html.P(
                        "Convert the kinship matrix into a 2D scatterplot. Closely related lines should cluster together; hover over points to see names and inferred eras."
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.RadioItems(
                                    id="matrix-ordination-method",
                                    options=[
                                        {"label": "MDS from relationship distance", "value": "mds"},
                                        {"label": "PCA of matrix rows", "value": "pca"},
                                    ],
                                    value="mds",
                                    labelStyle={"display": "block", "marginBottom": "6px"},
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                dcc.Checklist(
                                    id="matrix-ordination-labels",
                                    options=[{"label": "Show point labels", "value": "labels"}],
                                    value=[],
                                ),
                                width=6,
                            ),
                        ]
                    ),
                    dcc.Loading(dcc.Graph(id="matrix-ordination-plot", config=HIGHRES_GRAPH_CONFIG), type="default"),
                ]
            ),
            className="mb-3",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="subset-dropdown",
                    options=[],
                    multi=True,
                    placeholder="Select lines for subset matrix...",
                    style=CUSTOM_CSS["dropdown"],
                )
            ),
            style={"marginTop": "20px"},
        ),
        html.Div(id="subset-matrix-display", style=CUSTOM_CSS["table_wrap"]),
        dcc.Store(id="matrix-store"),
    ]
    return base_layout(content)


def pedigree_explorer_layout():
    content = [
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="progeny-module-dropdown",
                    options=[
                        {"label": "Lookup progeny of specific pairing", "value": "specific-pairing"},
                        {"label": "Lookup progeny of single parent", "value": "single-parent"},
                        {"label": "Generate family tree", "value": "family-tree"},
                        {"label": "Generate descendant tree", "value": "descendant-tree"},
                        {"label": "Generate combined family tree for two lines", "value": "combined-family-tree"},
                        {"label": "Generate family tree with temporary progeny", "value": "temp-progeny-tree"},
                        {"label": "Interactive pedigree viewer", "value": "interactive-pedigree"},
                        {"label": "Generation ring layout", "value": "generation-ring"},
                        {"label": "Group / era / founder insights", "value": "group-era-insights"},
                        {"label": "Ancestor contribution chart", "value": "ancestor-contribution"},
                        {"label": "Furthest direct male/female ancestors", "value": "direct-line-ancestors"},
                    ],
                    multi=True,
                    placeholder="Select one or more functions",
                    style=CUSTOM_CSS["dropdown"],
                ),
                width=12,
            )
        ),
        html.Br(),
        dbc.Row(dbc.Col(html.Label("Relationship calculation for kinship-colored trees:", style={"fontWeight": "bold"}))),
        dbc.Row(dbc.Col(html.Div(method_slider(), style={"width": "360px"}), width="auto"), justify="center"),
        html.Hr(),
        html.Div(id="selected-progeny-modules", children=[]),
        dbc.Row(dbc.Col(html.A("Back to Generate Matrix", href="/main-page", className="btn btn-secondary", style=CUSTOM_CSS["button"]), width=12)),
    ]
    return base_layout(content)


def add_pedigree_entries_layout():
    content = [
        dbc.Row(dbc.Col(html.H3("Add Pedigree Entries", style=CUSTOM_CSS["header"]))),
        dbc.Row(
            [
                dbc.Col(html.Label("Upload .txt or .csv pedigree entries:"), width=3),
                dbc.Col(
                    dcc.Upload(
                        id="add-ped-upload",
                        children=html.Div(["Drag & Drop or ", html.A("Select File")]),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                        },
                        multiple=False,
                    ),
                    width=6,
                ),
                dbc.Col(html.Div(id="add-ped-upload-status", style={"color": "green"}), width=12),
            ]
        ),
        html.Hr(),
        dbc.Row([dbc.Col(html.H4("Rows Needing Parent Corrections:"), width=12), dbc.Col(html.Div(id="add-ped-missing-parents-table"), width=12)]),
        dcc.Store(id="add-ped-missing-rows-store"),
        dcc.Store(id="add-ped-valid-rows-store"),
        html.Hr(),
        dbc.Row(dbc.Col(html.A("Back to Generate Matrix", href="/main-page", className="btn btn-secondary", style=CUSTOM_CSS["button"]), width=12)),
    ]
    return base_layout(content)

# =============================================================================
# Page routing and common callbacks
# =============================================================================


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/main-page":
        return main_page_layout()
    if pathname == "/progeny-finder":
        return pedigree_explorer_layout()
    if pathname in {"/add-pedigree-entries", "/dummy-progeny-matrix"}:
        return add_pedigree_entries_layout()
    return splash_page_layout()


@app.callback(
    Output("url", "pathname"),
    Input("home-button", "n_clicks"),
    prevent_initial_call=True,
)
def navigate_home(_):
    return "/"


@app.callback(
    Output("info-modal", "is_open"),
    [Input("open-info-modal", "n_clicks"), Input("close-info-modal", "n_clicks")],
    State("info-modal", "is_open"),
)
def toggle_info_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open


@app.callback(
    Output("data-modal", "is_open"),
    [Input("view-data-button", "n_clicks"), Input("close-data-modal", "n_clicks")],
    State("data-modal", "is_open"),
)
def toggle_data_modal(view_clicks, close_clicks, is_open):
    if view_clicks or close_clicks:
        return not is_open
    return is_open


@app.callback(Output("current-data-table", "children"), Input("data-modal", "is_open"))
def update_current_data_table(is_open):
    if not is_open:
        raise PreventUpdate
    if df.empty:
        return "No data available."
    preview = df.head(1000)
    header = html.Thead(html.Tr([html.Th(col) for col in preview.columns]))
    body = html.Tbody([html.Tr([html.Td(row.get(col, "")) for col in preview.columns]) for _, row in preview.iterrows()])
    note = html.Div(f"Showing {len(preview):,} of {len(df):,} rows.", style={"fontWeight": "bold", "marginBottom": "10px"})
    return html.Div([note, html.Table([header, body], className="table table-striped table-sm")])


@app.callback(
    Output("clear-data-button", "children"),
    Input("clear-data-button", "n_clicks"),
    prevent_initial_call=True,
)
def clear_data(_):
    set_active_dataframe(default_df.copy(), persist=False)
    if USER_FILE.exists():
        try:
            USER_FILE.unlink()
        except OSError:
            pass
    return [html.I(className="fas fa-trash"), " Cleared!"]

# =============================================================================
# Splash upload callbacks
# =============================================================================


@app.callback(Output("splash-upload-section", "style"), Input("splash-data-choice", "value"))
def toggle_splash_upload_section(choice):
    return {"display": "block"} if choice == "upload" else {"display": "none"}


@app.callback(
    Output("splash-data-modal", "is_open"),
    Input("splash-modal-proceed-btn", "n_clicks"),
    State("splash-data-modal", "is_open"),
)
def close_splash_modal(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open


@app.callback(
    Output("splash-upload-status", "children"),
    Input("splash-upload-data", "contents"),
    State("splash-upload-data", "filename"),
    prevent_initial_call=True,
)
def upload_splash_pedigree(contents, filename):
    if contents is None:
        raise PreventUpdate
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        if filename and filename.lower().endswith(".csv"):
            uploaded = pd.read_csv(BytesIO(decoded))
        else:
            uploaded = pd.read_csv(BytesIO(decoded), sep="\t")
        msg = set_active_dataframe(uploaded, persist=True)
        return f"Successfully uploaded {filename}. {msg} Saved to {USER_FILE.relative_to(BASE_DIR)}."
    except Exception as exc:
        return f"Error processing file: {exc}"

# =============================================================================
# Matrix page callbacks
# =============================================================================


@app.callback(
    Output("line-name-dropdown", "value"),
    [Input("line-name-dropdown", "value"), Input("paste-line-names", "value")],
)
def update_line_selection(selected_lines, pasted_lines):
    selected_lines = selected_lines or []
    if not pasted_lines:
        return selected_lines
    valid_names = set(filtered_df["LineName"].astype(str).tolist())
    pasted = [name.strip() for name in pasted_lines.split(",") if name.strip()]
    merged = list(dict.fromkeys(selected_lines + [name for name in pasted if name in valid_names]))
    return merged


@app.callback(Output("selected-line-names-list", "children"), Input("line-name-dropdown", "value"))
def update_selected_line_names_list(selected_line_names):
    if not selected_line_names:
        return "No lines selected."
    return [html.Li(name) for name in selected_line_names]


@app.callback(
    [
        Output("heatmap-image", "src"),
        Output("download-full-link", "href"),
        Output("download-heatmap-link", "href"),
        Output("download-heatmap-link", "style"),
        Output("matrix-store", "data"),
        Output("matrix-status", "children"),
    ],
    Input("generate-amatrix-button", "n_clicks"),
    [
        State("line-name-dropdown", "value"),
        State("kinship-method-slider", "value"),
        State("pedigree-expansion-slider", "value"),
    ],
    prevent_initial_call=True,
)
def generate_amatrix_and_heatmap(n_clicks, selected_line_names, method_choice, expansion_choice):
    if not n_clicks or not selected_line_names:
        raise PreventUpdate

    start_time = time.time()
    if expansion_choice == 0:
        all_related_lines = collect_selected_only(selected_line_names, filtered_df)
        expansion_label = "selected lines only"
    elif expansion_choice == 2:
        all_related_lines = collect_lines_with_ancestors_and_descendants(selected_line_names, filtered_df)
        expansion_label = "selected lines + ancestors + descendants"
    else:
        all_related_lines = collect_lines_with_ancestors(selected_line_names, filtered_df)
        expansion_label = "selected lines + ancestors"

    if not all_related_lines:
        return "", "", "", {**CUSTOM_CSS["button"], "display": "none"}, None, "No valid selected lines were found in the active pedigree."

    relatives_df = filtered_df[filtered_df["LineName"].isin(all_related_lines)].copy()
    current_matrix = compute_selected_matrix(relatives_df, method_choice)

    timestamp = int(time.time())
    matrix_file = OUTPUT_DIR / f"full_matrix_{timestamp}.csv"
    current_matrix.to_csv(matrix_file)
    full_matrix_link = f"/download?filename={urllib.parse.quote(str(matrix_file))}&type=full"

    heatmap_file = OUTPUT_DIR / f"heatmap_{timestamp}.png"
    n_lines = len(current_matrix)
    if n_lines <= MAX_CLUSTER_SIZE:
        heatmap_plot = sns.clustermap(current_matrix, method="average", cmap="Spectral", figsize=(15, 15))
        heatmap_plot.savefig(heatmap_file, dpi=450, bbox_inches="tight")
        plt.close(heatmap_plot.fig)
    else:
        plt.figure(figsize=(15, 15))
        sns.heatmap(current_matrix, cmap="Spectral")
        plt.title(f"Heatmap without clustering ({n_lines:,} lines > {MAX_CLUSTER_SIZE})")
        plt.savefig(heatmap_file, dpi=450, bbox_inches="tight")
        plt.close()

    heatmap_src = f"/download?filename={urllib.parse.quote(str(heatmap_file))}&type=image"
    heatmap_download = f"/download?filename={urllib.parse.quote(str(heatmap_file))}&type=png_download"
    heatmap_style = {**CUSTOM_CSS["button"], "display": "inline-block"}
    elapsed = time.time() - start_time
    status = f"Generated {n_lines:,} × {n_lines:,} matrix using {expansion_label} in {elapsed:.2f} seconds."
    store = {"path": str(matrix_file), "lines": current_matrix.index.tolist()}
    return heatmap_src, full_matrix_link, heatmap_download, heatmap_style, store, status


@app.callback(Output("subset-dropdown", "options"), Input("matrix-store", "data"))
def update_subset_dropdown_options(matrix_data):
    if not matrix_data:
        return []
    return [{"label": line, "value": line} for line in matrix_data.get("lines", [])]


@app.callback(
    Output("subset-matrix-display", "children"),
    Input("subset-dropdown", "value"),
    State("matrix-store", "data"),
)
def display_subset_matrix(subset_values, matrix_data):
    if not subset_values or not matrix_data:
        return ""
    matrix_path = matrix_data.get("path")
    if not matrix_path or not os.path.exists(matrix_path):
        return "Matrix file not found. Regenerate the matrix."
    full_matrix = pd.read_csv(matrix_path, index_col=0)
    subset_values = [line for line in subset_values if line in full_matrix.index]
    if not subset_values:
        return "No selected subset lines are in the current matrix."
    subset_matrix = full_matrix.loc[subset_values, subset_values]
    rounded = subset_matrix.round(4)
    header = html.Thead(html.Tr([html.Th("")] + [html.Th(col) for col in rounded.columns]))
    body = html.Tbody(
        [
            html.Tr([html.Td(rounded.index[i])] + [html.Td(rounded.iloc[i, j]) for j in range(len(rounded.columns))])
            for i in range(len(rounded))
        ]
    )
    return html.Table([header, body], className="table table-striped table-sm")


@app.callback(
    Output("download-subset-link", "href"),
    Input("subset-dropdown", "value"),
    State("matrix-store", "data"),
)
def make_subset_download(selected_lines, matrix_data):
    if not selected_lines or not matrix_data:
        raise PreventUpdate
    matrix_path = matrix_data.get("path")
    if not matrix_path or not os.path.exists(matrix_path):
        raise PreventUpdate
    full_matrix = pd.read_csv(matrix_path, index_col=0)
    selected_lines = [line for line in selected_lines if line in full_matrix.index]
    if not selected_lines:
        raise PreventUpdate
    subset = full_matrix.loc[selected_lines, selected_lines]
    subset_file = OUTPUT_DIR / f"subset_matrix_{int(time.time())}.csv"
    subset.to_csv(subset_file)
    return f"/download?filename={urllib.parse.quote(str(subset_file))}&type=subset"

# =============================================================================
# Add pedigree entries callbacks
# =============================================================================


def split_valid_invalid_new_rows(new_rows: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Split uploaded rows into valid rows and rows with unrecognized parents."""
    existing = set(df["LineName"].astype(str).tolist())
    incoming = set(new_rows["LineName"].astype(str).tolist())
    allowed_parents = existing.union(incoming)

    invalid: list[dict] = []
    valid: list[dict] = []
    for _, row in new_rows.iterrows():
        mp = clean_parent_value(row.get("MaleParent"))
        fp = clean_parent_value(row.get("FemaleParent"))
        male_ok = mp is None or mp in allowed_parents
        female_ok = fp is None or fp in allowed_parents
        if male_ok and female_ok:
            valid.append(row.to_dict())
        else:
            invalid.append(row.to_dict())
    return valid, invalid


@app.callback(
    [
        Output("add-ped-upload-status", "children"),
        Output("add-ped-missing-rows-store", "data"),
        Output("add-ped-valid-rows-store", "data"),
    ],
    [Input("add-ped-upload", "contents"), Input("add-ped-save-corrections-btn", "n_clicks")],
    [
        State("add-ped-upload", "filename"),
        State({"type": "missing-parent-male", "index": ALL}, "value"),
        State({"type": "missing-parent-female", "index": ALL}, "value"),
        State("add-ped-missing-rows-store", "data"),
        State("add-ped-valid-rows-store", "data"),
    ],
    prevent_initial_call=True,
)
def manage_additional_pedigree(contents, save_clicks, filename, male_values, female_values, invalid_rows, valid_rows):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "add-ped-upload":
        if contents is None:
            raise PreventUpdate
        try:
            _, content_string = contents.split(",", 1)
            decoded = base64.b64decode(content_string)
            if filename and filename.lower().endswith(".csv"):
                new_rows = pd.read_csv(BytesIO(decoded))
            else:
                new_rows = pd.read_csv(BytesIO(decoded), sep="\t")
            new_rows = normalize_pedigree_df(new_rows)
        except Exception as exc:
            return f"Error reading file: {exc}", [], []

        valid, invalid = split_valid_invalid_new_rows(new_rows)
        if valid and not invalid:
            combined = pd.concat([df, pd.DataFrame(valid)], ignore_index=True)
            msg = set_active_dataframe(combined, persist=True)
            return f"Uploaded {filename}. Added {len(valid):,} valid rows. {msg}", [], []

        return (
            f"Uploaded {filename}. Valid rows waiting: {len(valid):,}; rows needing correction: {len(invalid):,}.",
            invalid,
            valid,
        )

    if trigger == "add-ped-save-corrections-btn":
        invalid_rows = invalid_rows or []
        valid_rows = valid_rows or []
        corrected = []
        for i, row in enumerate(invalid_rows):
            row = dict(row)
            row["MaleParent"] = male_values[i] if i < len(male_values) and male_values[i] else ""
            row["FemaleParent"] = female_values[i] if i < len(female_values) and female_values[i] else ""
            corrected.append(row)

        rows_to_add = valid_rows + corrected
        if not rows_to_add:
            return "No rows to save.", [], []

        combined = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
        msg = set_active_dataframe(combined, persist=True)
        return f"Saved {len(rows_to_add):,} new/corrected rows. {msg}", [], []

    raise PreventUpdate


@app.callback(Output("add-ped-missing-parents-table", "children"), Input("add-ped-missing-rows-store", "data"))
def show_missing_parents_table(invalid_rows):
    save_button = html.Button("Save Corrections", id="add-ped-save-corrections-btn", style=CUSTOM_CSS["button"])
    if not invalid_rows:
        return html.Div(["No invalid rows to correct.", html.Br(), html.Div(save_button, style={"display": "none"})])

    existing_lines = sorted(df["LineName"].astype(str).unique().tolist())
    options = [{"label": x, "value": x} for x in existing_lines]
    rows_html = []
    for i, row in enumerate(invalid_rows):
        row_ln = row.get("LineName", "")
        row_mp = clean_parent_value(row.get("MaleParent")) or ""
        row_fp = clean_parent_value(row.get("FemaleParent")) or ""
        male_known = row_mp in existing_lines if row_mp else True
        female_known = row_fp in existing_lines if row_fp else True

        male_cell = [
            dcc.Dropdown(
                id={"type": "missing-parent-male", "index": i},
                options=options,
                value=row_mp if male_known else None,
                placeholder="Select/correct Male Parent",
                style={"width": "260px"},
            )
        ]
        if row_mp and not male_known:
            male_cell.append(html.Span(f" Invalid: {row_mp}", style={"color": "red", "marginLeft": "8px"}))

        female_cell = [
            dcc.Dropdown(
                id={"type": "missing-parent-female", "index": i},
                options=options,
                value=row_fp if female_known else None,
                placeholder="Select/correct Female Parent",
                style={"width": "260px"},
            )
        ]
        if row_fp and not female_known:
            female_cell.append(html.Span(f" Invalid: {row_fp}", style={"color": "red", "marginLeft": "8px"}))

        rows_html.append(html.Tr([html.Td(row_ln), html.Td(male_cell), html.Td(female_cell)]))

    table = html.Table(
        [html.Thead(html.Tr([html.Th("LineName"), html.Th("MaleParent"), html.Th("FemaleParent")])), html.Tbody(rows_html)],
        className="table table-striped",
    )
    return html.Div([table, save_button])

# =============================================================================
# Pedigree Explorer dynamic modules and callbacks
# =============================================================================


@app.callback(Output("selected-progeny-modules", "children"), Input("progeny-module-dropdown", "value"))
def display_selected_progeny_modules(selected_functions):
    if not selected_functions:
        return []
    selected_functions = selected_functions if isinstance(selected_functions, list) else [selected_functions]
    modules = []

    if "specific-pairing" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Lookup Progeny of Specific Pairing"),
                        dcc.Dropdown(id="progeny-line-dropdown", options=line_options(), placeholder="Select female parent", style=CUSTOM_CSS["dropdown"]),
                        html.Br(),
                        dcc.Dropdown(id="progeny-line-dropdown-2", options=line_options(), placeholder="Select male parent", style=CUSTOM_CSS["dropdown"]),
                        html.Button("Find Progeny", id="find-progeny-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="progeny-results"),
                    ]
                ),
                className="mb-3",
            )
        )

    if "single-parent" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Lookup Progeny of Single Parent"),
                        dcc.Dropdown(id="single-parent-dropdown", options=line_options(), placeholder="Select parent", style=CUSTOM_CSS["dropdown"]),
                        html.Button("Find Progeny", id="find-single-parent-progeny-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="single-parent-progeny-results"),
                    ]
                ),
                className="mb-3",
            )
        )

    if "family-tree" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Generate Family Tree"),
                        dcc.Dropdown(id="family-tree-dropdown", options=line_options(), placeholder="Select line", style=CUSTOM_CSS["dropdown"]),
                        html.Br(),
                        html.Label("Generation depth:"),
                        dcc.Slider(id="generation-depth-slider", min=0, max=0, step=1, marks={0: "0"}, value=0),
                        html.Br(),
                        html.Label("Highlight lineage:"),
                        dcc.RadioItems(
                            id="lineage-highlight-radio",
                            options=[
                                {"label": "None", "value": "none"},
                                {"label": "Female lineage", "value": "female"},
                                {"label": "Male lineage", "value": "male"},
                                {"label": "Both", "value": "both"},
                            ],
                            value="none",
                            labelStyle={"display": "inline-block", "marginRight": "18px"},
                        ),
                        html.Br(),
                        html.Label("Tree orientation / flip:"),
                        dcc.RadioItems(
                            id="family-tree-orientation-radio",
                            options=[
                                {"label": "Ancestors above", "value": "TB"},
                                {"label": "Ancestors below", "value": "BT"},
                                {"label": "Ancestors left", "value": "LR"},
                                {"label": "Ancestors right", "value": "RL"},
                            ],
                            value="TB",
                            labelStyle={"display": "inline-block", "marginRight": "18px"},
                        ),
                        html.Button("Generate Family Tree", id="generate-family-tree-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="family-tree-image"),
                    ]
                ),
                className="mb-3",
            )
        )

    if "descendant-tree" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Generate Descendant Tree"),
                        dcc.Dropdown(id="descendant-tree-dropdown", options=line_options(), placeholder="Select line", style=CUSTOM_CSS["dropdown"]),
                        html.Button("Generate Descendant Tree", id="generate-descendant-tree-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="descendant-tree-image"),
                    ]
                ),
                className="mb-3",
            )
        )

    if "combined-family-tree" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Generate Combined Family Tree"),
                        dcc.Dropdown(id="combined-family-tree-dropdown-1", options=line_options(), placeholder="Select first line", style=CUSTOM_CSS["dropdown"]),
                        html.Br(),
                        dcc.Dropdown(id="combined-family-tree-dropdown-2", options=line_options(), placeholder="Select second line", style=CUSTOM_CSS["dropdown"]),
                        html.Button("Generate Combined Family Tree", id="generate-combined-family-tree-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="combined-family-tree-image"),
                    ]
                ),
                className="mb-3",
            )
        )

    if "temp-progeny-tree" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Generate Family Tree with Temporary Progeny"),
                        dcc.Dropdown(id="temp-female-parent-dropdown", options=line_options(), placeholder="Select female parent", style=CUSTOM_CSS["dropdown"]),
                        html.Br(),
                        dcc.Dropdown(id="temp-male-parent-dropdown", options=line_options(), placeholder="Select male parent", style=CUSTOM_CSS["dropdown"]),
                        html.Button("Generate Temporary Progeny Tree", id="generate-temp-progeny-tree-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="temp-progeny-tree-image"),
                    ]
                ),
                className="mb-3",
            )
        )

    if "interactive-pedigree" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Interactive pedigree viewer"),
                        html.P("Zoom, drag, and inspect ancestor nodes. This requires dash-cytoscape."),
                        dcc.Dropdown(id="interactive-pedigree-dropdown", options=line_options(), placeholder="Select line", style=CUSTOM_CSS["dropdown"]),
                        html.Br(),
                        html.Label("Ancestor depth:"),
                        dcc.Slider(
                            id="interactive-pedigree-depth-slider",
                            min=1,
                            max=8,
                            step=1,
                            value=4,
                            marks={i: str(i) for i in range(1, 9)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Button("Open Interactive Viewer", id="generate-interactive-pedigree-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="interactive-pedigree-output"),
                    ]
                ),
                className="mb-3",
            )
        )

    if "generation-ring" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Pedigree fan / generation ring layout"),
                        html.P("The fan chart draws the selected genotype in the center and expands known female and male ancestry as concentric wedges, similar to circular pedigree wheels."),
                        dcc.Dropdown(id="generation-ring-dropdown", options=line_options(), placeholder="Select line", style=CUSTOM_CSS["dropdown"]),
                        html.Br(),
                        html.Label("Chart style:"),
                        dcc.RadioItems(
                            id="generation-ring-style-radio",
                            options=[
                                {"label": "Pedigree fan chart", "value": "fan"},
                                {"label": "Classic node/link rings", "value": "node-ring"},
                            ],
                            value="fan",
                            labelStyle={"display": "inline-block", "marginRight": "20px"},
                            style={"fontWeight": "bold", "marginBottom": "10px"},
                        ),
                        html.Label("Ancestor depth:"),
                        dcc.Slider(
                            id="generation-ring-depth-slider",
                            min=1,
                            max=10,
                            step=1,
                            value=5,
                            marks={i: str(i) for i in range(1, 11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Button("Generate Ring Layout", id="generate-generation-ring-button", style=CUSTOM_CSS["button"]),
                        html.A(
                            "Download large full-label visualization HTML",
                            id="download-generation-ring-link",
                            href="",
                            download="pedigree_fan_full_labels.html",
                            className="btn btn-success",
                            style={**CUSTOM_CSS["button"], "display": "none"},
                        ),
                        dcc.Loading(dcc.Graph(id="generation-ring-plot", config=HIGHRES_GRAPH_CONFIG), type="default"),
                    ]
                ),
                className="mb-3",
            )
        )

    if "group-era-insights" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Group / era / founder insights"),
                        html.P("Type one or more groups/patterns such as L01-, HoCP96-, 2001, or a regex. The app finds matching LineName values and summarizes parent usage, inferred era, and founder origins."),
                        dcc.Textarea(
                            id="group-pattern-input",
                            placeholder="Examples: L01-\nHoCP96-\n2001",
                            style={"width": "100%", "height": "90px"},
                        ),
                        html.Br(),
                        dcc.RadioItems(
                            id="group-pattern-mode",
                            options=[
                                {"label": "Contains", "value": "contains"},
                                {"label": "Starts with", "value": "startswith"},
                                {"label": "Regex", "value": "regex"},
                            ],
                            value="contains",
                            labelStyle={"display": "inline-block", "marginRight": "18px"},
                        ),
                        html.Br(),
                        html.Button("Generate Group Insights", id="generate-group-insights-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="group-insights-summary", style={"fontWeight": "bold", "marginTop": "10px"}),
                        dcc.Loading(dcc.Graph(id="group-parent-usage-plot", config=HIGHRES_GRAPH_CONFIG), type="default"),
                        dcc.Loading(dcc.Graph(id="group-era-plot", config=HIGHRES_GRAPH_CONFIG), type="default"),
                        dcc.Loading(dcc.Graph(id="group-founder-cloud-plot", config=HIGHRES_GRAPH_CONFIG), type="default"),
                        html.H5("Matched lines"),
                        html.Div(id="group-matched-lines-table", style=CUSTOM_CSS["table_wrap"]),
                        html.H5("Top founder-like ancestors"),
                        html.Div(id="group-founder-table", style=CUSTOM_CSS["table_wrap"]),
                    ]
                ),
                className="mb-3",
            )
        )

    if "ancestor-contribution" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Ancestor contribution chart"),
                        html.P("Estimate which ancestors/founders dominate a selected genotype's pedigree. Contributions are expected pedigree fractions, so duplicate ancestor paths are accumulated."),
                        dcc.Dropdown(id="ancestor-contribution-dropdown", options=line_options(), placeholder="Select line", style=CUSTOM_CSS["dropdown"]),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Max ancestor depth:"),
                                        dcc.Slider(
                                            id="ancestor-contribution-depth-slider",
                                            min=2,
                                            max=16,
                                            step=1,
                                            value=10,
                                            marks={i: str(i) for i in range(2, 17, 2)},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                    width=8,
                                ),
                                dbc.Col(
                                    dcc.Checklist(
                                        id="ancestor-contribution-founder-only",
                                        options=[{"label": "Founder-like ancestors only", "value": "founders"}],
                                        value=[],
                                    ),
                                    width=4,
                                ),
                            ]
                        ),
                        html.Button("Generate Ancestor Contribution", id="generate-ancestor-contribution-button", style=CUSTOM_CSS["button"]),
                        dcc.Loading(dcc.Graph(id="ancestor-contribution-plot", config=HIGHRES_GRAPH_CONFIG), type="default"),
                        html.Div(id="ancestor-contribution-table", style=CUSTOM_CSS["table_wrap"]),
                    ]
                ),
                className="mb-3",
            )
        )

    if "direct-line-ancestors" in selected_functions:
        modules.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Furthest direct male/female ancestors"),
                        html.P(
                            "Paste genotype names and the app will follow the direct male line "
                            "(male parent of male parent, etc.) and direct female line "
                            "(female parent of female parent, etc.) until each side terminates."
                        ),
                        dcc.Textarea(
                            id="direct-line-genotype-input",
                            placeholder="Examples:\nL01-299\nHoCP96-540\nLCP85-384",
                            style={"width": "100%", "height": "130px"},
                        ),
                        html.Br(),
                        html.Button("Find Furthest Direct Ancestors", id="generate-direct-line-ancestors-button", style=CUSTOM_CSS["button"]),
                        html.Div(id="direct-line-ancestors-summary", style={"fontWeight": "bold", "marginTop": "10px"}),
                        html.A(
                            "Download Direct-Line Ancestor Table",
                            id="download-direct-line-ancestors-link",
                            href="",
                            className="btn btn-success",
                            style={**CUSTOM_CSS["button"], "display": "none"},
                        ),
                        html.Div(id="direct-line-ancestors-table", style=CUSTOM_CSS["table_wrap"]),
                    ]
                ),
                className="mb-3",
            )
        )

    return modules


@app.callback(
    Output("progeny-results", "children"),
    Input("find-progeny-button", "n_clicks"),
    [State("progeny-line-dropdown", "value"), State("progeny-line-dropdown-2", "value")],
    prevent_initial_call=True,
)
def find_progeny(n_clicks, female_parent, male_parent):
    if not n_clicks or not female_parent or not male_parent:
        raise PreventUpdate
    matches = filtered_df[
        (filtered_df["FemaleParent"].map(clean_parent_value) == female_parent)
        & (filtered_df["MaleParent"].map(clean_parent_value) == male_parent)
    ]
    if matches.empty:
        return "No progeny found for this pairing."
    return html.Ul([html.Li(row["LineName"]) for _, row in matches.iterrows()])


@app.callback(
    Output("single-parent-progeny-results", "children"),
    Input("find-single-parent-progeny-button", "n_clicks"),
    State("single-parent-dropdown", "value"),
    prevent_initial_call=True,
)
def find_single_parent_progeny(n_clicks, parent):
    if not n_clicks or not parent:
        raise PreventUpdate
    matches = filtered_df[
        (filtered_df["FemaleParent"].map(clean_parent_value) == parent)
        | (filtered_df["MaleParent"].map(clean_parent_value) == parent)
    ]
    if matches.empty:
        return "No progeny found for this parent."

    items = []
    for _, row in matches.iterrows():
        items.append(html.Li(f"{row['LineName']} (Female: {row['FemaleParent']}; Male: {row['MaleParent']})"))
    return html.Ul(items)


@app.callback(
    [
        Output("direct-line-ancestors-summary", "children"),
        Output("direct-line-ancestors-table", "children"),
        Output("download-direct-line-ancestors-link", "href"),
        Output("download-direct-line-ancestors-link", "style"),
    ],
    Input("generate-direct-line-ancestors-button", "n_clicks"),
    State("direct-line-genotype-input", "value"),
    prevent_initial_call=True,
)
def generate_direct_line_ancestor_lookup(n_clicks, genotype_text):
    if not n_clicks:
        raise PreventUpdate

    genotypes = parse_user_genotype_list(genotype_text or "")
    hidden_style = {**CUSTOM_CSS["button"], "display": "none"}
    visible_style = {**CUSTOM_CSS["button"], "display": "inline-block"}

    if not genotypes:
        return "Paste at least one genotype name.", "", "", hidden_style

    result_df = direct_line_ancestor_table(genotypes, df)
    found_n = int(result_df["FoundInPedigree"].sum()) if not result_df.empty else 0

    file_name = f"direct_line_ancestors_{int(time.time())}.csv"
    file_path = OUTPUT_DIR / file_name
    result_df.to_csv(file_path, index=False)
    href = f"/download?filename={urllib.parse.quote(str(file_path))}&type=csv"

    summary = (
        f"Processed {len(result_df):,} genotypes; {found_n:,} were found in the active pedigree. "
        "Depth is the number of parent-to-parent steps from the input genotype to the furthest direct ancestor."
    )
    display_cols = [
        "Genotype",
        "FoundInPedigree",
        "FurthestDirectMaleAncestor",
        "DirectMaleDepth",
        "FurthestDirectFemaleAncestor",
        "DirectFemaleDepth",
        "DirectMaleStatus",
        "DirectFemaleStatus",
        "DirectMalePath",
        "DirectFemalePath",
    ]
    return summary, dataframe_to_dash_table(result_df[display_cols], max_rows=150), href, visible_style


@app.callback(
    [Output("generation-depth-slider", "max"), Output("generation-depth-slider", "marks"), Output("generation-depth-slider", "value")],
    Input("family-tree-dropdown", "value"),
)
def update_generation_depth_slider(selected_line_name):
    if not selected_line_name:
        return 0, {0: "0"}, 0
    _, _, generations = find_ancestors(selected_line_name, filtered_df)
    max_generation = max(generations.keys()) if generations else 0
    marks = {i: str(i) for i in range(max_generation + 1)}
    return max_generation, marks, max_generation


@app.callback(
    Output("family-tree-image", "children"),
    [Input("generate-family-tree-button", "n_clicks"), Input("lineage-highlight-radio", "value")],
    [
        State("family-tree-dropdown", "value"),
        State("generation-depth-slider", "value"),
        State("kinship-method-slider", "value"),
        State("family-tree-orientation-radio", "value"),
    ],
    prevent_initial_call=True,
)
def generate_family_tree(n_clicks, lineage_mode, selected_line_name, generation_depth, method_choice, orientation):
    if not n_clicks or not selected_line_name:
        raise PreventUpdate

    all_ancestors, relationships, generations = find_ancestors(selected_line_name, filtered_df)
    full_nodes = all_ancestors.union({selected_line_name})
    full_relatives_df = filtered_df[filtered_df["LineName"].isin(full_nodes)].copy()
    full_matrix = compute_selected_matrix(full_relatives_df, method_choice)

    subset_nodes: set[str] = {selected_line_name}
    for gen in range((generation_depth or 0) + 1):
        subset_nodes.update(generations.get(gen, []))
    subset_relationships = [(p, c, role) for p, c, role in relationships if p in subset_nodes and c in subset_nodes]

    kinship_values = {}
    for line in subset_nodes:
        if selected_line_name in full_matrix.index and line in full_matrix.columns:
            kinship_values[line] = float(full_matrix.loc[selected_line_name, line])
        else:
            kinship_values[line] = 0.0

    parent_count = {line: 0 for line in subset_nodes}
    for parent, child, _ in subset_relationships:
        if child in parent_count:
            parent_count[child] += 1

    founders = {line for line, count in parent_count.items() if count == 0 and line != selected_line_name}
    poly_founders = {line for line in founders if "POLY" in line.upper() or PATTERN_POLY_P.match(line)}
    half_defined = {line for line, count in parent_count.items() if count == 1}

    kin_vals = list(kinship_values.values()) or [0.0]
    if min(kin_vals) == max(kin_vals):
        norm = Normalize(vmin=min(kin_vals) - 0.01, vmax=max(kin_vals) + 0.01)
    else:
        norm = Normalize(vmin=min(kin_vals), vmax=max(kin_vals))
    cmap = cm.get_cmap("Spectral")

    def get_node_color(kval):
        color = cmap(norm(kval))
        return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    female_line = get_direct_line(selected_line_name, filtered_df, "FemaleParent")
    male_line = get_direct_line(selected_line_name, filtered_df, "MaleParent")

    dot = graphviz.Digraph(comment="Family Tree")
    rankdir = orientation if orientation in {"BT", "TB", "LR", "RL"} else "BT"
    dot.attr("graph", rankdir=rankdir, splines="ortho", nodesep="0.35", ranksep="0.6")
    dot.attr("node", shape="ellipse", style="filled", fontsize="20", fontname="Helvetica")
    dot.attr("edge", arrowsize="0.8")

    for line in sorted(subset_nodes):
        kv = kinship_values.get(line, 0.0)
        label = f"{line}\nKinship: {kv:.2f}"
        fill = get_node_color(kv)
        base_style = "filled"
        penwidth = "1.5"
        border_color = "black"

        if line in poly_founders:
            base_style = normalize_style(base_style, "dashed")
        elif line in half_defined:
            base_style = normalize_style(base_style, "dotted")
            penwidth = "2"

        # This is the important fixed section: highlight is additive and does not get
        # overwritten by founder / half-defined styling.
        is_female_highlight = lineage_mode in {"female", "both"} and line in female_line
        is_male_highlight = lineage_mode in {"male", "both"} and line in male_line
        if is_female_highlight or is_male_highlight:
            base_style = normalize_style(base_style, "bold")
            penwidth = "5"
            if is_female_highlight and is_male_highlight:
                border_color = "purple"
            elif is_female_highlight:
                border_color = "red"
            else:
                border_color = "blue"

        if line == selected_line_name:
            base_style = normalize_style(base_style, "bold")
            penwidth = "5"
            border_color = "green"

        dot.node(line, label=label, fillcolor=fill, fontcolor="black", color=border_color, style=base_style, penwidth=penwidth)

    for parent, child, role in subset_relationships:
        edge_color = "blue" if role == "male" else "red" if role == "female" else "black"
        dot.edge(parent, child, color=edge_color)

    summary = (
        "Pedigree Completeness:\n"
        f"Founders: {len(founders)}\n"
        f"Polycross founders: {len(poly_founders)}\n"
        f"Half-defined genotypes: {len(half_defined)}\n"
        f"Displayed nodes: {len(subset_nodes)}"
    )
    dot.node("pedigree_summary", label=summary, shape="box", style="filled", fillcolor="white", fontsize="18")
    dot.edge("pedigree_summary", selected_line_name, style="invis")

    return graphviz_visualization_block(dot, f"family_tree_{selected_line_name}_{rankdir}")


@app.callback(
    Output("descendant-tree-image", "children"),
    Input("generate-descendant-tree-button", "n_clicks"),
    State("descendant-tree-dropdown", "value"),
    prevent_initial_call=True,
)
def generate_descendant_tree(n_clicks, selected_line_name):
    if not n_clicks or not selected_line_name:
        raise PreventUpdate
    descendants, relationships, generations = find_descendants(selected_line_name, filtered_df)
    nodes = descendants.union({selected_line_name})
    dot = graphviz.Digraph(comment="Descendant Tree")
    dot.attr("graph", rankdir="TB", splines="ortho", nodesep="0.35", ranksep="0.6")
    dot.attr("node", shape="ellipse", style="filled", fontsize="20", fontname="Helvetica")

    for node in sorted(nodes):
        fill = "green" if node == selected_line_name else "lightgrey"
        dot.node(node, label=node, fillcolor=fill, fontcolor="black", color="black")
    for parent, child, role in relationships:
        if parent in nodes and child in nodes:
            edge_color = "blue" if role == "male" else "red" if role == "female" else "black"
            dot.edge(parent, child, color=edge_color)
    return graphviz_visualization_block(dot, f"descendant_tree_{selected_line_name}")


@app.callback(
    Output("combined-family-tree-image", "children"),
    Input("generate-combined-family-tree-button", "n_clicks"),
    [State("combined-family-tree-dropdown-1", "value"), State("combined-family-tree-dropdown-2", "value")],
    prevent_initial_call=True,
)
def generate_combined_family_tree(n_clicks, line1, line2):
    if not n_clicks or not line1 or not line2:
        raise PreventUpdate

    ancestors1, relationships1, _ = find_ancestors(line1, filtered_df)
    ancestors2, relationships2, _ = find_ancestors(line2, filtered_df)
    all_lines = ancestors1.union(ancestors2, {line1, line2})
    all_relationships = relationships1 + relationships2

    line_colors = {line: "#ADD8E6" for line in ancestors1}
    for line in ancestors2:
        line_colors[line] = "#FFFFE0" if line in line_colors else "#FFB6C1"
    line_colors[line1] = "green"
    line_colors[line2] = "green"

    dot = graphviz.Digraph(comment="Combined Family Tree")
    dot.attr("graph", rankdir="BT", splines="ortho", nodesep="0.35", ranksep="0.6")
    dot.attr("node", shape="ellipse", style="filled", fontsize="20", fontname="Helvetica")
    for line in sorted(all_lines):
        dot.node(line, label=line, fillcolor=line_colors.get(line, "lightgrey"), fontcolor="black", color="black")

    added_edges = set()
    for parent, child, role in all_relationships:
        if parent not in all_lines or child not in all_lines or (parent, child) in added_edges:
            continue
        if parent in ancestors1 and child in ancestors1:
            color = "#2980b9"
        elif parent in ancestors2 and child in ancestors2:
            color = "#c0392b"
        else:
            color = "#777777"
        dot.edge(parent, child, color=color)
        added_edges.add((parent, child))

    return graphviz_visualization_block(dot, f"combined_family_tree_{line1}_{line2}")


@app.callback(
    Output("temp-progeny-tree-image", "children"),
    Input("generate-temp-progeny-tree-button", "n_clicks"),
    [State("temp-female-parent-dropdown", "value"), State("temp-male-parent-dropdown", "value"), State("kinship-method-slider", "value")],
    prevent_initial_call=True,
)
def generate_temp_progeny_tree(n_clicks, female_parent, male_parent, method_choice):
    if not n_clicks or not female_parent or not male_parent:
        raise PreventUpdate

    temp_progeny_name = f"Temp_Progeny_{int(time.time())}"
    temp_row = pd.DataFrame(
        [{"LineName": temp_progeny_name, "FemaleParent": female_parent, "MaleParent": male_parent}]
    )
    temp_df = pd.concat([filtered_df, temp_row], ignore_index=True)

    ancestors, relationships, _ = find_ancestors(temp_progeny_name, temp_df)
    all_lines = ancestors.union({temp_progeny_name})
    relatives_df = temp_df[temp_df["LineName"].isin(all_lines)].copy()
    kinship_matrix = compute_selected_matrix(relatives_df, method_choice)

    kinship_values = {}
    for line in all_lines:
        if temp_progeny_name in kinship_matrix.index and line in kinship_matrix.columns:
            kinship_values[line] = float(kinship_matrix.loc[temp_progeny_name, line])
        else:
            kinship_values[line] = 0.0
    non_self_values = [v for k, v in kinship_values.items() if k != temp_progeny_name]
    q25, q50 = np.quantile(non_self_values, [0.25, 0.5]) if non_self_values else (0, 0)

    def kin_color(value):
        if value <= q25:
            return "#ADD8E6"
        if value <= q50:
            return "#FFB6C1"
        return "#FFFFE0"

    dot = graphviz.Digraph(comment="Family Tree with Temporary Progeny")
    dot.attr("graph", rankdir="BT", splines="ortho", nodesep="0.35", ranksep="0.6")
    dot.attr("node", shape="ellipse", style="filled", fontsize="20", fontname="Helvetica")

    for line in sorted(all_lines):
        kv = kinship_values.get(line, 0.0)
        fill = "orange" if line == temp_progeny_name else kin_color(kv)
        penwidth = "5" if line == temp_progeny_name else "1.5"
        dot.node(line, label=f"{line}\nKinship: {kv:.2f}", fillcolor=fill, fontcolor="black", color="black", penwidth=penwidth)

    for parent, child, role in relationships:
        if parent in all_lines and child in all_lines:
            edge_color = "blue" if role == "male" else "red" if role == "female" else "black"
            dot.edge(parent, child, color=edge_color)
    return graphviz_visualization_block(dot, f"temporary_progeny_tree_{female_parent}_{male_parent}")


# =============================================================================
# New visualization callbacks: matrix ordination, group insights, ancestor contribution,
# Cytoscape interactive viewer, and generation-ring layout
# =============================================================================


@app.callback(
    Output("matrix-ordination-plot", "figure"),
    [Input("matrix-store", "data"), Input("matrix-ordination-method", "value"), Input("matrix-ordination-labels", "value")],
)
def update_matrix_ordination_plot(matrix_data, method, label_values):
    if not matrix_data or not matrix_data.get("path"):
        fig = go.Figure()
        fig.update_layout(title="Generate a kinship matrix first to view PCA/MDS ordination.", height=420)
        return fig
    matrix_path = matrix_data.get("path")
    if not matrix_path or not os.path.exists(matrix_path):
        fig = go.Figure()
        fig.update_layout(title="Matrix file not found. Regenerate the matrix.", height=420)
        return fig
    try:
        matrix_df = pd.read_csv(matrix_path, index_col=0)
        ord_df = matrix_to_ordination(matrix_df, method or "mds")
        show_labels = bool(label_values and "labels" in label_values)
        return build_ordination_figure(ord_df, method or "mds", show_labels)
    except Exception as exc:
        fig = go.Figure()
        fig.update_layout(title=f"Could not compute ordination: {exc}", height=420)
        return fig


@app.callback(
    [
        Output("group-insights-summary", "children"),
        Output("group-parent-usage-plot", "figure"),
        Output("group-era-plot", "figure"),
        Output("group-founder-cloud-plot", "figure"),
        Output("group-matched-lines-table", "children"),
        Output("group-founder-table", "children"),
    ],
    Input("generate-group-insights-button", "n_clicks"),
    [State("group-pattern-input", "value"), State("group-pattern-mode", "value")],
    prevent_initial_call=True,
)
def generate_group_insights(n_clicks, pattern_text, match_mode):
    if not n_clicks:
        raise PreventUpdate
    if not pattern_text or not pattern_text.strip():
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Enter a group/pattern first", height=420)
        return "Enter at least one group/pattern.", empty_fig, empty_fig, empty_fig, "", ""

    matched, parent_counts, timeline_counts, founders_df, matched_df = build_group_insights(pattern_text, match_mode or "contains")
    if not matched:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No matching lines found", height=420)
        return "No matching LineName values found.", empty_fig, empty_fig, empty_fig, "", ""

    # Collapse parent counts across role for a simpler most-used-parent plot.
    if parent_counts.empty:
        parent_plot_df = pd.DataFrame(columns=["Parent", "Uses"])
    else:
        parent_plot_df = parent_counts.groupby("Parent", as_index=False).agg(Uses=("Uses", "sum")).sort_values("Uses", ascending=False)

    parent_fig = horizontal_bar_figure(parent_plot_df, "Uses", "Parent", "Most-used direct parents in matched group", top_n=30)
    era_fig = timeline_figure(timeline_counts)
    cloud_fig = founder_cloud_figure(founders_df, top_n=50)

    known_years = matched_df["InferredYear"].notna().sum() if not matched_df.empty else 0
    summary = f"Matched {len(matched):,} lines. Inferred years/series for {known_years:,}. Found {len(parent_plot_df):,} direct parents and {len(founders_df):,} founder-like ancestors."
    return (
        summary,
        parent_fig,
        era_fig,
        cloud_fig,
        dataframe_to_dash_table(matched_df[["LineName", "FemaleParent", "MaleParent", "InferredYear", "Era"]].sort_values("LineName"), max_rows=80),
        dataframe_to_dash_table(founders_df.head(80), max_rows=80),
    )


@app.callback(
    [Output("ancestor-contribution-plot", "figure"), Output("ancestor-contribution-table", "children")],
    Input("generate-ancestor-contribution-button", "n_clicks"),
    [
        State("ancestor-contribution-dropdown", "value"),
        State("ancestor-contribution-depth-slider", "value"),
        State("ancestor-contribution-founder-only", "value"),
    ],
    prevent_initial_call=True,
)
def generate_ancestor_contribution(n_clicks, selected_line, max_depth, founder_only_values):
    if not n_clicks or not selected_line:
        raise PreventUpdate
    contrib_df = compute_ancestor_contributions(selected_line, filtered_df, max_depth=max_depth or 10)
    if founder_only_values and "founders" in founder_only_values:
        contrib_df = contrib_df[contrib_df["FounderLike"]].copy()
    if contrib_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No ancestors found for {selected_line}", height=420)
        return fig, "No ancestors found."

    plot_df = contrib_df.head(30).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["Percent"],
            y=plot_df["Ancestor"],
            orientation="h",
            text=plot_df["Percent"].map(lambda x: f"{x:.1f}%"),
            textposition="auto",
            customdata=np.stack([plot_df["PathCount"], plot_df["Era"], plot_df["FounderLike"]], axis=-1),
            hovertemplate="Ancestor: %{y}<br>Expected contribution: %{x:.2f}%<br>Path count: %{customdata[0]}<br>Era: %{customdata[1]}<br>Founder-like: %{customdata[2]}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top expected ancestor contributions for {selected_line}",
        xaxis_title="Expected pedigree contribution (%)",
        yaxis_title="",
        height=max(520, 24 * len(plot_df) + 140),
        margin={"l": 170, "r": 20, "t": 70, "b": 40},
    )
    table = dataframe_to_dash_table(
        contrib_df.assign(Percent=contrib_df["Percent"].map(lambda x: round(x, 3))).head(100),
        max_rows=100,
    )
    return fig, table


@app.callback(
    [
        Output("generation-ring-plot", "figure"),
        Output("download-generation-ring-link", "href"),
        Output("download-generation-ring-link", "style"),
    ],
    Input("generate-generation-ring-button", "n_clicks"),
    [
        State("generation-ring-dropdown", "value"),
        State("generation-ring-depth-slider", "value"),
        State("generation-ring-style-radio", "value"),
    ],
    prevent_initial_call=True,
)
def generate_generation_ring(n_clicks, selected_line, max_depth, ring_style):
    if not n_clicks or not selected_line:
        raise PreventUpdate
    depth = max_depth or 5
    style = ring_style or "fan"
    fig = generation_ring_figure(selected_line, depth, style)
    export_fig = generation_ring_figure(selected_line, depth, style, large_export=True)
    export_path = save_plotly_html(export_fig, f"generation_ring_{style}_{selected_line}_{depth}_full_labels")
    link_style = {**CUSTOM_CSS["button"], "display": "inline-block"}
    return fig, download_href(export_path, "html"), link_style


@app.callback(
    Output("interactive-pedigree-output", "children"),
    Input("generate-interactive-pedigree-button", "n_clicks"),
    [State("interactive-pedigree-dropdown", "value"), State("interactive-pedigree-depth-slider", "value")],
    prevent_initial_call=True,
)
def generate_interactive_pedigree(n_clicks, selected_line, max_depth):
    if not n_clicks or not selected_line:
        raise PreventUpdate
    if not HAS_CYTOSCAPE:
        return dbc.Alert(
            [
                html.Div("dash-cytoscape is not installed, so the interactive viewer cannot render."),
                html.Code("pip install dash-cytoscape"),
            ],
            color="warning",
            style={"marginTop": "12px"},
        )

    elements = cytoscape_elements_for_line(selected_line, max_depth or 4)
    stylesheet = [
        {"selector": "node", "style": {"content": "data(label)", "font-size": "12px", "text-valign": "center", "text-halign": "center", "background-color": "#BFD7EA", "border-width": 1, "border-color": "#2c3e50", "width": 48, "height": 48}},
        {"selector": ".selected", "style": {"background-color": "#2ECC71", "border-width": 4, "width": 70, "height": 70}},
        {"selector": ".founder", "style": {"background-color": "#F8E9A1", "border-style": "dashed"}},
        {"selector": "edge", "style": {"curve-style": "bezier", "target-arrow-shape": "triangle", "width": 2, "line-color": "#999", "target-arrow-color": "#999"}},
        {"selector": ".male", "style": {"line-color": "#3498DB", "target-arrow-color": "#3498DB"}},
        {"selector": ".female", "style": {"line-color": "#E74C3C", "target-arrow-color": "#E74C3C"}},
    ]
    return html.Div(
        [
            html.Div(
                f"Showing {sum(1 for e in elements if 'source' not in e.get('data', {})):,} nodes and {sum(1 for e in elements if 'source' in e.get('data', {})):,} edges. Drag nodes, scroll to zoom, and hover/click nodes in the viewer.",
                style={"fontWeight": "bold", "marginTop": "10px", "marginBottom": "10px"},
            ),
            cyto.Cytoscape(
                id="interactive-pedigree-cytoscape",
                elements=elements,
                stylesheet=stylesheet,
                layout={"name": "breadthfirst", "directed": True, "roots": f"#{selected_line}", "spacingFactor": 1.3},
                style={"width": "100%", "height": "760px", "border": "1px solid #ccc", "backgroundColor": "white"},
                userZoomingEnabled=True,
                userPanningEnabled=True,
                boxSelectionEnabled=True,
            ),
        ]
    )


# =============================================================================
# Download route
# =============================================================================


@app.server.route("/download")
def download_file():
    filename_quoted = flask.request.args.get("filename")
    file_type = flask.request.args.get("type", "full")
    if not filename_quoted:
        return "File not found", 404

    filename = urllib.parse.unquote(filename_quoted)
    if not os.path.exists(filename):
        return "File not found", 404

    suffix = Path(filename).suffix.lower()
    if file_type == "image":
        return send_file(filename, mimetype="image/png", as_attachment=False)

    if file_type == "png_download" or suffix == ".png":
        return send_file(filename, mimetype="image/png", as_attachment=True, download_name=os.path.basename(filename))
    if file_type == "svg" or suffix == ".svg":
        return send_file(filename, mimetype="image/svg+xml", as_attachment=True, download_name=os.path.basename(filename))
    if file_type == "pdf" or suffix == ".pdf":
        return send_file(filename, mimetype="application/pdf", as_attachment=True, download_name=os.path.basename(filename))
    if file_type == "html" or suffix in {".html", ".htm"}:
        return send_file(filename, mimetype="text/html", as_attachment=True, download_name=os.path.basename(filename))

    if file_type == "subset":
        download_name = "subset_matrix.csv"
    elif file_type == "full":
        download_name = "full_matrix.csv"
    else:
        download_name = os.path.basename(filename) or "download.csv"
    return send_file(filename, mimetype="text/csv", as_attachment=True, download_name=download_name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
