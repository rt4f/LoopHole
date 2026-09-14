#!/usr/bin/env python3
"""Build the LoopHole Engineering Handbook PDFs from Markdown sources.

The Markdown in ``docs/handbook/source/`` is the source of truth; the PDFs in
``docs/handbook/`` are generated from it. Edit the Markdown, then rebuild:

    pip install reportlab
    python docs/handbook/build_handbook.py            # all parts
    python docs/handbook/build_handbook.py 03         # only parts whose file starts with 03

Supported Markdown subset: front matter (``---`` block with part/title/subtitle/
audience), ``#``..``####`` headings, paragraphs, ``-``/``1.`` lists (nested by two
spaces), fenced code blocks, pipe tables, ``> [!NOTE|WARN|CRIT|OK] Title`` callouts,
``---`` rules, inline ``code``/**bold**/*italic*, and the directives
``::diagram <name>`` and ``::pagebreak``.
"""

from __future__ import annotations

import datetime
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source"
OUT_DIR = HERE

# ---------------------------------------------------------------------------
# Fonts (Windows first, then common Linux/macOS locations)
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = {
    "Body": ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"],
    "Body-Bold": ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"],
    "Body-Italic": ["C:/Windows/Fonts/ariali.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", "/Library/Fonts/Arial Italic.ttf"],
    "Body-BoldItalic": ["C:/Windows/Fonts/arialbi.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf", "/Library/Fonts/Arial Bold Italic.ttf"],
    "Mono": ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/Library/Fonts/Courier New.ttf"],
    "Mono-Bold": ["C:/Windows/Fonts/consolab.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/Library/Fonts/Courier New Bold.ttf"],
    "Mono-Italic": ["C:/Windows/Fonts/consolai.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf", "/Library/Fonts/Courier New Italic.ttf"],
    "Mono-BoldItalic": ["C:/Windows/Fonts/consolaz.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf", "/Library/Fonts/Courier New Bold Italic.ttf"],
}


def _register_fonts() -> None:
    for name, paths in _FONT_CANDIDATES.items():
        for path in paths:
            if Path(path).exists():
                pdfmetrics.registerFont(TTFont(name, path))
                break
        else:
            raise SystemExit(
                f"No TrueType font found for {name}. Install DejaVu fonts or edit _FONT_CANDIDATES."
            )
    registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")
    registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono-Italic", boldItalic="Mono-BoldItalic")


# ---------------------------------------------------------------------------
# Palette and styles
# ---------------------------------------------------------------------------

NAVY = colors.HexColor("#1F3A5F")
ACCENT = colors.HexColor("#2E6DA4")
INK = colors.HexColor("#1D2733")
MUTED = colors.HexColor("#5B6775")
RULE = colors.HexColor("#D0D7DE")
CODE_BG = colors.HexColor("#F5F7F9")
ZEBRA = colors.HexColor("#F7F9FB")

CALLOUTS = {
    "NOTE": (colors.HexColor("#175CD3"), colors.HexColor("#EFF6FF"), "Note"),
    "WARN": (colors.HexColor("#B54708"), colors.HexColor("#FFF8EB"), "Warning"),
    "CRIT": (colors.HexColor("#B42318"), colors.HexColor("#FEF3F2"), "Critical"),
    "OK": (colors.HexColor("#067647"), colors.HexColor("#EEFBF3"), "Good to know"),
}

SEVERITY_COLORS = {
    "critical": "#B42318", "p0": "#B42318",
    "high": "#C4320A", "p1": "#C4320A",
    "medium": "#B54708", "p2": "#B54708",
    "low": "#067647", "p3": "#067647",
}

PAGE_W, PAGE_H = A4
MARGIN_L = MARGIN_R = 20 * mm
MARGIN_T = 22 * mm
MARGIN_B = 20 * mm
FRAME_W = PAGE_W - MARGIN_L - MARGIN_R


def _styles() -> Dict[str, ParagraphStyle]:
    s: Dict[str, ParagraphStyle] = {}
    s["body"] = ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14, textColor=INK, spaceAfter=5)
    s["h1"] = ParagraphStyle("h1", fontName="Body-Bold", fontSize=20, leading=25, textColor=NAVY, spaceAfter=10, keepWithNext=1)
    s["h2"] = ParagraphStyle("h2", fontName="Body-Bold", fontSize=14.5, leading=19, textColor=NAVY, spaceBefore=14, spaceAfter=6, keepWithNext=1)
    s["h3"] = ParagraphStyle("h3", fontName="Body-Bold", fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=4, keepWithNext=1)
    s["h4"] = ParagraphStyle("h4", fontName="Body-Bold", fontSize=10, leading=13.5, textColor=INK, spaceBefore=7, spaceAfter=3, keepWithNext=1)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], spaceAfter=2.5)
    s["cell"] = ParagraphStyle("cell", fontName="Body", fontSize=8.2, leading=10.6, textColor=INK)
    s["cellhead"] = ParagraphStyle("cellhead", fontName="Body-Bold", fontSize=8.2, leading=10.6, textColor=colors.white)
    s["caption"] = ParagraphStyle("caption", fontName="Body-Italic", fontSize=8.2, leading=11, textColor=MUTED, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
    s["callout_title"] = ParagraphStyle("callout_title", fontName="Body-Bold", fontSize=9.4, leading=13, textColor=INK, spaceAfter=2)
    s["callout"] = ParagraphStyle("callout", parent=s["body"], fontSize=9.1, leading=13, spaceAfter=3)
    s["toc_title"] = ParagraphStyle("toc_title", parent=s["h1"])
    return s


STYLES: Dict[str, ParagraphStyle] = {}

# ---------------------------------------------------------------------------
# Inline Markdown -> ReportLab mini-HTML
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str, code_size: float = 8.7) -> str:
    codes: List[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = _esc(text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda m: f'<link href="{m.group(2)}" color="#2E6DA4"><u>{m.group(1)}</u></link>',
        text,
    )
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<![\s*])\*(?![\w*])", r"<i>\1</i>", text)

    def unstash(m: re.Match) -> str:
        return f'<font face="Mono" size="{code_size}" color="#8A1C3C">{_esc(codes[int(m.group(1))])}</font>'

    return re.sub(r"\x00(\d+)\x00", unstash, text)


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------


class Heading(Paragraph):
    """Paragraph that registers itself in the TOC and PDF outline."""

    def __init__(self, text: str, style: ParagraphStyle, level: int, plain: str) -> None:
        super().__init__(text, style)
        self.toc_level = level
        self.plain = plain


def _code_block(lines: List[str]) -> Table:
    wrapped: List[str] = []
    width = 104
    for line in lines:
        line = line.expandtabs(4)
        while len(line) > width:
            wrapped.append(line[:width])
            line = "  \u21aa " + line[width:]
        wrapped.append(line)
    if not wrapped:
        wrapped = [""]
    table = Table([[w] for w in wrapped], colWidths=[FRAME_W], hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Mono", 7.6),
        ("LEADING", (0, 0), (-1, -1), 9.3),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 0.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.4),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 5),
        ("LINEBEFORE", (0, 0), (0, -1), 2, RULE),
    ]
    table.setStyle(TableStyle(style))
    table.spaceBefore = 3
    table.spaceAfter = 8
    return table


def _split_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [c.strip().replace("\\|", "|") for c in cells]


def _plain_len(text: str) -> int:
    return len(re.sub(r"[`*]", "", text))


def _table(rows: List[List[str]]) -> Table:
    header, body = rows[0], rows[1:]
    ncols = len(header)
    body = [r + [""] * (ncols - len(r)) if len(r) < ncols else r[:ncols] for r in body]

    # Column widths proportional to (capped) content length, with a floor.
    weights = []
    for c in range(ncols):
        longest = max([_plain_len(header[c])] + [_plain_len(r[c]) for r in body])
        avg = sum(_plain_len(r[c]) for r in body) / max(1, len(body))
        weights.append(max(6.0, min(55.0, 0.55 * longest + 0.45 * avg)))
    total = sum(weights)
    widths = [FRAME_W * w / total for w in weights]
    floor = 17 * mm
    for _ in range(4):
        deficit = sum(floor - w for w in widths if w < floor)
        if deficit <= 0:
            break
        big = [i for i, w in enumerate(widths) if w > floor]
        big_total = sum(widths[i] for i in big)
        widths = [floor if w < floor else w - deficit * (w / big_total) for w in widths]

    def cell(text: str, head: bool = False) -> Paragraph:
        sev = SEVERITY_COLORS.get(text.strip().lower())
        if sev and not head:
            return Paragraph(f'<font color="{sev}"><b>{_esc(text.strip())}</b></font>', STYLES["cell"])
        return Paragraph(inline(text, 7.7), STYLES["cellhead" if head else "cell"])

    data = [[cell(h, True) for h in header]] + [[cell(v) for v in r] for r in body]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    table.setStyle(TableStyle(style))
    table.spaceBefore = 4
    table.spaceAfter = 9
    return table


def _callout(kind: str, title: str, lines: List[str]) -> Table:
    edge, bg, default_title = CALLOUTS.get(kind, CALLOUTS["NOTE"])
    flow = [Paragraph(inline(title or default_title), STYLES["callout_title"])]
    para: List[str] = []
    for line in lines + [""]:
        stripped = line.strip()
        if not stripped:
            if para:
                flow.append(Paragraph(inline(" ".join(para)), STYLES["callout"]))
                para = []
            continue
        if stripped.startswith(("- ", "* ")):
            if para:
                flow.append(Paragraph(inline(" ".join(para)), STYLES["callout"]))
                para = []
            flow.append(Paragraph(inline(stripped[2:]), STYLES["callout"], bulletText="\u2022"))
            continue
        para.append(stripped)
    table = Table([[flow]], colWidths=[FRAME_W], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    table.spaceBefore = 4
    table.spaceAfter = 9
    return table


# ---------------------------------------------------------------------------
# Diagrams: boxes (mm, origin top-left) and arrows between box edges
# ---------------------------------------------------------------------------

FILLS = {
    "input": "#EAF2FB", "stage": "#FFFFFF", "data": "#F3F0FF", "ext": "#F2F4F7",
    "risk": "#FEF3F2", "ok": "#EEFBF3", "warn": "#FFF8EB",
}


def make_diagram(spec: dict) -> Drawing:
    w_mm, h_mm = spec["size"]
    scale = min(1.0, (FRAME_W / mm) / w_mm)
    d = Drawing(w_mm * mm * scale, h_mm * mm * scale)
    boxes: Dict[str, Tuple[float, float, float, float]] = {}

    def to_pt(x: float, y: float) -> Tuple[float, float]:
        return x * mm * scale, (h_mm - y) * mm * scale

    for group in spec.get("groups", []):
        gx, gy, gw, gh, label = group
        x0, y1 = to_pt(gx, gy)
        d.add(Rect(x0, y1 - gh * mm * scale, gw * mm * scale, gh * mm * scale, rx=4, ry=4,
                   fillColor=None, strokeColor=colors.HexColor("#98A2B3"), strokeWidth=0.7, strokeDashArray=[3, 2]))
        d.add(String(x0 + 4, y1 - 9 * scale, label, fontName="Body-Bold", fontSize=7.4 * max(scale, 0.8), fillColor=MUTED))

    for node in spec["nodes"]:
        nid, x, y, w, h, text, kind = node
        x0, y1 = to_pt(x, y)
        bw, bh = w * mm * scale, h * mm * scale
        boxes[nid] = (x0, y1 - bh, bw, bh)
        d.add(Rect(x0, y1 - bh, bw, bh, rx=3, ry=3, fillColor=colors.HexColor(FILLS.get(kind, "#FFFFFF")),
                   strokeColor=NAVY if kind != "risk" else colors.HexColor("#B42318"), strokeWidth=0.9))
        lines = text.split("\n")
        fs = 7.4 * max(scale, 0.8)
        lead = fs * 1.25
        top = y1 - bh / 2 + (len(lines) - 1) * lead / 2 - fs / 3
        for i, line in enumerate(lines):
            d.add(String(x0 + bw / 2, top - i * lead, line, fontName="Body-Bold" if i == 0 else "Body",
                         fontSize=fs if i == 0 else fs * 0.92, fillColor=INK, textAnchor="middle"))

    for edge in spec.get("edges", []):
        src, dst = edge[0], edge[1]
        label = edge[2] if len(edge) > 2 else ""
        dashed = len(edge) > 3 and edge[3] == "dashed"
        sx, sy, sw, sh = boxes[src]
        tx, ty, tw, th = boxes[dst]
        scx, scy = sx + sw / 2, sy + sh / 2
        tcx, tcy = tx + tw / 2, ty + th / 2
        dx, dy = tcx - scx, tcy - scy
        if abs(dx) * sh > abs(dy) * sw:  # mostly horizontal
            x1, y1p = (sx + sw, scy) if dx > 0 else (sx, scy)
            x2, y2 = (tx, tcy) if dx > 0 else (tx + tw, tcy)
        else:
            x1, y1p = (scx, sy) if dy < 0 else (scx, sy + sh)
            x2, y2 = (tcx, ty + th) if dy < 0 else (tcx, ty)
        color = colors.HexColor("#475467")
        d.add(Line(x1, y1p, x2, y2, strokeColor=color, strokeWidth=0.9, strokeDashArray=[3, 2] if dashed else None))
        import math
        ang = math.atan2(y2 - y1p, x2 - x1)
        ah = 5
        d.add(Polygon([x2, y2,
                       x2 - ah * math.cos(ang - 0.4), y2 - ah * math.sin(ang - 0.4),
                       x2 - ah * math.cos(ang + 0.4), y2 - ah * math.sin(ang + 0.4)],
                      fillColor=color, strokeColor=color, strokeWidth=0.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1p + y2) / 2
            lfs = 6.6 * max(scale, 0.8)
            lw = pdfmetrics.stringWidth(label, "Body-Italic", lfs)
            d.add(Rect(mx - lw / 2 - 1.5, my - lfs * 0.4, lw + 3, lfs * 1.25, fillColor=colors.white, strokeColor=None))
            d.add(String(mx, my - lfs * 0.15, label, fontName="Body-Italic", fontSize=lfs, fillColor=MUTED, textAnchor="middle"))
    return d


# ---------------------------------------------------------------------------
# Markdown document parser
# ---------------------------------------------------------------------------


def parse_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    meta: Dict[str, str] = {}
    if text.startswith("---\n"):
        end = text.index("\n---\n", 4)
        for line in text[4:end].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[end + 5:]
    return meta, text


def build_story(body: str, diagrams: Dict[str, dict]) -> List:
    story: List = []
    lines = body.splitlines()
    i = 0
    para: List[str] = []
    first_h1 = True

    def flush_para() -> None:
        if para:
            story.append(Paragraph(inline(" ".join(p.strip() for p in para)), STYLES["body"]))
            para.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped.startswith("```"):
            flush_para()
            block: List[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i].rstrip("\n"))
                i += 1
            story.append(_code_block(block))
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para()
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                if not first_h1:
                    story.append(PageBreak())
                first_h1 = False
            elif level == 2:
                story.append(CondPageBreak(60 * mm))
            elif level == 3:
                story.append(CondPageBreak(35 * mm))
            story.append(Heading(inline(text, STYLES[f"h{level}"].fontSize * 0.92), STYLES[f"h{level}"], level - 1, re.sub(r"[`*]", "", text)))
            if level == 1:
                story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceBefore=0, spaceAfter=8))
            i += 1
            continue

        if stripped == "---":
            flush_para()
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=4, spaceAfter=8))
            i += 1
            continue

        if stripped.startswith("::pagebreak"):
            flush_para()
            story.append(PageBreak())
            i += 1
            continue

        if stripped.startswith("::diagram"):
            flush_para()
            name = stripped.split(None, 1)[1].strip()
            spec = diagrams[name]
            story.append(Spacer(1, 3))
            story.append(make_diagram(spec))
            if spec.get("caption"):
                story.append(Paragraph(inline(spec["caption"]), STYLES["caption"]))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_para()
            rows: List[List[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = _split_row(lines[i])
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in row if c):
                    rows.append(row)
                i += 1
            story.append(_table(rows))
            continue

        if stripped.startswith(">"):
            flush_para()
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            kind, title = "NOTE", ""
            head = re.match(r"^\[!(\w+)\]\s*(.*)$", block[0].strip()) if block else None
            if head:
                kind, title = head.group(1).upper(), head.group(2)
                block = block[1:]
            story.append(_callout(kind, title, block))
            continue

        bm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if bm:
            flush_para()
            indent = len(bm.group(1)) // 2
            marker = bm.group(2)
            text = bm.group(3)
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip() or re.match(r"^\s*([-*]|\d+\.)\s+", nxt) or not nxt.startswith("  "):
                    break
                if nxt.strip().startswith(("```", "|", ">", "#")):
                    break
                text += " " + nxt.strip()
                i += 1
            style = ParagraphStyle(
                f"b{indent}", parent=STYLES["bullet"], leftIndent=14 + indent * 14, bulletIndent=3 + indent * 14
            )
            bullet = "\u2022" if marker in "-*" else marker
            if indent and marker in "-*":
                bullet = "\u2013"
            story.append(Paragraph(inline(text), style, bulletText=bullet))
            continue

        para.append(line)
        i += 1

    flush_para()
    return story


# ---------------------------------------------------------------------------
# Document template
# ---------------------------------------------------------------------------


class HandbookDoc(BaseDocTemplate):
    def __init__(self, filename: str, meta: Dict[str, str], commit: str, source_name: str) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=MARGIN_L,
            rightMargin=MARGIN_R,
            topMargin=MARGIN_T,
            bottomMargin=MARGIN_B,
            title=f"LoopHole Handbook Part {meta.get('part', '')}: {meta.get('title', '')}",
            author="LoopHole team",
            subject=meta.get("subtitle", ""),
        )
        self.meta = meta
        self.commit = commit
        self.source_name = source_name
        self._heading_seq = 0
        frame = Frame(MARGIN_L, MARGIN_B, FRAME_W, PAGE_H - MARGIN_T - MARGIN_B, id="body")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._draw_cover),
            PageTemplate(id="body", frames=[frame], onPage=self._draw_body_page),
        ])

    def beforeDocument(self) -> None:
        # multiBuild runs several passes; keys must be identical each pass for the TOC to settle.
        self._heading_seq = 0

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Heading):
            self._heading_seq += 1
            key = f"h{self._heading_seq}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.plain, key, level=flowable.toc_level, closed=flowable.toc_level > 0)
            if flowable.toc_level <= 1:
                self.notify("TOCEntry", (flowable.toc_level, flowable.plain, self.page, key))

    def _draw_cover(self, canv, doc) -> None:
        m = self.meta
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, PAGE_H - 118 * mm, PAGE_W, 118 * mm, stroke=0, fill=1)
        canv.setFillColor(colors.HexColor("#9DB9D8"))
        canv.setFont("Body-Bold", 11)
        canv.drawString(MARGIN_L, PAGE_H - 34 * mm, "LOOPHOLE ENGINEERING HANDBOOK")
        canv.setFillColor(colors.white)
        canv.setFont("Body-Bold", 13)
        canv.drawString(MARGIN_L, PAGE_H - 48 * mm, f"Part {m.get('part', '?')}")
        title = m.get("title", "")
        size = 27 if len(title) < 34 else 22
        canv.setFont("Body-Bold", size)
        y = PAGE_H - 66 * mm
        for chunk in _wrap_words(title, 34 if size == 27 else 44):
            canv.drawString(MARGIN_L, y, chunk)
            y -= size * 1.2
        canv.setFillColor(colors.HexColor("#DCE7F3"))
        canv.setFont("Body", 11)
        y -= 4
        for chunk in _wrap_words(m.get("subtitle", ""), 78):
            canv.drawString(MARGIN_L, y, chunk)
            y -= 15
        canv.setFillColor(INK)
        y = PAGE_H - 140 * mm
        rows = [
            ("Audience", m.get("audience", "")),
            ("Covers", m.get("covers", "")),
            ("Repository state", f"branch main lineage, commit {self.commit}"),
            ("Generated", datetime.date.today().isoformat()),
            ("Source", f"docs/handbook/source/{self.source_name}"),
        ]
        for label, value in rows:
            if not value:
                continue
            canv.setFont("Body-Bold", 9.5)
            canv.setFillColor(MUTED)
            canv.drawString(MARGIN_L, y, label.upper())
            canv.setFont("Body", 10.5)
            canv.setFillColor(INK)
            yy = y - 14
            for chunk in _wrap_words(value, 92):
                canv.drawString(MARGIN_L, yy, chunk)
                yy -= 13.5
            y = yy - 9
        canv.setFont("Body", 8.5)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN_L, 16 * mm, "Parts: 1 Orientation · 2 Architecture · 3 Module Reference · 4 Toolchain, Tests & CI · 5 Tech-Debt Audit & Plan")
        canv.restoreState()

    def _draw_body_page(self, canv, doc) -> None:
        canv.saveState()
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(MARGIN_L, PAGE_H - 14 * mm, PAGE_W - MARGIN_R, PAGE_H - 14 * mm)
        canv.setFont("Body", 7.8)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN_L, PAGE_H - 12 * mm, f"LoopHole Engineering Handbook · Part {self.meta.get('part', '')}: {self.meta.get('title', '')}")
        canv.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 12 * mm, f"commit {self.commit}")
        canv.line(MARGIN_L, 13 * mm, PAGE_W - MARGIN_R, 13 * mm)
        canv.drawString(MARGIN_L, 9 * mm, "Regenerate: python docs/handbook/build_handbook.py")
        canv.drawRightString(PAGE_W - MARGIN_R, 9 * mm, f"Page {doc.page}")
        canv.restoreState()


def _wrap_words(text: str, width: int) -> List[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def build_one(src: Path, diagrams: Dict[str, dict], commit: str) -> Path:
    meta, body = parse_front_matter(src.read_text(encoding="utf-8"))
    out = OUT_DIR / (src.stem + ".pdf")
    doc = HandbookDoc(str(out), meta, commit, src.name)

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="Body-Bold", fontSize=10.5, leading=15, leftIndent=0, firstLineIndent=0, spaceBefore=5, textColor=NAVY),
        ParagraphStyle("toc1", fontName="Body", fontSize=9.2, leading=12.5, leftIndent=14, firstLineIndent=0, textColor=INK),
    ]
    story: List = [NextPageTemplate("body"), PageBreak(), Paragraph("Contents", STYLES["toc_title"]),
                   HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=8), toc, PageBreak()]
    story += build_story(body, diagrams)
    doc.multiBuild(story)
    return out


def main(argv: List[str]) -> int:
    global STYLES
    _register_fonts()
    STYLES = _styles()
    sys.path.insert(0, str(HERE))
    from diagrams import DIAGRAMS  # noqa: E402

    prefix = argv[1] if len(argv) > 1 else ""
    commit = _git_commit()
    sources = sorted(p for p in SOURCE_DIR.glob("*.md") if p.name.startswith(prefix))
    if not sources:
        print(f"No sources matching '{prefix}' in {SOURCE_DIR}", file=sys.stderr)
        return 1
    for src in sources:
        out = build_one(src, DIAGRAMS, commit)
        print(f"built {out.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
