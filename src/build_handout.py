"""Static fallback handout: the six figures, numbered, one per page.

If Colab fails for a student, this is what lets them still answer every question.

    python build_handout.py FIGDIR OUT.pdf
"""

import sys, os
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from PIL import Image

INK = colors.HexColor("#1a1a2e")
MUTED = colors.HexColor("#5b5b6b")
RULE = colors.HexColor("#c9c9d4")
ACCENT = colors.HexColor("#c8553d")

PAGE = landscape(A4)
W, H = PAGE
M = 16 * mm

FIGURES = [
    ("figure1.png", "Figure 1",
     "PCA of the bulk developmental cortex samples",
     "Q1", "64 samples. Colour = age in post-conceptional weeks; open circles = controls, "
           "triangles = cases. Axis labels carry the share of variance each component explains."),
    ("figure2.png", "Figure 2",
     "Volcano plot, cases vs controls",
     "Q2", "The 40 age-matched late-fetal samples only. x = log2 fold change, y = -log10 p. "
           "Dashed lines mark |log2FC| = 0.58 and the FDR 5% threshold. Coloured points are "
           "significant members of the two co-expression modules."),
    ("figure3.png", "Figure 3",
     "Single-cell QC metrics with MAD-based outlier flags",
     "Q3", "8,000 raw barcodes, before any filtering. Blue = kept, red = flagged by a "
           "median-absolute-deviation rule on log counts, log genes, and mitochondrial fraction. "
           "The dotted line on the right panel is a fixed 5% cut-off, for comparison."),
    ("figure4.png", "Figure 4",
     "UMAP and marker dot plot; clusters are unlabelled",
     "Q4", "Leiden clusters, lettered by size. In the dot plot, dot size = the fraction of cells "
           "in that cluster expressing the gene, colour = the mean expression scaled across "
           "clusters. Marker groups are bracketed above the columns."),
    ("figure5.png", "Figure 5",
     "Estimated cell-type proportions in the Block A bulk samples",
     "Q5", "Each bulk sample of the contrast cohort deconvolved against the Block B cluster "
           "signature. Left: the full composition, controls then cases. Right: the difference in "
           "mean proportion; * marks p < 0.05, Mann-Whitney."),
    ("figure5b.png", "Figure 5b",
     "Effect size before and after adjusting for composition",
     "Q5", "The same case/control contrast, re-fitted with the estimated neuronal fraction as a "
           "covariate. A gene on the dashed diagonal is one that composition does not explain; a "
           "gene pulled toward zero is one where it does."),
    ("figure6.png", "Figure 6",
     "Visium section with cell types mapped onto the spots",
     "Q6", "About 1,500 spots, 55 um across. Left: the six numbered spot groups, in no "
           "particular order. Right: the dominant estimated cell type per spot, from the same "
           "single-cell signature."),
]


def header(c, page_no):
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawString(M, H - M + 4 * mm,
                 "TRANSCRIPTOMIC ANALYSES IN NEURODEVELOPMENT  ·  M2 BRAIN DISORDERS, UE2"
                 "  ·  CONTINUOUS ASSESSMENT — FIGURE HANDOUT")
    c.drawRightString(W - M, H - M + 4 * mm, str(page_no))
    c.setStrokeColor(RULE)
    c.setLineWidth(0.5)
    c.line(M, H - M + 2 * mm, W - M, H - M + 2 * mm)


def cover(c):
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(M, H - 62 * mm, "The six figures")
    c.setFont("Helvetica", 13)
    c.setFillColor(MUTED)
    c.drawString(M, H - 72 * mm,
                 "Transcriptomic analyses in neurodevelopment  ·  continuous assessment")

    c.setStrokeColor(ACCENT); c.setLineWidth(2)
    c.line(M, H - 78 * mm, M + 45 * mm, H - 78 * mm)

    style = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15,
                           textColor=INK)
    p = Paragraph(
        "This handout exists so that a failed Colab runtime cannot cost you marks. The figures "
        "here are the reference output of the notebook: if your own run produced them, answer "
        "from your own; if it did not, answer from these. Either way you are marked on what you "
        "read out of the figure, not on whether your code ran."
        "<br/><br/>"
        "Every figure is numbered exactly as the worksheet refers to it, and each page names the "
        "question it belongs to. Figure 5b is the extra panel that Block C produces alongside "
        "Figure 5; the worksheet does not ask about it directly, but Q5 is much easier to answer "
        "with it in front of you.", style)
    p.wrapOn(c, W - 2 * M - 60 * mm, 100 * mm)
    p.drawOn(c, M, H - 128 * mm)

    # contents
    c.setFont("Helvetica-Bold", 8.5); c.setFillColor(MUTED)
    x = W - M - 52 * mm
    c.drawString(x, H - 84 * mm, "CONTENTS")
    y = H - 92 * mm
    for i, (_, num, title, q, _d) in enumerate(FIGURES):
        c.setFont("Helvetica-Bold", 9); c.setFillColor(INK)
        c.drawString(x, y, num)
        c.setFont("Helvetica", 9); c.setFillColor(MUTED)
        c.drawRightString(W - M, y, q)
        y -= 5.2 * mm
        c.setFont("Helvetica", 7.5)
        short = title if len(title) <= 44 else title.split(";")[0].split(",")[0]
        c.drawString(x, y, short[:44])
        y -= 7.5 * mm

    c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
    c.drawString(M, M, "Adrien Dufour  ·  Inserm  ·  M2 Brain Disorders, UE2 Genetics and Neurodevelopment")
    c.showPage()


def figure_page(c, path, num, title, q, desc, page_no):
    header(c, page_no)

    c.setFillColor(INK); c.setFont("Helvetica-Bold", 15)
    c.drawString(M, H - M - 6 * mm, num)
    c.setFont("Helvetica", 12); c.setFillColor(MUTED)
    c.drawString(M + 26 * mm, H - M - 6 * mm, title)

    # question chip
    tw = c.stringWidth(q, "Helvetica-Bold", 9) + 8 * mm
    c.setFillColor(ACCENT)
    c.roundRect(W - M - tw, H - M - 9 * mm, tw, 6.5 * mm, 1.5 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W - M - tw / 2, H - M - 7 * mm, q)

    # caption at the bottom
    style = ParagraphStyle("cap", fontName="Helvetica", fontSize=8.5, leading=12,
                           textColor=MUTED)
    p = Paragraph(desc, style)
    _, ph = p.wrapOn(c, W - 2 * M, 40 * mm)
    p.drawOn(c, M, M + 2 * mm)

    # the image, fitted to what is left
    top = H - M - 14 * mm
    bottom = M + ph + 8 * mm
    box_w, box_h = W - 2 * M, top - bottom
    iw, ih = Image.open(path).size
    scale = min(box_w / iw, box_h / ih)
    w, h = iw * scale, ih * scale
    c.drawImage(path, M + (box_w - w) / 2, bottom + (box_h - h) / 2, w, h,
                preserveAspectRatio=True, anchor="c")
    c.showPage()


def main(figdir, out):
    c = canvas.Canvas(out, pagesize=PAGE)
    c.setTitle("The six figures — M2BD continuous assessment")
    c.setAuthor("Adrien Dufour, Inserm")
    cover(c)
    for i, (fn, num, title, q, desc) in enumerate(FIGURES):
        path = os.path.join(figdir, fn)
        if not os.path.exists(path):
            print("missing:", path); continue
        figure_page(c, path, num, title, q, desc, i + 2)
    c.save()
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
