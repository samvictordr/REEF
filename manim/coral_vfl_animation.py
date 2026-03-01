"""
Coral Holobiont Multi-Omics — Vertical Federated Learning
ManimCE Animation  (manim v0.20.0)

Run (preview, low quality — fast):
    manim render -pql coral_vfl_animation.py CoralVFLPresentation

Write to video file (high quality):
    manim render -qh coral_vfl_animation.py CoralVFLPresentation

Write individual act scenes:
    manim render -pql coral_vfl_animation.py Act1Intro
    manim render -pql coral_vfl_animation.py Act2DataChallenge
    ... etc.
"""

from manim import *

# ─────────────────────────────────────────────
#  Palette
# ─────────────────────────────────────────────
C_TRANSCRIPTOMICS = "#4E9AF1"   # sky-blue
C_PROTEOMICS      = "#5CDB95"   # mint-green
C_METABOLOMICS    = "#F7A440"   # amber
C_MICROBIOME      = "#B57BEE"   # lavender
C_SERVER          = "#F25F5C"   # coral-red
C_GRADIENT        = "#FFE066"   # warm-yellow (gradient arrows)
C_BG_DARK         = "#111111"
C_TITLE           = "#EEEEEE"
C_TEXT            = "#EEEEEE"
C_SUBTITLE        = "#AAAAAA"
C_HIGHLIGHT       = "#FF6B6B"
C_PANEL           = "#1A1F2B"

OMIC_COLORS = {
    "Transcriptomics": C_TRANSCRIPTOMICS,
    "Proteomics":      C_PROTEOMICS,
    "Metabolomics":    C_METABOLOMICS,
    "Microbiome":      C_MICROBIOME,
}

OMIC_DIMS = {
    "Transcriptomics": "106 K features",
    "Proteomics":      "5.4 K features",
    "Metabolomics":    "3.2 K features",
    "Microbiome":      "7.0 K features",
}

CARD_PAD = 0.3
ARROW_TIP_LEN = 0.15   # smaller arrowheads globally

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def panel(w, h, color=C_PANEL, stroke=WHITE, stroke_w=1.5):
    return RoundedRectangle(
        width=w, height=h,
        corner_radius=0.25,
        fill_color=color, fill_opacity=1,
        stroke_color=stroke, stroke_width=stroke_w,
    )


def header_text(t, font_size=28, color=C_TITLE):
    return Tex(r"\text{" + t + "}", font_size=font_size, color=color)


def small_text(t, font_size=20, color=C_SUBTITLE):
    return Tex(r"\text{" + t + "}", font_size=font_size, color=color)


def omic_label(name, show_dim=True):
    color = OMIC_COLORS[name]
    label = Tex(r"\text{" + name + "}", font_size=22, color=color)
    rule = Line(LEFT * label.get_width()/2, RIGHT * label.get_width()/2, color=color)
    rule.set_stroke(width=2.5).next_to(label, DOWN, buff=0.08)
    group = VGroup(label, rule)
    if show_dim:
        dim = Tex(r"\text{" + OMIC_DIMS[name].strip() + "}", font_size=15, color=C_SUBTITLE)
        dim.next_to(rule, DOWN, buff=0.1)
        group.add(dim)
    return group


def lab_node(name, color):
    circ = Circle(radius=0.55, fill_color=color, fill_opacity=0.15,
                  stroke_color=color, stroke_width=2.5)
    lbl = Tex(r"\text{" + name + "}", font_size=17, color=color)
    lbl.next_to(circ, DOWN, buff=0.15)
    icon = Text("\U0001f52c", font_size=22, font="Latin Modern Roman").move_to(circ)
    return VGroup(circ, icon, lbl)


def mlp_icon(color=WHITE, height=1.0, width=0.6):
    """Stylized MLP icon -- stacked dots with connecting lines."""
    layers = [3, 4, 3]
    dot_r = 0.06
    h_gap = width / (len(layers) - 1)
    v_gap = height / (max(layers) - 1) if max(layers) > 1 else height
    all_dots = []
    coords = []
    for li, n in enumerate(layers):
        col_coords = []
        x = -width / 2 + li * h_gap
        total_h = (n - 1) * v_gap
        for di in range(n):
            y = -total_h / 2 + di * v_gap
            c = Dot(radius=dot_r, color=color).move_to([x, y, 0])
            all_dots.append(c)
            col_coords.append([x, y, 0])
        coords.append(col_coords)
    lines = []
    for li in range(len(layers) - 1):
        for a in coords[li]:
            for b in coords[li + 1]:
                ln = Line(a, b, color=color)
                ln.set_stroke(width=0.8, opacity=0.55)
                lines.append(ln)
    return VGroup(*lines, *all_dots)


def gradient_arrow(start, end):
    return Arrow(
        start, end,
        color=C_GRADIENT,
        stroke_width=4,
        buff=0.08,
        tip_length=ARROW_TIP_LEN,
    )


def forward_arrow(start, end):
    return Arrow(
        start, end,
        color=WHITE,
        stroke_width=3,
        buff=0.08,
        tip_length=ARROW_TIP_LEN,
    )


# ══════════════════════════════════════════════════════
#  ACT 1 — Title & Research Context
# ══════════════════════════════════════════════════════
class Act1Intro(Scene):
    """Introduces coral research and multi-omics challenge."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        # ── coral silhouette (abstract polyps) ──────────────
        polyps = VGroup(*[
            Circle(radius=r, fill_color=c, fill_opacity=0.7,
                   stroke_color=WHITE, stroke_width=0.8)
            .move_to([x, y, 0])
            for (r, x, y, c) in [
                (0.35, -5.5, -3.2, "#FF8C69"),
                (0.28, -4.8, -3.0, "#FF6B6B"),
                (0.40, -5.0, -2.6, "#FFA07A"),
                (0.25, -4.4, -3.3, "#FF8C69"),
                (0.32,  5.2, -3.0, "#FF6B6B"),
                (0.38,  4.7, -2.7, "#FFA07A"),
                (0.22,  5.5, -3.4, "#FF8C69"),
                (0.30,  4.3, -3.3, "#FF6B6B"),
            ]
        ])

        # ocean floor bar  (pushed lower so footer text sits above it)
        floor = Line([-7, -3.85, 0], [7, -3.85, 0])
        floor.set_stroke(color="#2196F3", width=3, opacity=0.5)

        # ── title ────────────────────────────────────────────
        title = VGroup(
            Tex(r"\text{Domain-Aware Priors Enable Vertical Federated Learning}", font_size=36, color=C_TITLE),
            Tex(r"\text{in Data-Scarce Coral Multi-Omics}", font_size=36, color=C_TITLE),
        ).arrange(DOWN, buff=0.15).move_to(ORIGIN + UP * 0.5)

        subtitle = Tex(
            r"\text{Vertical Federated Learning}",
            r" $\times$ ",
            r"\text{Domain-Aware Feature Selection}",
            r" $\times$ ",
            r"\text{XAI Inference}",
            font_size=20, color=C_SUBTITLE,
        ).next_to(title, DOWN, buff=0.35)

        # italic species name with LaTeX
        paper_ref = VGroup(
            Tex(r"\text{Dataset: Stephens et al.\ (2022)}", font_size=16, color=C_SUBTITLE),
            MathTex(r"\cdot", color=C_SUBTITLE).scale(0.4),
            Tex(r"\textit{Montipora capitata}", color=C_SUBTITLE).scale(0.55),
            MathTex(r"\cdot", color=C_SUBTITLE).scale(0.4),
            Tex(r"\text{DOI: 10.5281/zenodo.6861688}", font_size=16, color=C_SUBTITLE),
        ).arrange(RIGHT, buff=0.1).to_edge(DOWN, buff=0.3)

        # ── play ─────────────────────────────────────────────
        self.play(FadeIn(polyps, shift=UP * 0.3), FadeIn(floor), run_time=1.0)
        self.play(Write(title), run_time=1.8)
        self.play(Write(subtitle), run_time=1.2)
        # Bring footer text above the ocean floor line
        self.bring_to_front(paper_ref)
        self.play(Write(paper_ref), run_time=0.8)
        self.wait(2.0)

        # ── transition out ────────────────────────────────────
        self.play(FadeOut(VGroup(title, subtitle, paper_ref, polyps, floor)))


# ══════════════════════════════════════════════════════
#  ACT 2 — The Multi-Omics Data Challenge
# ══════════════════════════════════════════════════════
class Act2DataChallenge(Scene):
    """Shows the high-dimensional, small-N multi-omics problem."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{Multi-Omics: Big P, Small N}", font_size=36, color=WHITE)
        heading.to_edge(UP, buff=0.5)
        self.play(Write(heading), run_time=1.0)

        # ── data dimensionality overview ─────────────────────
        desc = Tex(
            r"\text{13 coral samples}", r" $\times$ ", r"$\sim$", r"\text{122\,000 molecular features}",
            font_size=22, color=C_SUBTITLE,
        ).next_to(heading, DOWN, buff=0.3)
        self.play(Write(desc))

        # ── omic labels arranged vertically with dims ─────────
        cards = VGroup(*[
            omic_label(name, show_dim=True)
            for name in OMIC_COLORS
        ]).arrange(DOWN, buff=0.3).shift(LEFT * 2.8)

        self.play(
            LaggedStart(*[Write(c) for c in cards], lag_ratio=0.25),
            run_time=1.8,
        )

        # ── annotation: overfitting danger ───────────────────
        warn_icon = Text("\u26a0", font_size=36, color=C_HIGHLIGHT, font="Latin Modern Roman")
        warn_lines = VGroup(
            Tex(r"\text{Severe overfitting risk:}", font_size=19, color=C_HIGHLIGHT),
            MathTex(r"p \gg n \qquad (122\mathrm{K} \gg 13)", color=C_TEXT).scale(0.62),
            Tex(r"\text{Domain selection needed}", font_size=17, color=C_SUBTITLE),
        ).arrange(DOWN, buff=0.12)
        warn_content = VGroup(warn_icon, warn_lines).arrange(DOWN, buff=0.15)
        warn_box = panel(
            warn_content.get_width() + 2 * CARD_PAD,
            warn_content.get_height() + 2 * CARD_PAD,
            color="#2A1A1A", stroke=C_HIGHLIGHT,
        )
        warn_content.move_to(warn_box)
        warn_grp = VGroup(warn_box, warn_content).shift(RIGHT * 3.0)

        self.play(DrawBorderThenFill(warn_box), run_time=0.6)
        self.play(FadeIn(warn_icon), Write(warn_lines), run_time=1.2)
        self.wait(2)

        # ── stress axis: bleaching labels ────────────────────
        stress_label = Tex(
            r"\text{Bleaching Task:  stressed  vs  ambient}",
            font_size=20, color=C_METABOLOMICS,
        ).to_edge(DOWN, buff=0.55)
        self.play(Write(stress_label))
        self.wait(1.5)

        self.play(FadeOut(VGroup(heading, desc, cards, warn_grp, stress_label)))


# ══════════════════════════════════════════════════════
#  ACT 3 — The Privacy Problem & VFL Motivation
# ══════════════════════════════════════════════════════
class Act3PrivacyProblem(Scene):
    """Shows why cross-lab privacy is important, motivating VFL."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{Multi-Lab Data: Privacy Barriers}", font_size=34, color=WHITE)
        heading.to_edge(UP, buff=0.5)
        self.play(Write(heading))

        lab_positions = [LEFT * 5, LEFT * 1.7, RIGHT * 1.7, RIGHT * 5]
        lab_names = list(OMIC_COLORS.keys())
        lab_colors = list(OMIC_COLORS.values())

        labs = VGroup()
        for pos, name, col in zip(lab_positions, lab_names, lab_colors):
            g = lab_node(name, col).move_to(pos)
            labs.add(g)

        self.play(LaggedStart(*[Write(l) for l in labs], lag_ratio=0.2),
                  run_time=1.5)

        # red-X "cannot share" lines
        crosses = VGroup()
        for i in range(len(labs) - 1):
            mid = (labs[i][0].get_center() + labs[i + 1][0].get_center()) / 2.0
            c1 = Line(mid + UL * 0.22, mid + DR * 0.22, color=RED).set_stroke(width=3)
            c2 = Line(mid + UR * 0.22, mid + DL * 0.22, color=RED).set_stroke(width=3)
            crosses.add(c1, c2)

        lock_text = VGroup(
            Text("\U0001f512", font_size=22, font="Latin Modern Roman"),
            Tex(r"\text{Raw data cannot leave the lab}", font_size=22, color=RED),
        ).arrange(RIGHT, buff=0.15)
        lock_text.next_to(labs, DOWN, buff=0.45)

        self.play(Create(crosses), run_time=0.8)
        self.play(Write(lock_text), run_time=0.8)
        self.wait(1.2)

        # ── solution: share embeddings ────────────────────────
        solution = VGroup(
            Tex(r"\text{Solution:  Vertical Federated Learning}", font_size=22, color=C_TRANSCRIPTOMICS),
            Tex(r"\text{Share embeddings, not raw features}", font_size=22, color=C_TRANSCRIPTOMICS),
        ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.6)
        self.play(Write(solution), run_time=1.2)
        self.wait(2)

        self.play(FadeOut(VGroup(heading, labs, crosses, lock_text, solution)))


# ══════════════════════════════════════════════════════
#  ACT 4 — VFL Architecture (forward + backward pass)
# ══════════════════════════════════════════════════════
class Act4VFLArchitecture(MovingCameraScene):
    """Animates a full VFL forward pass and gradient return."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{Vertical Federated Learning --- Architecture}", font_size=32, color=WHITE)
        heading.to_edge(UP, buff=0.4)
        self.play(Write(heading))

        omic_list = list(OMIC_COLORS.keys())
        colors     = list(OMIC_COLORS.values())
        n          = len(omic_list)
        silo_x     = -5.2
        enc_x      = -3.0
        emb_x      = -0.9
        server_x   =  1.8
        ys = [1.9, 0.65, -0.65, -1.9]

        # ── Silo raw-data panels ──────────────────────────────
        silos = VGroup()
        for y, name, col in zip(ys, omic_list, colors):
            lbl = Tex(r"\text{" + name + "}", font_size=17, color=col)
            box = panel(lbl.get_width() + 2 * CARD_PAD, lbl.get_height() + 2 * CARD_PAD, color=C_PANEL, stroke=col)
            lbl.move_to(box)
            silo = VGroup(box, lbl).move_to([silo_x, y, 0])
            silos.add(silo)

        # ── Encoder MLP icons ────────────────────────────────
        encoders = VGroup()
        for y, col in zip(ys, colors):
            bg = panel(0.9, 0.9, color="#111822", stroke=col, stroke_w=1.2)
            mlp = mlp_icon(color=col, height=0.45, width=0.45)
            mlp.move_to(bg).shift(UP * 0.08)
            enc_lbl = Tex(r"\text{Enc}", font_size=11, color=col)
            enc_lbl.next_to(mlp, DOWN, buff=0.06)
            enc = VGroup(bg, mlp, enc_lbl).move_to([enc_x, y, 0])
            encoders.add(enc)

        # ── Embedding circles (with soft glow) ────────────────
        embeddings = VGroup()
        for y, col in zip(ys, colors):
            glow = Circle(radius=0.44, fill_color=col, fill_opacity=0.13, stroke_width=0)
            circ = Circle(radius=0.28, fill_color=col, fill_opacity=0.85,
                          stroke_color="#EEEEEE", stroke_width=1.5)
            emb_txt = MathTex(r"\mathbf{e}", color="#EEEEEE").scale(0.52)
            # Compose group, then position text on top with high z_index
            emb = VGroup(glow, circ, emb_txt).move_to([emb_x, y, 0])
            emb_txt.move_to(circ).set_z_index(5)
            embeddings.add(emb)

        # ── Server fusion panel ───────────────────────────────
        server_panel = panel(2.2, 3.0, color="#1E0A0A", stroke=C_SERVER)
        server_panel.move_to([server_x, 0, 0])
        server_lbl  = Tex(r"\text{Server}", font_size=20, color=C_SERVER)
        server_lbl.move_to(server_panel).shift(UP * 1.1)
        fusion_lbl  = Tex(r"\text{Fusion Head}", font_size=15, color=C_TEXT)
        fusion_lbl.move_to(server_panel).shift(UP * 0.2)
        loss_lbl    = VGroup(
            Tex(r"\text{Loss }", font_size=14, color=C_GRADIENT),
            MathTex(r"\mathcal{L}", color=C_GRADIENT).scale(0.85),
        ).arrange(RIGHT, buff=0.05).move_to(server_panel).shift(DOWN * 0.6)
        # Clean inline label replacing the old labels_note box
        labels_tag = Tex(r"\text{(labels: server-only)}", font_size=13, color=C_GRADIENT)
        labels_tag.next_to(loss_lbl, DOWN, buff=0.15)
        server_grp  = VGroup(server_panel, server_lbl, fusion_lbl, loss_lbl, labels_tag)

        # ── Output prediction ─────────────────────────────────
        pred_content = VGroup(
            Tex(r"\text{Predict}", font_size=14, color=C_PROTEOMICS),
            Tex(r"\text{Bleaching}", font_size=14, color=C_PROTEOMICS),
        ).arrange(DOWN, buff=0.06)
        pred_box = panel(
            pred_content.get_width() + 2 * CARD_PAD,
            pred_content.get_height() + 2 * CARD_PAD,
            color="#0B1810", stroke=C_PROTEOMICS,
        )
        pred_content.move_to(pred_box)
        pred_grp = VGroup(pred_box, pred_content).move_to([server_x + 2.2, 0, 0])

        # ── Column headers ────────────────────────────────────
        col_headers = VGroup(
            Tex(r"\text{Silos / Labs}", font_size=15, color=C_SUBTITLE).move_to([silo_x, 2.7, 0]),
            Tex(r"\text{Encoders}",     font_size=15, color=C_SUBTITLE).move_to([enc_x,  2.7, 0]),
            Tex(r"\text{Embeddings}",   font_size=15, color=C_SUBTITLE).move_to([emb_x,  2.7, 0]),
        )

        # ── Arrows: silo → encoder ────────────────────────────
        fwd_arrows_1 = VGroup(*[
            Arrow([silo_x + 1.0, y, 0], [enc_x - 0.45, y, 0],
                  color=WHITE, stroke_width=3, buff=0.05, tip_length=ARROW_TIP_LEN)
            for y in ys
        ])
        # ── Arrows: encoder → embedding ──────────────────────
        fwd_arrows_2 = VGroup(*[
            Arrow([enc_x + 0.45, y, 0], [emb_x - 0.28, y, 0],
                  color=WHITE, stroke_width=3, buff=0.05, tip_length=ARROW_TIP_LEN)
            for y in ys
        ])
        # ── Arrows: embedding → server ────────────────────────
        fwd_arrows_3 = VGroup(*[
            Arrow([emb_x + 0.28, y, 0], [server_x - 1.1, y * 0.7, 0],
                  color=WHITE, stroke_width=3, buff=0.05, tip_length=ARROW_TIP_LEN)
            for y in ys
        ])
        # ── Arrow: server → prediction ────────────────────────
        fwd_arrow_out = Arrow([server_x + 1.1, 0, 0], [server_x + 1.55, 0, 0],
                               color=WHITE, stroke_width=3, buff=0.05, tip_length=ARROW_TIP_LEN)

        # ── Gradient arrows (backward): server → encoders ─────
        grad_arrows = VGroup(*[
            Arrow([server_x - 1.1, y * 0.7, 0], [emb_x + 0.28, y, 0],
                  color=C_GRADIENT, stroke_width=4, buff=0.05, tip_length=ARROW_TIP_LEN)
            for y in ys
        ])

        # ── ASSEMBLE ─────────────────────────────────────────
        self.play(Write(col_headers), run_time=0.7)
        self.play(
            LaggedStart(*[Write(s) for s in silos], lag_ratio=0.15),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*[DrawBorderThenFill(e) for e in encoders], lag_ratio=0.15),
            run_time=0.9,
        )
        self.play(Create(fwd_arrows_1), run_time=0.7)
        self.play(
            LaggedStart(*[DrawBorderThenFill(e) for e in embeddings], lag_ratio=0.15),
            run_time=0.8,
        )
        self.play(Create(fwd_arrows_2), run_time=0.7)

        # server appears
        self.play(DrawBorderThenFill(server_grp))
        self.play(Create(fwd_arrows_3), run_time=0.8)
        self.play(DrawBorderThenFill(pred_grp), Create(fwd_arrow_out))
        self.wait(0.8)

        # ── Camera zoom into fusion head for emphasis ─────────
        frame = self.camera.frame
        frame.save_state()
        self.play(
            frame.animate.move_to([server_x, 0.1, 0]).scale(0.58),
            run_time=1.2, rate_func=smooth,
        )
        self.wait(0.7)
        self.play(Restore(frame), run_time=0.9, rate_func=smooth)

        # ── Backward pass (gradient) ─────────────────────────
        grad_label = Tex(r"$\leftarrow$ \text{Gradients back to silos}", font_size=19, color=C_GRADIENT)
        grad_label.to_edge(DOWN, buff=0.4)
        self.play(Write(grad_label), run_time=0.7)
        self.play(Create(grad_arrows), run_time=1.0)

        # ── Pulsing sparks travelling along gradient paths ────
        pulses = [Dot(radius=0.09, color=C_GRADIENT).move_to(arr.get_start())
                  for arr in grad_arrows]
        self.add(*pulses)
        self.play(
            LaggedStart(
                *[MoveAlongPath(p, arr, rate_func=smooth)
                  for p, arr in zip(pulses, grad_arrows)],
                lag_ratio=0.2, run_time=1.6,
            ),
        )
        self.remove(*pulses)
        self.wait(0.3)

        # privacy note
        privacy_note = Tex(
            r"\text{Raw features never leave the silo} $\checkmark$",
            font_size=19, color=C_PROTEOMICS,
        ).next_to(grad_label, UP, buff=0.25)
        self.play(Write(privacy_note))
        self.wait(2.5)

        self.play(FadeOut(VGroup(
            heading, col_headers, silos, encoders, fwd_arrows_1,
            fwd_arrows_2, embeddings, server_grp, fwd_arrows_3,
            fwd_arrow_out, pred_grp,
            grad_arrows, grad_label, privacy_note,
        )))


# ══════════════════════════════════════════════════════
#  ACT 5 — Domain-Aware Feature Selection
# ══════════════════════════════════════════════════════
class Act5DomainAware(Scene):
    """Explains two-phase domain-aware feature selection with bio-weights."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{REEF: Two-Phase Training}", font_size=32, color=WHITE)
        heading.to_edge(UP, buff=0.45)
        self.play(Write(heading))

        # ── Phase timeline ────────────────────────────────────
        phase_colors = [C_TRANSCRIPTOMICS, C_PROTEOMICS, C_METABOLOMICS]
        phase_bgs    = ["#101A25", "#0F1F10", "#1A130A"]
        labels_p = [
            ("Phase 1", "Warmup", r"\text{20 rounds}", r"\text{Full feature set}", C_TRANSCRIPTOMICS),
            ("Phase 2", "Domain Selection", r"\text{Gradient saliency}", r"$\to$ \text{top-K features}", C_PROTEOMICS),
            ("Phase 3", "Refinement", r"\text{Sparse feature set}", r"$2\times$ \text{more rounds}", C_METABOLOMICS),
        ]

        phases = VGroup()
        phase_texts = VGroup()
        for bg_col, stroke_col, (p_num, p_name, b1, b2, col) in zip(phase_bgs, phase_colors, labels_p):
            t_line1 = Tex(r"\text{" + p_num + "}", font_size=17, color=col)
            t_line2 = Tex(r"\text{" + p_name + "}", font_size=17, color=col)
            t_top = VGroup(t_line1, t_line2).arrange(DOWN, buff=0.04)
            b_line1 = Tex(b1, font_size=14, color=WHITE)
            b_line2 = Tex(b2, font_size=14, color=WHITE)
            b_bot = VGroup(b_line1, b_line2).arrange(DOWN, buff=0.04)
            content = VGroup(t_top, b_bot).arrange(DOWN, buff=0.12)
            box = panel(content.get_width() + 2 * CARD_PAD, content.get_height() + 2 * CARD_PAD,
                        color=bg_col, stroke=stroke_col)
            content.move_to(box)
            phases.add(box)
            phase_texts.add(content)

        phase_groups = VGroup(*[VGroup(phases[i], phase_texts[i]) for i in range(3)])
        phase_groups.arrange(RIGHT, buff=0.3).shift(UP * 0.8)

        arrows_p = VGroup(*[
            Arrow(phase_groups[i].get_right(), phase_groups[i + 1].get_left(), buff=0.05,
                  color=WHITE, stroke_width=3, tip_length=ARROW_TIP_LEN)
            for i in range(len(phase_groups) - 1)
        ])

        self.play(
            LaggedStart(*[DrawBorderThenFill(p) for p in phases], lag_ratio=0.2),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*[Write(t) for t in phase_texts], lag_ratio=0.2),
            run_time=1.2,
        )
        self.play(Create(arrows_p), run_time=0.7)
        self.wait(0.8)

        # ── Feature reduction bar ────────────────────────────
        bar_note = VGroup(
            Tex(r"\text{Feature reduction: }", font_size=18, color=C_TEXT),
            MathTex(r"106\,\mathrm{K} \;\longrightarrow\; {\sim}500 \;\mathrm{per\;omic}",
                color=C_TEXT).scale(0.62),
        ).arrange(RIGHT, buff=0.08).shift(UP * -0.5)
        self.play(Write(bar_note), run_time=1.0)

        # simple bar comparison
        bar_full = Rectangle(width=5.0, height=0.35,
                              fill_color=C_TRANSCRIPTOMICS, fill_opacity=0.8,
                              stroke_width=0).shift(DOWN * 1.05 + LEFT * 1.0)
        bar_sel  = Rectangle(width=0.25, height=0.35,
                              fill_color=C_METABOLOMICS, fill_opacity=0.9,
                              stroke_width=0).next_to(bar_full, DOWN, buff=0.18, aligned_edge=LEFT)
        lbl_full = Tex(r"\text{All features (106K)}", font_size=14, color=C_SUBTITLE)
        lbl_full.next_to(bar_full, RIGHT, buff=0.12)
        lbl_sel  = Tex(r"\text{Selected (}$\approx$\text{500)}", font_size=14, color=C_SUBTITLE)
        lbl_sel.next_to(bar_sel, RIGHT, buff=0.12)

        # ── Live value counter: 106 000 → 500 ────────────────
        tracker = ValueTracker(106000)
        counter_pos = bar_full.get_left() + LEFT * 1.2
        def _make_counter(val):
            return Text(f"{int(val):,}", font_size=28, color=C_TRANSCRIPTOMICS,
                        font="Latin Modern Roman")
        counter = _make_counter(106000)
        counter.move_to(counter_pos)
        counter.add_updater(
            lambda m: m.become(
                _make_counter(tracker.get_value()).move_to(counter_pos)
            )
        )
        self.add(counter)

        self.play(GrowFromEdge(bar_full, LEFT), Write(lbl_full))
        self.play(
            GrowFromEdge(bar_sel, LEFT),
            Write(lbl_sel),
            tracker.animate.set_value(500),
            run_time=2.5, rate_func=smooth,
        )
        counter.clear_updaters()
        self.wait(0.8)

        # ── Biological importance weights ────────────────────
        bio_heading = Tex(r"\text{Omic Biological Weights}", font_size=20, color=C_GRADIENT)
        bio_heading.shift(DOWN * 2.1)
        bio_entries = [
            ("Transcriptomics", r"1.5\times", "(primary regulatory driver)"),
            ("Proteomics",      r"1.0\times", "(effector baseline)"),
            ("Metabolomics",    r"0.8\times", "(functional readout)"),
            ("Microbiome",      r"0.5\times", "(environmental symbiont)"),
        ]
        bio_rows = VGroup(*[
            VGroup(
                Dot(radius=0.06, color=OMIC_COLORS[name]),
                Tex(r"\text{" + name + "}", font_size=16, color=OMIC_COLORS[name]),
                Tex(r"$" + weight + r"$ \text{" + desc + "}", font_size=16, color=WHITE),
            ).arrange(RIGHT, buff=0.15)
            for name, weight, desc in bio_entries
        ]).arrange(DOWN, buff=0.15, aligned_edge=LEFT).next_to(bio_heading, DOWN, buff=0.2)
        bio_rows.shift(LEFT * 0.5)

        self.play(Write(bio_heading))
        self.play(
            LaggedStart(*[Write(r) for r in bio_rows], lag_ratio=0.15),
            run_time=1.5,
        )
        self.wait(2.5)

        self.play(FadeOut(VGroup(
            heading, phase_groups, arrows_p,
            bar_note, bar_full, bar_sel, lbl_full, lbl_sel, counter,
            bio_heading, bio_rows,
        )))


# ══════════════════════════════════════════════════════
#  ACT 6 — The Missing Omic Problem
# ══════════════════════════════════════════════════════
class Act6MissingOmic(Scene):
    """Visualises the incomplete multi-lab data matrix."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{Real-World Problem: Missing Omic Blocks}", font_size=32, color=WHITE)
        heading.to_edge(UP, buff=0.45)
        self.play(Write(heading))

        # ── data matrix grid ─────────────────────────────────
        rows   = 5
        cols   = 4
        cell_w = 1.6
        cell_h = 0.6
        omic_list  = list(OMIC_COLORS.keys())
        omic_colors = list(OMIC_COLORS.values())

        # availability pattern  (True = present, False = missing)
        availability = [
            [True,  True,  True,  True ],
            [True,  True,  False, True ],
            [True,  False, True,  False],
            [False, True,  True,  True ],
            [True,  True,  True,  False],
        ]

        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                present = availability[r][c]
                color   = omic_colors[c] if present else "#2A2A2A"
                stroke  = omic_colors[c] if present else "#444444"
                rect = Rectangle(
                    width=cell_w, height=cell_h,
                    fill_color=color, fill_opacity=0.75 if present else 0.55,
                    stroke_color=stroke, stroke_width=1.2,
                )
                rect.move_to([c * (cell_w + 0.06) - 2.85, -r * (cell_h + 0.06) + 0.9, 0])
                inner = MathTex(
                    r"\checkmark" if present else r"\times",
                    font_size=18,
                    color=WHITE if present else "#666666",
                )
                inner.move_to(rect)
                grid.add(VGroup(rect, inner))

        # column headers
        col_heads = VGroup(*[
            Tex(r"\text{" + name + "}", font_size=14, color=omic_colors[i])
            .move_to([i * (cell_w + 0.06) - 2.85, 1.35, 0])
            for i, name in enumerate(omic_list)
        ])
        # row labels
        row_labels = VGroup(*[
            Tex(r"\text{Sample " + str(r + 1) + "}", font_size=13, color=C_SUBTITLE)
            .move_to([-5.0, -r * (cell_h + 0.06) + 0.9, 0])
            for r in range(rows)
        ])

        self.play(Write(col_heads), Write(row_labels), run_time=0.7)
        self.play(
            LaggedStart(*[DrawBorderThenFill(g) for g in grid], lag_ratio=0.05),
            run_time=1.5,
        )
        self.wait(0.8)

        # ── naive approaches ─────────────────────────────────
        naive_title = Tex(r"\text{Naive Strategies}", font_size=18, color=RED)
        naive_items = VGroup(*[
            Tex(s, font_size=15, color=WHITE)
            for s in [
                r"\text{1.  Complete-case: discard samples}",
                r"\text{2.  Zero imputation: fill with 0}",
                r"\text{3.  Mean imputation: fill with }$\mu$",
                r"$\quad\to$ \text{all lose information}",
            ]
        ]).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        naive_content = VGroup(naive_title, naive_items).arrange(DOWN, buff=0.2)
        naive_box = panel(
            naive_content.get_width() + 2 * CARD_PAD,
            naive_content.get_height() + 2 * CARD_PAD,
            color="#1A1010", stroke=RED,
        )
        naive_content.move_to(naive_box)
        naive_grp = VGroup(naive_box, naive_content).shift(RIGHT * 3.8)

        self.play(DrawBorderThenFill(naive_box), Write(naive_title), run_time=0.7)
        self.play(
            LaggedStart(*[Write(it) for it in naive_items], lag_ratio=0.2),
            run_time=1.5,
        )
        self.wait(2)

        self.play(FadeOut(VGroup(
            heading, grid, col_heads, row_labels, naive_grp,
        )))


# ══════════════════════════════════════════════════════
#  ACT 7 — XAI/Gradient Saliency-Based Omic Inference (Main Contribution)
# ══════════════════════════════════════════════════════
class Act7XAIInference(Scene):
    """Animates the gradient saliency-based missing omic inference pipeline."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{Core Contribution: XAI-Inferred Missing Omics}", font_size=30, color=WHITE)
        heading.to_edge(UP, buff=0.45)
        self.play(Write(heading))

        subtitle = Tex(
            r"\text{Use gradient saliency from available omics to infer a latent proxy for the missing one}",
            font_size=18, color=C_SUBTITLE,
        ).next_to(heading, DOWN, buff=0.25)
        self.play(Write(subtitle))
        self.wait(0.4)

        # ── pipeline blocks ───────────────────────────────────
        step_data = [
            (["Available", "Omics"],               C_TRANSCRIPTOMICS),
            (["Surrogate", "Model", "(LightGBM)"], C_PROTEOMICS),
            (["Gradient", "Saliency", "Values"],       C_METABOLOMICS),
            (["Aggregate", "$\\to$ Latent", "Vector (64D)"], C_MICROBIOME),
            (["VFL", "Fusion", "Head"],             C_SERVER),
        ]

        step_boxes = VGroup()
        step_labels = VGroup()
        for lines, col in step_data:
            lbl_lines_list = []
            for l in lines:
                if "$" in l:
                    lbl_lines_list.append(Tex(l, font_size=14, color=col))
                else:
                    lbl_lines_list.append(Tex(r"\text{" + l + "}", font_size=14, color=col))
            lbl_lines = VGroup(*lbl_lines_list).arrange(DOWN, buff=0.04)
            box = panel(
                max(lbl_lines.get_width() + 2 * CARD_PAD, 1.65),
                max(lbl_lines.get_height() + 2 * CARD_PAD, 1.4),
                color=C_PANEL, stroke=col,
            )
            lbl_lines.move_to(box)
            step_boxes.add(box)
            step_labels.add(lbl_lines)

        step_groups = VGroup(*[VGroup(step_boxes[i], step_labels[i]) for i in range(len(step_data))])
        step_groups.arrange(RIGHT, buff=0.26).shift(UP * 0.3)

        step_arrows = VGroup(*[
            Arrow(step_groups[i].get_right(), step_groups[i + 1].get_left(),
                  buff=0.05, color=WHITE, stroke_width=3, tip_length=ARROW_TIP_LEN)
            for i in range(len(step_groups) - 1)
        ])

        self.play(
            LaggedStart(*[DrawBorderThenFill(b) for b in step_boxes], lag_ratio=0.2),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[Write(l) for l in step_labels], lag_ratio=0.15),
            run_time=1.0,
        )
        self.play(Create(step_arrows), run_time=0.7)
        self.wait(0.6)

        # ── Gradient saliency bar chart (schematic) ────────────────────────
        shap_title = Tex(r"\text{Gradient Saliency Importance --- Available Omics}", font_size=17, color=C_GRADIENT)
        shap_title.shift(DOWN * 1.55)
        bar_data = [
            ("Heat-shock proteins", 0.82, C_TRANSCRIPTOMICS),
            ("Antioxidant genes",   0.65, C_TRANSCRIPTOMICS),
            ("Lipid metabolites",   0.48, C_METABOLOMICS),
            ("Symbiont ratio",      0.31, C_MICROBIOME),
            ("Other",               0.18, "#888888"),
        ]
        bar_w = 0.7
        bar_base_y = -2.35   # baseline for bars (pushed further down)
        bars   = VGroup()
        btexts = VGroup()
        for idx, (label, val, col) in enumerate(bar_data):
            h = val * 0.85
            b = Rectangle(width=bar_w, height=h,
                          fill_color=col, fill_opacity=0.85, stroke_width=0)
            b.move_to([idx * (bar_w + 0.25) - 3.3, bar_base_y + h / 2, 0])
            t = Tex(r"\text{" + label + "}", font_size=10, color=C_SUBTITLE)
            t.next_to(b, DOWN, buff=0.1)
            bars.add(b)
            btexts.add(t)

        y_label = Tex(r"\text{Importance}", font_size=13, color=C_SUBTITLE).rotate(PI / 2)
        y_label.move_to([-4.0, -2.2, 0])

        self.play(Write(shap_title))
        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.15),
            run_time=1.2,
        )
        self.play(
            LaggedStart(*[Write(t) for t in btexts], lag_ratio=0.1),
            Write(y_label),
            run_time=0.9,
        )

        # ── highlight: proxy replaces missing block ───────────
        proxy_note = VGroup(
            Tex(r"\text{Proxy latent vector (64D)}", font_size=16, color=C_MICROBIOME),
            Tex(r"\text{replaces missing omic in VFL}", font_size=16, color=C_MICROBIOME),
        ).arrange(DOWN, buff=0.08).shift(RIGHT * 4.0 + DOWN * 1.5)
        proxy_arrow = Arrow(
            step_groups[-2].get_bottom() + DOWN * 0.05,
            proxy_note.get_top(),
            buff=0.05, color=C_MICROBIOME, stroke_width=2, tip_length=ARROW_TIP_LEN,
        )

        self.play(Write(proxy_note), Create(proxy_arrow))
        self.wait(2.5)

        self.play(FadeOut(VGroup(
            heading, subtitle, step_groups, step_arrows,
            shap_title, bars, btexts, y_label, proxy_note, proxy_arrow,
        )))


# ══════════════════════════════════════════════════════
#  ACT 8 — Experiment Setup & Strategies Compared
# ══════════════════════════════════════════════════════
class Act8ExperimentSetup(Scene):
    """Shows the H2 experiment: 5 strategies, 5-fold CV, metrics."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{Experiment H2 --- Strategies Compared}", font_size=32, color=WHITE)
        heading.to_edge(UP, buff=0.45)
        self.play(Write(heading))

        # ── strategy cards ────────────────────────────────────
        strategies = [
            ("Complete-case",   ["Baseline:", "all-omics only"],       "#555555"),
            ("Zero imputation", ["Fill missing", "with 0"],            "#884444"),
            ("Mean imputation", ["Fill missing", r"with $\mu$"],       "#886644"),
            ("Mosaic mask",     ["Mask at fusion", "(model-aware)"],    "#445588"),
            ("XAI-inferred",    ["Saliency proxy", "(our method)"],         C_PROTEOMICS),
        ]

        cards = VGroup()
        for idx, (name, body_lines, col) in enumerate(strategies):
            if idx == 4:
                name_tex = Tex(r"\text{XAI-inferred }$\bigstar$", font_size=15, color=col)
            else:
                name_tex = Tex(r"\text{" + name + "}", font_size=15, color=col)
            body_parts = []
            for bl in body_lines:
                if "$" in bl:
                    body_parts.append(Tex(bl, font_size=13, color=WHITE))
                else:
                    body_parts.append(Tex(r"\text{" + bl + "}", font_size=13, color=WHITE))
            body_vg = VGroup(*body_parts).arrange(DOWN, buff=0.04)
            content = VGroup(name_tex, body_vg).arrange(DOWN, buff=0.12)
            box = panel(
                max(content.get_width() + 2 * CARD_PAD, 2.0),
                max(content.get_height() + 2 * CARD_PAD, 1.3),
                color=C_PANEL, stroke=col,
            )
            content.move_to(box)
            card = VGroup(box, content)
            cards.add(card)
        cards.arrange(RIGHT, buff=0.22).shift(UP * 0.45)

        self.play(
            LaggedStart(*[Write(c) for c in cards], lag_ratio=0.15),
            run_time=1.5,
        )

        # highlight XAI card
        highlight_rect = SurroundingRectangle(cards[-1], color=C_PROTEOMICS,
                                              stroke_width=2.5, buff=0.04)
        self.play(Create(highlight_rect))
        self.wait(0.5)

        # ── cross-validation setup ────────────────────────────
        cv_title = Tex(r"\text{5-Fold Cross-Validation (seed=42)}", font_size=20, color=C_SUBTITLE)
        cv_title.shift(DOWN * 0.75)
        self.play(Write(cv_title))

        folds = VGroup(*[
            Rectangle(width=0.8, height=0.45,
                      fill_color="#4E9AF1" if i != 2 else C_HIGHLIGHT,
                      fill_opacity=0.8, stroke_width=0)
            for i in range(5)
        ]).arrange(RIGHT, buff=0.1).next_to(cv_title, DOWN, buff=0.25)
        fold_labels = VGroup(*[
            Tex(r"\text{Test}" if i == 2 else r"\text{Train}", font_size=12, color=WHITE).move_to(folds[i])
            for i in range(5)
        ])
        self.play(DrawBorderThenFill(folds), Write(fold_labels))
        self.wait(0.4)

        # ── metrics ───────────────────────────────────────────
        metrics_title = Tex(r"\text{Primary Metrics}", font_size=20, color=C_GRADIENT)
        metrics_title.shift(DOWN * 2.0)
        metric_items = VGroup(
            Tex(r"\text{AUROC (classification)}", font_size=17, color=WHITE),
            Tex(r"\text{F1-score (balanced accuracy)}", font_size=17, color=WHITE),
            Tex(r"\text{MSE (regression)}", font_size=17, color=WHITE),
        ).arrange(RIGHT, buff=0.6).next_to(metrics_title, DOWN, buff=0.18)

        self.play(Write(metrics_title))
        self.play(
            LaggedStart(*[Write(m) for m in metric_items], lag_ratio=0.2),
            run_time=1.2,
        )
        self.wait(2)

        self.play(FadeOut(VGroup(
            heading, cards, highlight_rect, cv_title, folds, fold_labels,
            metrics_title, metric_items,
        )))


# ══════════════════════════════════════════════════════
#  ACT 9 — NVFlare Integration
# ══════════════════════════════════════════════════════
class Act9NVFlare(Scene):
    """Shows how NVFlare orchestrates the federated training."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{NVFlare Orchestration}", font_size=34, color=WHITE)
        heading.to_edge(UP, buff=0.45)
        self.play(Write(heading))

        # ── server box ────────────────────────────────────────
        srv_content = VGroup(
            Tex(r"\text{NVFlare Server}", font_size=17, color=C_SERVER),
            Tex(r"\text{(Aggregator)}", font_size=17, color=C_SERVER),
        ).arrange(DOWN, buff=0.04)
        srv = panel(
            srv_content.get_width() + 2 * CARD_PAD,
            srv_content.get_height() + 2 * CARD_PAD,
            color="#1E0A0A", stroke=C_SERVER,
        )
        srv_content.move_to(srv)
        srv_grp = VGroup(srv, srv_content).shift(UP * 0.8)

        # ── client boxes ─────────────────────────────────────
        client_positions = [LEFT * 4, LEFT * 1.4, RIGHT * 1.4, RIGHT * 4]
        clients = VGroup()
        for pos, (name, col) in zip(client_positions, OMIC_COLORS.items()):
            lbl1 = Tex(r"\text{Client}", font_size=14, color=C_SUBTITLE)
            lbl2 = Tex(r"\text{" + name + "}", font_size=14, color=col)
            content = VGroup(lbl1, lbl2).arrange(DOWN, buff=0.1)
            box = panel(
                max(content.get_width() + 2 * CARD_PAD, 1.7),
                max(content.get_height() + 2 * CARD_PAD, 1.0),
                color=C_PANEL, stroke=col,
            )
            content.move_to(box)
            clients.add(VGroup(box, content).move_to(pos + DOWN * 1.4))

        # arrows
        arrows_down = VGroup(*[
            Arrow(srv_grp.get_bottom() + [0, 0.05, 0],
                  c.get_top() - [0, 0.05, 0],
                  buff=0.05, color=WHITE, stroke_width=3, tip_length=ARROW_TIP_LEN)
            for c in clients
        ])
        arrows_up = VGroup(*[
            Arrow(c.get_top() - [0, 0.05, 0],
                  srv_grp.get_bottom() + [0, 0.05, 0],
                  buff=0.05, color=C_GRADIENT, stroke_width=3, tip_length=ARROW_TIP_LEN)
            for c in clients
        ])

        # labels
        down_lbl = Tex(r"\text{Model weights / config}", font_size=14, color=WHITE)
        down_lbl.shift(LEFT * 5.5 + DOWN * 0.2)
        up_lbl = Tex(r"\text{Embeddings / gradients}", font_size=14, color=C_GRADIENT)
        up_lbl.shift(LEFT * 5.5 + DOWN * 0.6)

        self.play(DrawBorderThenFill(srv), Write(srv_content))
        self.play(
            LaggedStart(*[Write(c) for c in clients], lag_ratio=0.15),
            run_time=1.0,
        )
        self.play(Create(arrows_down), Write(down_lbl), run_time=0.9)
        self.play(Create(arrows_up), Write(up_lbl), run_time=0.9)
        self.wait(2.7)

        self.play(FadeOut(VGroup(
            heading, srv_grp, clients, arrows_down, arrows_up,
            down_lbl, up_lbl,
        )))


# ══════════════════════════════════════════════════════
#  ACT 10 — Conclusion
# ══════════════════════════════════════════════════════
class Act10Conclusion(Scene):
    """Summarises the paper contributions and key takeaways."""

    def construct(self):
        self.camera.background_color = C_BG_DARK

        heading = Tex(r"\text{Summary \& Contributions}", font_size=36, color=WHITE)
        heading.to_edge(UP, buff=0.5)
        self.play(Write(heading))

        # ── contribution bullets ──────────────────────────────
        bullets = [
            ("1.", "VFL with NVFlare for coral multi-omics",
             r"\text{Privacy-preserving cross-lab integration}", C_TRANSCRIPTOMICS),
            ("2.", "Domain-aware feature selection",
             r"\text{Gradient saliency reduces 106K }$\to$\text{ 500 features/omic}", C_PROTEOMICS),
            ("3.", "XAI/Gradient saliency-based missing omic inference",
             r"\text{Latent proxy from available omics replaces missing block}", C_METABOLOMICS),
            ("4.", "Holobiont bleaching prediction",
             r"\text{Thermal stress classification \& regression tasks}", C_MICROBIOME),
        ]

        rows = VGroup()
        for num, title, body, col in bullets:
            n   = Tex(r"\text{" + num + "}", font_size=22, color=col)
            t   = Tex(r"\text{" + title + "}", font_size=20, color=col)
            b   = Tex(body, font_size=16, color=C_SUBTITLE)
            row = VGroup(n, VGroup(t, b).arrange(DOWN, buff=0.08, aligned_edge=LEFT))
            row.arrange(RIGHT, buff=0.25)
            rows.add(row)
        rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT).shift(DOWN * 0.3 + LEFT * 0.5)

        for row in rows:
            self.play(Write(row), run_time=0.6)

        self.wait(0.8)

        # ── closing tag-line ──────────────────────────────────
        tagline = Tex(
            r"\text{Privacy-preserving}",
            r" $\cdot$ ",
            r"\text{Biologically interpretable}",
            r" $\cdot$ ",
            r"\text{Missing-data robust}",
            font_size=20, color=C_GRADIENT,
        ).to_edge(DOWN, buff=0.55)
        self.play(Write(tagline))
        self.wait(2.5)

        # ── fade out & final title ────────────────────────────
        self.play(FadeOut(VGroup(heading, rows, tagline)))

        final = VGroup(
            Tex(r"\text{Coral Holobiont Multi-Omics VFL}", font_size=36, color=WHITE),
            Tex(
                r"\text{Vertical Federated Learning}",
                r" $\times$ ",
                r"\text{Domain-Aware Selection}",
                r" $\times$ ",
                r"\text{XAI Inference}",
                font_size=18, color=C_SUBTITLE,
            ),
        ).arrange(DOWN, buff=0.3)
        self.play(Write(final), run_time=1.5)
        self.wait(3)
        self.play(FadeOut(final))


# ══════════════════════════════════════════════════════
#  FULL PRESENTATION  (chains all acts)
# ══════════════════════════════════════════════════════
class CoralVFLPresentation(MovingCameraScene):
    """
    Plays all acts in sequence as one continuous presentation.

    Render to video:
        manim render -qh coral_vfl_animation.py CoralVFLPresentation
    """

    def construct(self):
        self.camera.background_color = C_BG_DARK

        acts = [
            Act1Intro,
            Act2DataChallenge,
            Act3PrivacyProblem,
            Act4VFLArchitecture,
            Act5DomainAware,
            Act6MissingOmic,
            Act7XAIInference,
            Act8ExperimentSetup,
            Act9NVFlare,
            Act10Conclusion,
        ]

        frame = self.camera.frame
        for i, act_cls in enumerate(acts):
            # Call each act's construct as an unbound method on *this* scene,
            # so camera / renderer / mobject list are shared properly.
            act_cls.construct(self)
            if i < len(acts) - 1:
                # Cinematic inter-act breath: gentle zoom surge -> snap back
                self.play(
                    frame.animate.scale(1.10),
                    run_time=0.40, rate_func=rush_into,
                )
                self.play(
                    frame.animate.scale(1 / 1.10),
                    run_time=0.35, rate_func=rush_from,
                )
                self.wait(0.12)
