#!/usr/bin/env python3
"""Build site/index.html, the weightsmith post, from site/scripts/post/template.html.

    python3 site/scripts/build_post.py            # write site/index.html
    python3 site/scripts/build_post.py --extract REV
        # one-time: copy the animated diagrams, the hero and the reference list out of the pages at git revision REV
        # into site/scripts/post/figs/, so the build no longer depends on those pages

Template tokens
    {{fig:NAME}}caption{{/fig}}      an imported figure (post/figs/NAME.html) with a numbered caption
    {{chart:NAME}}caption{{/chart}}  a chart drawn here from the numbers in CHARTS, with a numbered caption
    {{fign:ID}}                      "Figure N." for a figure written by hand in the template
    {{figref:ID}}                    "Figure N", a reference to any of the above
    {{c:key1,key2}}                  a citation; references are numbered in order of first use
    {{hero}}caption{{/hero}}         the opening video with its timeline (post/figs/hero.html)
    {{toc}} {{refs}} {{readtime}}    the contents list, the reference list, the reading time
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
POST = SITE / "scripts" / "post"
FIGS = POST / "figs"

# figures taken from the comparison page: name -> its generated region there
IMPORTED = {"where": "dg:where", "where-results": "dg:where-results", "fix": "dg:fix",
            "align": "dg:groot"}

# "The edit in four steps" is drawn by the comparison page's generator with this post's running example: π0.5 on the
# cube task, its two cameras, and the post's word for what the edit copies (activations).
SCHEMATIC = dict(
    o_words=["the cube", "then the banana", "in the bowl"],
    c_words=["the cube in the bowl", "then put the banana", "in the bowl"],
    cams=("pi05_ext", "pi05_wrist"),
    noun="activations",
    link="attends to activations",
    steps=["The original wording goes in. The VLM backbone attaches “in the bowl” to the banana, and the robot drops "
           "the cube.",
           "The clear wording goes in. The activations in the VLM backbone change, and the robot completes the task.",
           "Offline, we solve for the weight change that turns the first set of activations into the second. Each "
           "layer takes one least-squares solve, and no robot runs.",
           "Same words, edited weights. The original wording now produces the clear wording’s activations, and the "
           "robot completes the task."])

# citation keys in the order the comparison page numbered them (its reference list is the source of the entries)
OLD_KEYS = ("shridhar2021cliport ahn2022saycan fang2026counterfactual fei2026liberoplus kim2026liberopara "
            "abbas2026samegoal yang2026robolab brohan2023rt2 kim2024openvla nvidia2026cosmos3 black2024pi0 "
            "nvidia2025gr00tn1 askell2021context snell2022distilling meng2022rome meng2023memit turner2023steering "
            "zhang2026debugrobots pi2025pi05 abbas2026distilling shen2026wamsurvey kim2026cosmospolicy "
            "chi2023diffusionpolicy lipman2023flowmatching peebles2023dit nvidia2025isaaclab takehi2026sparks "
            "swann2026causal wang2024kesurvey cohen2024ripple wulff2026grounding fang2025alphaedit "
            "jiang2026groundinggap mcnemar1947 efron1979bootstrap wilson1927probable khazatsky2024droid").split()

# two reports the comparison page did not cite (paper/refs.bib has both)
EXTRA_REFS = {
    "guo2026robotweights": '<span class="ref-au">Guo, Xie.</span> <span class="ref-t">Weight Editing across Robot '
                           'Policies: VLA, WAM, and Humanoid Control.</span> Experiment thread on X, November 2026, '
                           'author-reported. <a href="https://x.com/Lingxiao234/status/2118048171649806424">'
                           'x.com/Lingxiao234</a>',
    "gupta2026robotweights": '<span class="ref-au">Gupta, Rohilla, Prashant, Zhu.</span> <span class="ref-t">Can '
                             'Frontier LLMs Fix Robots by Editing Their Weights?</span> Technical blog, November '
                             '2026. <a href="https://codnay.github.io/blog/llms-fix-robots-by-editing-weights/">'
                             'codnay.github.io</a>',
}

# The comparison page calls a run a "start". A run is one attempt at a task: the task's fixed scene plus a random
# seed. Runs whose seeds played no part in building an edit are held out. The imported drawings are reworded to match.
REWORD = [
    (r"\bunseen starts\b", "held-out runs"),
    (r"\bpractice starts\b", "practice runs"),
    (r"\bStarts the robot completed\b", "Runs the robot completed"),
    (r"\bshare of starts\b", "share of runs"),
    (r"\bstarts completed\b", "runs completed"),
    (r"\bstarts\b", "runs"),
    (r"\bStarts\b", "Runs"),
    (r", maybe chance\b", ", possibly by chance"),
]


# ---------------------------------------------------------------------------------------------------------- extract
def extract(rev):
    FIGS.mkdir(parents=True, exist_ok=True)
    show = lambda p: subprocess.check_output(["git", "show", f"{rev}:{p}"], cwd=SITE.parent).decode()
    cos, main = show("site/cosmos/index.html"), show("site/index.html")

    def region(name):
        return re.search(rf"<!-- gen:{re.escape(name)} -->(.*?)<!-- /gen:{re.escape(name)} -->", cos, re.S).group(1)

    for name, src in IMPORTED.items():
        frag = region(src)
        frag = frag[:frag.rindex("</figure>") + len("</figure>")]  # drop any paragraph the generator put after it
        (FIGS / f"{name}.html").write_text(frag + "\n")
    hero = re.search(r'<figure class="hero wide">.*?</figure>', main, re.S).group(0)
    (FIGS / "hero.html").write_text(hero + "\n")
    items = re.findall(r'<li id="ref-(\d+)">(.*?)</li>', region("references"), re.S)
    assert len(items) == len(OLD_KEYS), (len(items), len(OLD_KEYS))
    refs = {OLD_KEYS[int(n) - 1]: body for n, body in items}
    refs.update(EXTRA_REFS)
    (FIGS / "refs.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1) + "\n")
    print(f"extracted {len(IMPORTED)} figures, the hero and {len(refs)} references from {rev}")


# ----------------------------------------------------------------------------------------------------------- charts
def pct(a, n):
    return f"{100 * a / n:.1f}%"


def chart_rewordings():
    """Replays of 21 recorded drops with only the instruction changed (paper/scripts/make_figures.py, REWORDINGS)."""
    n = 21
    rows = [("“Put the cube then the banana in the bowl”", "the original", 21),
            ("“Put the rubiks cube then the banana in the bowl”", "", 21),
            ("“Put the banana then <b>the cube in the bowl</b>”", "", 3),
            ("“Put <b>the cube in the bowl</b> then the banana”", "", 3),
            ("“Put <b>the cube in the bowl</b> then put the banana in the bowl”", "the clear wording", 2),
            ("“Put <b>the cube in the bowl</b>”", "", 1)]
    out = ['<figure class="dgfig pc pc-bars reveal">',
           '<p class="dg-ttl">Drops that remain out of 21, by instruction</p>', '<div class="pc-rows">']
    for i, (lab, tag, v) in enumerate(rows):
        tag = f' <span class="pc-tag">{tag}</span>' if tag else ""
        val = f"<b>{v}</b> of {n}" if i == 0 else f"<b>{v}</b>"
        out.append(f'<div class="pc-row" style="--i:{i}" title="{v} of {n} drops remain"><p class="pc-lab">'
                   f'<span>{lab}{tag}</span></p><div class="pc-line"><span class="pc-track">'
                   f'<i class="pc-bar" style="width:{pct(v, n)}"></i></span><span class="pc-val">{val}</span></div></div>')
    out.append("</div>")
    return "\n".join(out)


def chart_pi05():
    """Paired held-out results on π0.5: (label, runs, unedited, compared policy, mark)."""
    groups = [
        ("Put the cube then the banana in the bowl", [
            ("Our edit", 128, 27, 106, "fix", "with our edit"),
            ("Compact version of the edit", 128, 19, 102, "fix", "with the compact edit"),
            ("Random edit of the same size", 64, 13, 9, "rand", "with a random edit of the same size"),
            ("Clear wording, no edit", 64, 13, 57, "ref", "unedited, given the clear wording"),
        ]),
        ("The same edit on two other tasks", [
            ("Pick up the eyeglasses", 64, 22, 44, "fix", "with our edit"),
            ("Put the cube to the left of the bowl", 64, 8, 36, "fix", "with our edit"),
        ]),
    ]
    out = ['<figure class="dgfig pc pc-dots reveal">',
           '<p class="dg-ttl">Share of held-out runs π0.5 completed</p>',
           '<ul class="dg-legend">'
           '<li><i class="pc-k pc-k-tick"></i>unedited, original wording</li>'
           '<li><i class="pc-k pc-dot pc-fix"></i>with our edit</li>'
           '<li><i class="pc-k pc-dot pc-rand"></i>with a random edit of the same size</li>'
           '<li><i class="pc-k pc-dot pc-ref"></i>unedited, given the clear wording</li></ul>']
    i = 0
    for head, rows in groups:
        out.append(f'<div class="pc-group"><p class="pc-gh">{head}</p>')
        for lab, n, u, e, mark, what in rows:
            tip = f"{lab}. Unedited with the original wording: {u} of {n} runs. {what.capitalize()}: {e} of {n} runs."
            out.append(f'<div class="pc-row" style="--u:{pct(u, n)};--e:{pct(e, n)};--i:{i}" title="{tip}">'
                       f'<p class="pc-lab"><span>{lab}</span><span class="pc-num">{u} → <b>{e}</b> of {n}</span></p>'
                       f'<div class="pc-rail"><span class="pc-tick"></span><span class="pc-run">'
                       f'<i class="pc-dot pc-{mark}"></i></span></div></div>')
            i += 1
        out.append("</div>")
    out.append('<div class="pc-axis" aria-hidden="true"><span>0</span><span>50%</span><span>100% of runs</span></div>')
    return "\n".join(out)


def chart_flips():
    """Paired outcomes that changed: (label, runs, failed→succeeded, succeeded→failed)."""
    rows = [("Our edit on Cube then banana", 128, 81, 2),
            ("Random edit of the same size", 64, 6, 10),
            ("An edit for the lookalike block", 64, 9, 16),
            ("One gripper bias value, on seven tasks", 240, 37, 37),
            ("No edit: the unedited policy run twice", 64, 17, 20)]
    full = 75.0  # each half spans 0 to 75% of the runs
    out = ['<figure class="dgfig pc pc-div reveal">',
           '<p class="dg-ttl">Runs whose outcome changed, as a share of the paired runs</p>',
           '<ul class="dg-legend pc-heads">'
           '<li>runs the edit broke<i class="pc-k pc-sw pc-broke"></i></li>'
           '<li><i class="pc-k pc-sw pc-fixb"></i>runs the edit fixed</li></ul>']
    for i, (lab, n, fixed, broke) in enumerate(rows):
        tip = (f"{lab}. Of {n} paired runs, {fixed} failed without the edit and succeeded with it, "
               f"and {broke} did the reverse.")
        out.append(f'<div class="pc-row" style="--i:{i}" title="{tip}"><p class="pc-lab"><span>{lab}</span>'
                   f'<span class="pc-num">{n} runs</span></p><div class="pc-split">'
                   f'<span class="pc-half pc-half-l"><b class="pc-n">{broke}</b><i class="pc-bar pc-broke" '
                   f'style="width:{100 * broke / n / full * 100:.1f}%"></i></span>'
                   f'<span class="pc-half pc-half-r"><i class="pc-bar pc-fixb" '
                   f'style="width:{100 * fixed / n / full * 100:.1f}%"></i><b class="pc-n">{fixed}</b></span></div></div>')
    out.append(f'<div class="pc-axis" aria-hidden="true"><span>{full:.0f}% of runs</span><span>0</span>'
               f'<span>{full:.0f}%</span></div>')
    return "\n".join(out)


def chart_side():
    """The world-action model's mustard edit on the tasks it was not built for, from cosmos/metrics.json."""
    sys.path.insert(0, str(SITE / "scripts"))
    import make_cosmos_metrics as gen
    cells = {r["key"]: r["cells"]["cosmos"]
             for t in json.loads((SITE / "cosmos" / "metrics.json").read_text())["tables"] if t["id"] == "side_effects"
             for r in t["rows"] if r["cells"].get("cosmos")}
    rows = [(gen.tn(tid), c["n"], c["a"], c["b"], c["p"] < gen.SIG and c["b"] < c["a"])
            for tid, c in cells.items() if not c.get("target")]
    assert len(rows) == 8 and all(b <= a or not worse for _, _, a, b, worse in rows), rows
    groups = [("Instructions that name the left side", [r for r in rows if "left" in r[0].lower()]),
              ("Other instructions", [r for r in rows if "left" not in r[0].lower()])]
    assert all(r[4] for r in groups[0][1]) and not any(r[4] for r in groups[1][1]), groups
    out = ['<figure class="dgfig pc pc-dots reveal">',
           '<p class="dg-ttl">Share of held-out runs the world-action model completed on other tasks</p>',
           '<ul class="dg-legend">'
           '<li><i class="pc-k pc-k-tick"></i>unedited</li>'
           '<li><i class="pc-k pc-dot pc-worse"></i>with the mustard edit, clearly worse</li>'
           '<li><i class="pc-k pc-dot pc-none"></i>with the mustard edit, no clear change</li></ul>']
    i = 0
    for head, group in groups:
        out.append(f'<div class="pc-group"><p class="pc-gh">{head}</p>')
        for lab, n, u, e, worse in group:
            tip = (f"{lab}. Unedited: {u} of {n} runs. With the mustard edit: {e} of {n} runs, "
                   f"{'clearly worse' if worse else 'no clear change'}.")
            same = " pc-same" if u == e else ""
            out.append(f'<div class="pc-row{same}" style="--u:{pct(u, n)};--e:{pct(e, n)};--i:{i}" title="{tip}">'
                       f'<p class="pc-lab"><span>{lab}</span><span class="pc-num">{u} → <b>{e}</b> of {n}</span></p>'
                       f'<div class="pc-rail"><span class="pc-tick"></span><span class="pc-run">'
                       f'<i class="pc-dot pc-{"worse" if worse else "none"}"></i></span></div></div>')
            i += 1
        out.append("</div>")
    out.append('<div class="pc-axis" aria-hidden="true"><span>0</span><span>50%</span><span>100% of runs</span></div>')
    return "\n".join(out)


CHARTS = {"rewordings": chart_rewordings, "pi05": chart_pi05, "flips": chart_flips, "side": chart_side}


# ------------------------------------------------------------------------------------------------------------ build
def schematic():
    sys.path.insert(0, str(SITE / "scripts"))
    import make_cosmos_metrics as gen
    return gen.dg_schematic_html(**SCHEMATIC)


def imported(name):
    frag = schematic() if name == "schematic" else (FIGS / f"{name}.html").read_text().strip()
    frag = re.sub(r"<figcaption>.*?</figcaption>", "", frag, flags=re.S)
    for pat, rep in REWORD:
        frag = re.sub(pat, rep, frag)
    frag = frag.replace('href="fig/', 'href="cosmos/fig/')  # the camera thumbnails live beside the records page
    if name in ("fix", "where-results"):  # static charts draw themselves when they scroll into view
        frag = frag.replace('<figure class="dgfig', '<figure class="reveal dgfig', 1)
    leftover = sorted(set(re.findall(r"[^<>\"]{0,40}\bstart\w*[^<>\"]{0,30}", frag)))
    for s in leftover:
        print(f"  check wording in {name}: …{s.strip()}…")
    return frag


def build():
    tpl = (POST / "template.html").read_text()
    refs = json.loads((FIGS / "refs.json").read_text())

    # figure numbers, in document order
    fig_no = {}
    for m in re.finditer(r"\{\{(fig|chart|fign):([\w-]+)\}\}", tpl):
        assert m.group(2) not in fig_no, f"figure id used twice: {m.group(2)}"
        fig_no[m.group(2)] = len(fig_no) + 1

    def cap(name, text):
        return f'<figcaption><span class="fig-n">Figure {fig_no[name]}.</span> {text.strip()}</figcaption>'

    def put_fig(m):
        frag = imported(m.group(1))
        i = frag.rindex("</figure>")
        return frag[:i] + cap(m.group(1), m.group(2)) + frag[i:]

    def put_chart(m):
        return CHARTS[m.group(1)]() + "\n" + cap(m.group(1), m.group(2)) + "</figure>"

    out = re.sub(r"\{\{fig:([\w-]+)\}\}(.*?)\{\{/fig\}\}", put_fig, tpl, flags=re.S)
    out = re.sub(r"\{\{chart:([\w-]+)\}\}(.*?)\{\{/chart\}\}", put_chart, out, flags=re.S)
    out = re.sub(r"\{\{fign:([\w-]+)\}\}", lambda m: f"Figure {fig_no[m.group(1)]}.", out)
    out = re.sub(r"\{\{figref:([\w-]+)\}\}", lambda m: f"Figure {fig_no[m.group(1)]}", out)

    def put_hero(m):  # the opening video keeps its markup (hero.js needs it); only the caption's sentences change
        frag = (FIGS / "hero.html").read_text().strip()
        return re.sub(r"<figcaption>.*?(?=<span class=\"scrub-hint\")", "<figcaption>" + m.group(1).strip() + " ",
                      frag, count=1, flags=re.S)

    out = re.sub(r"\{\{hero\}\}(.*?)\{\{/hero\}\}", put_hero, out, flags=re.S)

    # citations, numbered by first use
    order = []

    def title(key):
        t = re.search(r'<span class="ref-t">(.*?)</span>', refs[key], re.S).group(1)
        return html.escape(html.unescape(t).rstrip("."), quote=True)

    def put_cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        for k in keys:
            assert k in refs, f"unknown citation key: {k}"
            if k not in order:
                order.append(k)
        links = ", ".join(f'<a href="#ref-{order.index(k) + 1}" title="{title(k)}">{order.index(k) + 1}</a>'
                          for k in keys)
        return f'<span class="cite" data-ref="{",".join(keys)}">[{links}]</span>'

    out = re.sub(r"\{\{c:([\w,\s]+)\}\}", put_cite, out)
    ref_list = ('<ol class="refs">' + "".join(f'<li id="ref-{i}">{refs[k]}</li>' for i, k in enumerate(order, 1))
                + "</ol>")
    out = out.replace("{{refs}}", ref_list)

    # contents: every section's h2, with its h3s that carry an id
    toc = []
    for sec in re.finditer(r'<section id="([\w-]+)"[^>]*>(.*?)</section>', out, re.S):
        h2 = re.search(r"<h2[^>]*>(.*?)</h2>", sec.group(2), re.S)
        if not h2 or sec.group(1) == "tldr":
            continue
        subs = re.findall(r'<h3 id="([\w-]+)">(.*?)</h3>', sec.group(2), re.S)
        inner = ("<ol>" + "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in subs) + "</ol>") if subs else ""
        toc.append(f'<li><a href="#{sec.group(1)}">{h2.group(1)}</a>{inner}</li>')
    out = out.replace("{{toc}}", '<ol class="toc-l1">' + "".join(toc) + "</ol>")

    # reading time counts the prose and captions, not the contents list, tables, step lists or references
    body = out[out.index("<main"):out.index("</main>")]
    body = re.sub(r'<section id="references".*?</section>', " ", body, flags=re.S)
    body = re.sub(r'<ol class="anim-steps.*?</ol>', " ", body, flags=re.S)
    body = re.sub(r"<(svg|script|style|pre|nav|table)\b.*?</\1>", " ", body, flags=re.S)
    words = len(re.sub(r"<[^>]+>", " ", body).split())
    out = out.replace("{{readtime}}", f"{max(1, round(words / 230))} min read")

    left = re.findall(r"\{\{/?(?:fig|chart|fign|figref|c|hero|toc|refs|readtime)\b[^}]*\}\}", out)
    assert not left, f"unresolved tokens: {left[:5]}"
    (SITE / "index.html").write_text(out)
    print(f"wrote site/index.html: {len(out) // 1024} KB, {words} words, {len(fig_no)} figures, {len(order)} references")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--extract":
        extract(sys.argv[2])
    else:
        build()
