"""HTML / CSS / SVG builders for the dashboard.

Everything here returns plain strings that app.py passes to st.markdown(...,
unsafe_allow_html=True). There is no Streamlit code in this file, so the look
can be tested by rendering the HTML in any browser.

Rules that keep the markup safe inside st.markdown:
  - the HTML is joined into one line (a blank line would end the HTML block),
  - no dollar signs in text (they can trigger math rendering).
"""
from __future__ import annotations

import math
import re
from html import escape

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
BLUE, RED, GREEN, PURPLE, AMBER, TEAL = "#3b82f6", "#f43f5e", "#22c55e", "#8b5cf6", "#f59e0b", "#14b8a6"
MUTED = "#8fa3c2"

STYLE = """
<style>
.stApp{background:radial-gradient(1100px 520px at 85% -8%,rgba(37,99,235,.28),transparent 60%),
radial-gradient(900px 480px at -5% 105%,rgba(14,165,233,.14),transparent 60%),linear-gradient(180deg,#060b18,#0a1328);}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer{display:none !important;}
.block-container{padding-top:1.16rem !important;padding-bottom:2rem !important;max-width:1500px !important;}
.fa-root{font-family:Inter,"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;color:#e6edf7;}
.fa-root *{box-sizing:border-box;}
.fa-head{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:4px 0 18px;flex-wrap:wrap;}
.fa-brand{display:flex;align-items:center;gap:16px;}
.fa-logo{width:54px;height:54px;border-radius:16px;display:flex;align-items:center;justify-content:center;
background:linear-gradient(145deg,rgba(59,130,246,.35),rgba(14,165,233,.12));border:1px solid rgba(96,165,250,.45);
box-shadow:0 0 28px rgba(59,130,246,.35);}
.fa-title{font-size:30px;font-weight:700;letter-spacing:-.4px;line-height:1.1;}
.fa-title span{color:#4f9dff;}
.fa-sub{color:#9db0cf;font-size:14px;margin-top:4px;}
.fa-pills{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
.fa-pill{display:flex;align-items:center;gap:8px;padding:9px 14px;border-radius:999px;font-size:13px;color:#cfe0ff;
background:rgba(18,30,56,.8);border:1px solid rgba(96,140,220,.3);}
.fa-dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
.fa-dot.live{background:#22c55e;box-shadow:0 0 10px #22c55e;animation:fapulse 1.4s infinite;}
.fa-dot.idle{background:#f59e0b;}
.fa-dot.done{background:#60a5fa;}
@keyframes fapulse{0%{opacity:1}50%{opacity:.35}100%{opacity:1}}
.fa-grid{display:grid;gap:16px;margin-bottom:16px;}
.fa-kpis{grid-template-columns:repeat(4,minmax(0,1fr));}
.fa-mid{grid-template-columns:minmax(0,2fr) minmax(0,1fr);}
.fa-bot{grid-template-columns:minmax(0,1fr) minmax(0,1.35fr) minmax(0,.95fr);}
@media (max-width:1100px){.fa-kpis{grid-template-columns:repeat(2,minmax(0,1fr));}.fa-mid,.fa-bot{grid-template-columns:minmax(0,1fr);}}
@media (max-width:640px){.fa-kpis{grid-template-columns:minmax(0,1fr);}.fa-title{font-size:24px;}}
.fa-card{background:linear-gradient(180deg,rgba(19,32,60,.86),rgba(11,20,40,.88));border:1px solid rgba(96,140,220,.22);
border-radius:18px;padding:18px 20px;box-shadow:0 10px 34px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,255,255,.05);min-width:0;}
.fa-ctitle{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:650;margin-bottom:12px;}
.fa-ctitle svg{flex:none;}
.fa-legend{margin-left:auto;display:flex;gap:16px;font-size:13px;color:#b7c6e0;font-weight:500;}
.fa-legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;}
.fa-kpi{position:relative;overflow:hidden;min-height:146px;}
.fa-kpi.blue{background:linear-gradient(135deg,rgba(37,99,235,.34),rgba(11,20,40,.9) 62%);border-color:rgba(59,130,246,.5);}
.fa-kpi.red{background:linear-gradient(135deg,rgba(244,63,94,.30),rgba(11,20,40,.9) 62%);border-color:rgba(244,63,94,.45);}
.fa-kpi.green{background:linear-gradient(135deg,rgba(34,197,94,.26),rgba(11,20,40,.9) 62%);border-color:rgba(34,197,94,.42);}
.fa-kpi.purple{background:linear-gradient(135deg,rgba(139,92,246,.32),rgba(11,20,40,.9) 62%);border-color:rgba(139,92,246,.48);}
.fa-klabel{display:flex;align-items:center;gap:10px;font-size:12.5px;letter-spacing:.9px;text-transform:uppercase;color:#a9bddb;font-weight:600;}
.fa-kvalue{font-size:38px;font-weight:700;letter-spacing:-.8px;margin:8px 0 6px;line-height:1.05;}
.fa-kdelta{font-size:13px;color:#9db0cf;display:flex;gap:6px;align-items:center;flex-wrap:wrap;}
.fa-up{color:#34d399;font-weight:650;}.fa-down{color:#fb7185;font-weight:650;}.fa-flat{color:#9db0cf;font-weight:650;}
.fa-spark{position:absolute;right:16px;bottom:44px;width:42%;max-width:200px;height:56px;}
.fa-donut{display:flex;align-items:center;gap:22px;justify-content:center;flex-wrap:wrap;padding:6px 0 2px;}
.fa-dlegend{display:flex;flex-direction:column;gap:14px;font-size:14px;}
.fa-dlegend b{display:block;font-size:15px;}
.fa-dlegend span{color:#9db0cf;font-size:13px;}
.fa-bars{display:flex;flex-direction:column;gap:13px;}
.fa-bar{display:grid;grid-template-columns:minmax(120px,1.1fr) 2fr 52px;align-items:center;gap:12px;font-size:13.5px;color:#d5e1f5;}
.fa-track{height:11px;border-radius:7px;background:rgba(255,255,255,.07);overflow:hidden;}
.fa-fill{height:100%;border-radius:7px;background:linear-gradient(90deg,#be123c,#fb7185);}
.fa-bar em{font-style:normal;text-align:right;color:#cfdcf2;font-weight:600;}
.fa-tablewrap{overflow-x:auto;}
.fa-table{width:100%;border-collapse:collapse;font-size:13.5px;}
.fa-table th{text-align:left;color:#8fa3c2;font-weight:600;font-size:12px;letter-spacing:.6px;text-transform:uppercase;
padding:9px 10px;background:rgba(255,255,255,.035);}
.fa-table td{padding:10px 10px;border-top:1px solid rgba(96,140,220,.12);color:#dbe6f8;white-space:nowrap;}
.fa-table tr:first-child td{border-top:none;}
.fa-tag{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12px;font-weight:650;}
.fa-tag.block{background:rgba(244,63,94,.18);color:#fda4af;border:1px solid rgba(244,63,94,.4);}
.fa-tag.review{background:rgba(245,158,11,.16);color:#fcd34d;border:1px solid rgba(245,158,11,.4);}
.fa-tag.yes{background:rgba(34,197,94,.16);color:#86efac;border:1px solid rgba(34,197,94,.38);}
.fa-tag.no{background:rgba(148,163,184,.14);color:#cbd5e1;border:1px solid rgba(148,163,184,.3);}
.fa-status{display:flex;flex-direction:column;gap:14px;}
.fa-srow{display:flex;align-items:center;gap:14px;}
.fa-sicon{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex:none;}
.fa-sicon.ok{background:rgba(34,197,94,.16);border:1px solid rgba(34,197,94,.5);}
.fa-sicon.warn{background:rgba(245,158,11,.16);border:1px solid rgba(245,158,11,.5);}
.fa-sicon.bad{background:rgba(244,63,94,.16);border:1px solid rgba(244,63,94,.5);}
.fa-sicon.info{background:rgba(59,130,246,.16);border:1px solid rgba(59,130,246,.5);}
.fa-stext b{display:block;font-size:15px;}
.fa-stext span{color:#9db0cf;font-size:13px;}
.fa-meter{height:7px;border-radius:5px;background:rgba(255,255,255,.08);margin-top:7px;overflow:hidden;width:100%;}
.fa-meter i{display:block;height:100%;border-radius:5px;background:linear-gradient(90deg,#2563eb,#38bdf8);}
.fa-empty{padding:46px 24px;text-align:center;}
.fa-empty h3{margin:0 0 8px;font-size:22px;}
.fa-empty p{margin:0;color:#9db0cf;font-size:15px;}
.fa-note{color:#7f93b3;font-size:12.5px;margin:6px 2px 0;}
.stButton>button,[data-testid="stPopover"] button{width:100% !important;border-radius:12px !important;border:1px solid rgba(96,140,220,.38) !important;
background:rgba(18,30,56,.85) !important;color:#e6edf7 !important;font-weight:600 !important;min-height:42px;}
[data-testid="stPopover"]{width:100% !important;}
.stButton>button:hover,[data-testid="stPopover"] button:hover{border-color:#3b82f6 !important;color:#fff !important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#2563eb,#0ea5e9) !important;border:none !important;color:#fff !important;}
.fa-tag.approve{background:rgba(59,130,246,.16);color:#93c5fd;border:1px solid rgba(59,130,246,.38);}
.fa-fill.down{background:linear-gradient(90deg,#15803d,#4ade80);}
.fa-bars.wide .fa-bar{grid-template-columns:minmax(160px,1.7fr) 1.3fr 46px;}
.fa-pred{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.7fr);gap:30px;align-items:start;}
@media (max-width:900px){.fa-pred{grid-template-columns:minmax(0,1fr);}}
.fa-risk-big{font-size:58px;font-weight:700;letter-spacing:-1.5px;line-height:1.05;margin:6px 0 10px;}
.fa-risk-big small{font-size:20px;color:#8fa3c2;font-weight:500;letter-spacing:0;}
.fa-riskbar{position:relative;height:12px;border-radius:7px;margin:18px 0 8px;background:linear-gradient(90deg,#16a34a,#f59e0b 70%,#f43f5e);}
.fa-riskbar .mk{position:absolute;top:-5px;width:2px;height:22px;background:rgba(255,255,255,.55);}
.fa-riskbar .pt{position:absolute;top:-6px;width:16px;height:24px;margin-left:-8px;border-radius:5px;background:#fff;border:3px solid #0b1428;box-shadow:0 0 12px rgba(255,255,255,.6);}
.fa-sub2{font-size:13px;color:#9db0cf;margin:14px 0 8px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;}
.fa-modehead h2{margin:0 0 4px;font-size:22px;}
.fa-modehead p{margin:0 0 14px;color:#9db0cf;font-size:14px;}
.fa-pills2{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 16px;}
.fa-pills2 .fa-pill{font-size:13px;padding:8px 14px;}
.fa-pill b{color:#fff;font-weight:650;}
.fa-tag.fraud{background:rgba(244,63,94,.18);color:#fda4af;border:1px solid rgba(244,63,94,.4);}
.fa-tag.legit{background:rgba(148,163,184,.14);color:#cbd5e1;border:1px solid rgba(148,163,184,.3);}
.fa-detail-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;margin-bottom:6px;}
.fa-detail-head h3{margin:0;font-size:19px;}
.fa-kv{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 18px;margin:12px 0 4px;font-size:13.5px;}
.fa-kv span{color:#8fa3c2;display:block;font-size:12px;text-transform:uppercase;letter-spacing:.5px;}
.fa-kv b{font-weight:600;color:#e6edf7;}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px !important;border-color:rgba(96,140,220,.22) !important;
background:linear-gradient(180deg,rgba(19,32,60,.86),rgba(11,20,40,.88));box-shadow:0 10px 34px rgba(0,0,0,.38);}
[class*="st-key-alert_"] button{width:100% !important;justify-content:flex-start !important;text-align:left !important;min-height:38px;padding:4px 12px !important;font-weight:500 !important;font-size:13px !important;}
[class*="st-key-alert_"] button p{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fa-ptitle{font-size:27px;font-weight:700;letter-spacing:-.5px;line-height:1.12;}
.fa-psub{color:#9db0cf;font-size:13.5px;margin-top:5px;}
/* ---- sidebar: the only place where the mode is chosen ---- */
section[data-testid="stSidebar"]{width:272px !important;min-width:272px !important;
background:linear-gradient(180deg,#070d1a 0%,#0a1428 58%,#070e1c 100%) !important;
border-right:1px solid rgba(96,140,220,.16) !important;box-shadow:4px 0 30px rgba(0,0,0,.32);}
section[data-testid="stSidebar"]>div{background:transparent !important;}
[data-testid="stSidebarHeader"]{height:0 !important;min-height:0 !important;padding:0 !important;overflow:hidden;}
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"]{display:none !important;}
[data-testid="stSidebarUserContent"]{padding:20px 14px 18px !important;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.32rem !important;}
.fa-side-brand{display:flex;align-items:center;gap:11px;padding:2px 6px 22px;}
.fa-side-logo{width:38px;height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;
background:linear-gradient(145deg,rgba(59,130,246,.38),rgba(14,165,233,.10));border:1px solid rgba(96,165,250,.45);
box-shadow:0 0 18px rgba(59,130,246,.30);}
.fa-side-name{font-size:15.5px;font-weight:700;letter-spacing:-.2px;line-height:1.15;}
.fa-side-sub{font-size:10.5px;color:#8fa3c2;letter-spacing:.13em;text-transform:uppercase;margin-top:3px;}
.fa-side-label{font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:#7f93b3;font-weight:700;
margin:4px 0 18px;padding-left:9px;}
section[data-testid="stSidebar"] .stButton>button{justify-content:flex-start !important;text-align:left !important;
gap:10px !important;padding:0 13px !important;min-height:43px !important;border-radius:12px !important;
background:transparent !important;border:1px solid transparent !important;color:#b6c6e2 !important;
font-weight:600 !important;box-shadow:none !important;
transition:background .18s ease,border-color .18s ease,color .18s ease,transform .18s ease;}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(59,130,246,.13) !important;
border-color:rgba(96,140,220,.30) !important;color:#eef5ff !important;transform:translateX(2px);}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:linear-gradient(135deg,rgba(37,99,235,.96),rgba(14,165,233,.78)) !important;
border:1px solid rgba(125,184,255,.42) !important;color:#fff !important;
box-shadow:0 8px 22px rgba(37,99,235,.32) !important;transform:none;}
section[data-testid="stSidebar"] .stButton>button p{font-size:14px !important;font-weight:600 !important;}
section[data-testid="stSidebar"] .stButton>button>div{justify-content:flex-start !important;width:100%;}
@media (prefers-reduced-motion:reduce){section[data-testid="stSidebar"] .stButton>button{transition:none !important;}}
.fa-tag.open{background:rgba(59,130,246,.16);color:#93c5fd;border:1px solid rgba(59,130,246,.38);}
.fa-tag.investigating{background:rgba(245,158,11,.16);color:#fcd34d;border:1px solid rgba(245,158,11,.4);}
.fa-tag.confirmed_fraud{background:rgba(244,63,94,.18);color:#fda4af;border:1px solid rgba(244,63,94,.4);}
.fa-tag.false_positive{background:rgba(34,197,94,.14);color:#86efac;border:1px solid rgba(34,197,94,.34);}
.fa-tag.closed{background:rgba(148,163,184,.14);color:#cbd5e1;border:1px solid rgba(148,163,184,.3);}
.fa-tag.prio-urgent{background:rgba(244,63,94,.16);color:#fda4af;border:1px solid rgba(244,63,94,.36);}
.fa-tag.prio-high{background:rgba(245,158,11,.14);color:#fcd34d;border:1px solid rgba(245,158,11,.34);}
.fa-tag.prio-medium{background:rgba(59,130,246,.14);color:#93c5fd;border:1px solid rgba(59,130,246,.32);}
.fa-tag.prio-low{background:rgba(148,163,184,.12);color:#cbd5e1;border:1px solid rgba(148,163,184,.26);}
.fa-case-text{font-size:14px;color:#dbe6f8;line-height:1.55;background:rgba(255,255,255,.03);
border:1px solid rgba(96,140,220,.16);border-radius:12px;padding:12px 14px;}
[class*="st-key-case_"] button{width:100% !important;justify-content:flex-start !important;text-align:left !important;
min-height:46px;padding:4px 14px !important;font-weight:500 !important;font-size:13.5px !important;}
[class*="st-key-case_"] button>div{justify-content:flex-start !important;width:100%;}
[class*="st-key-case_"] button p{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:13.5px !important;}
</style>
"""

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_ICONS = {
    "shield": '<path d="M12 3l7 3v5c0 5-3.5 8.5-7 10-3.5-1.5-7-5-7-10V6z"/>',
    "shield-check": '<path d="M12 3l7 3v5c0 5-3.5 8.5-7 10-3.5-1.5-7-5-7-10V6z"/><path d="M9 12l2 2 4-4"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.01"/>',
    "card": '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M3 10h18"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "bars": '<path d="M5 20V10M12 20V4M19 20v-7"/>',
    "list": '<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"/>',
    "bell": '<path d="M6 16v-5a6 6 0 0112 0v5l2 2H4z"/><path d="M10 21h4"/>',
    "pulse": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "cash": '<rect x="3" y="7" width="18" height="10" rx="2"/><circle cx="12" cy="12" r="2.5"/>',
    "check": '<path d="M5 12l4.5 4.5L19 7"/>',
    "pie": '<path d="M12 3a9 9 0 109 9h-9z"/><path d="M15 3.5A9 9 0 0120.5 9H15z"/>',
}


def compact(html: str) -> str:
    """Join into one line: a blank line would end the HTML block in markdown."""
    return re.sub(r"\s*\n\s*", "", html).strip()


def icon(name: str, color: str = "#8fb4ff", size: int = 22) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')


def fmt_count(x: float) -> str:
    return f"{int(round(x)):,}"


def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= limit:
            return f"{sign}{x / limit:,.2f}{suffix}"
    return f"{sign}{x:,.0f}"


def _fmt_tick(v: float) -> str:
    sign, v = ("-" if v < 0 else ""), abs(v)
    if v >= 1e6:
        return f"{sign}{v / 1e6:g}M"
    if v >= 1e3:
        return f"{sign}{v / 1e3:g}k"
    return f"{sign}{v:g}"


def _nice_ticks(vmax: float, n: int = 4, vmin: float = 0.0) -> list[float]:
    """Round axis values covering [vmin, vmax] (always including zero)."""
    hi, lo = max(vmax, 0.0), min(vmin, 0.0)
    if hi - lo <= 0:
        return [0.0, 1.0]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    first, last = math.floor(lo / step + 1e-9), math.ceil(hi / step - 1e-9)
    return [i * step for i in range(first, last + 1)]


def _smooth_path(pts: list[tuple[float, float]], ymin: float | None = None, ymax: float | None = None) -> str:
    """Catmull-Rom spline through the points, as an SVG path.

    ymin / ymax keep the curve inside the plot area (a smooth curve can
    otherwise dip below zero between two low points).
    """
    def clamp(y):
        if ymin is not None:
            y = max(y, ymin)
        if ymax is not None:
            y = min(y, ymax)
        return y

    if len(pts) == 1:
        return f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(len(pts) - 1):
        p0, p1, p2 = pts[max(i - 1, 0)], pts[i], pts[i + 1]
        p3 = pts[min(i + 2, len(pts) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, clamp(p1[1] + (p2[1] - p0[1]) / 6))
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, clamp(p2[1] - (p3[1] - p1[1]) / 6))
        d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d


def _delta(change, suffix: str = "%", good_when_up: bool = True, money: bool = False, points: bool = False) -> str:
    """'up 12.5% vs previous 6 h' line with a colour that reflects good or bad."""
    if change is None:
        return '<span class="fa-flat">no data yet</span>'
    if abs(change) < 1e-9:
        return '<span class="fa-flat">no change</span> vs previous 6 h'
    up = change > 0
    good = up == good_when_up
    arrow = "&#9650;" if up else "&#9660;"
    cls = "fa-up" if good else "fa-down"
    if money:
        text = fmt_money(abs(change))
    elif points:
        text = f"{abs(change):.1f} pts"
    else:
        text = f"{abs(change):.1f}{suffix}"
    return f'<span class="{cls}">{arrow} {text}</span> vs previous 6 h'


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def sparkline(values, color: str, uid: str) -> str:
    v = [float(x) for x in values][-24:]
    if len(v) < 2:
        return ""
    w, h, pad = 200, 56, 4
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    pts = [(pad + i * (w - 2 * pad) / (len(v) - 1), h - pad - (x - lo) / span * (h - 2 * pad)) for i, x in enumerate(v)]
    line = _smooth_path(pts, 0, h)
    area = f"{line} L{pts[-1][0]:.1f},{h} L{pts[0][0]:.1f},{h} Z"
    return (f'<svg class="fa-spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".38"/><stop offset="1" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient></defs><path d="{area}" fill="url(#{uid})"/>'
            f'<path d="{line}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round"/></svg>')


def kpi_card(label: str, value: str, delta_html: str, ic: str, color: str, tone: str,
             spark_values, uid: str) -> str:
    return (f'<div class="fa-card fa-kpi {tone}"><div class="fa-klabel">{icon(ic, color, 26)}{escape(label)}</div>'
            f'<div class="fa-kvalue">{value}</div><div class="fa-kdelta">{delta_html}</div>'
            f'{sparkline(spark_values, color, uid)}</div>')


def area_chart(x_labels: list[str], series: list[dict], uid: str, width: int = 760, height: int = 250,
               empty_hint: str = "Press Start to begin the replay") -> str:
    """Smooth multi-series area chart. series: [{'name', 'color', 'values'}]."""
    n = len(x_labels)
    left, right, top, bottom = 46, 14, 14, 30
    pw, ph = width - left - right, height - top - bottom
    if n == 0:                                  # nothing replayed yet: empty frame
        grid = "".join(
            f'<line x1="{left}" x2="{width - right}" y1="{top + ph * i / 4:.1f}" y2="{top + ph * i / 4:.1f}" '
            f'stroke="rgba(143,163,194,.16)" stroke-dasharray="2 5"/>' for i in range(5))
        return (f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block">{grid}'
                f'<text x="{left - 8}" y="{top + ph + 4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">0</text>'
                f'<text x="{width / 2:.0f}" y="{top + ph / 2:.0f}" text-anchor="middle" font-size="14" fill="{MUTED}">'
                f'{escape(empty_hint)}</text></svg>')
    vmax = max((max(s["values"]) for s in series if len(s["values"])), default=0)
    vmin = min((min(s["values"]) for s in series if len(s["values"])), default=0)
    ticks = _nice_ticks(float(vmax), vmin=float(vmin))
    ybot, ytop = ticks[0], ticks[-1]
    if ytop <= ybot:
        ytop = ybot + 1.0

    def px(i):
        return left + (pw * i / (n - 1) if n > 1 else pw / 2)

    def py(v):
        return top + ph - ((v - ybot) / (ytop - ybot)) * ph

    zero_y = py(0.0)

    parts = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;display:block">']
    parts.append("<defs>" + "".join(
        f'<linearGradient id="{uid}{i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{s["color"]}" stop-opacity=".42"/>'
        f'<stop offset="1" stop-color="{s["color"]}" stop-opacity=".02"/></linearGradient>' for i, s in enumerate(series)) + "</defs>")
    for t in ticks:
        y = py(t)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="rgba(143,163,194,.16)" stroke-dasharray="2 5"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">{_fmt_tick(t)}</text>')
    shown = sorted({round(i * (n - 1) / 5) for i in range(6)}) if n > 5 else list(range(n))
    for i in shown:
        anchor = "end" if (i == n - 1 and n > 1) else "middle"
        parts.append(f'<text x="{px(i):.1f}" y="{height - 8}" text-anchor="{anchor}" font-size="12" fill="{MUTED}">{escape(x_labels[i])}</text>')
    for i, s in enumerate(series):
        pts = [(px(j), py(float(v))) for j, v in enumerate(s["values"])]
        if not pts:
            continue
        line = _smooth_path(pts, top, top + ph)
        if len(pts) > 1:
            parts.append(f'<path d="{line} L{pts[-1][0]:.1f},{zero_y:.1f} L{pts[0][0]:.1f},{zero_y:.1f} Z" fill="url(#{uid}{i})"/>')
        parts.append(f'<path d="{line}" fill="none" stroke="{s["color"]}" stroke-width="2.4" stroke-linecap="round"/>')
        for j in range(0, len(pts), max(1, len(pts) // 12)):
            parts.append(f'<circle cx="{pts[j][0]:.1f}" cy="{pts[j][1]:.1f}" r="3.2" fill="{s["color"]}"/>')
        parts.append(f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="5" fill="{s["color"]}" stroke="#0b1428" stroke-width="2"/>')
    parts.append("</svg>")
    return "".join(parts)


def donut(segments: list[tuple[str, float, str]], center_big: str, center_small: str, size: int = 176, stroke: int = 24) -> str:
    """segments: [(label, value, color)]."""
    total = sum(v for _, v, _ in segments) or 1.0
    r = (size - stroke) / 2
    circ = 2 * math.pi * r
    cx = size / 2
    parts = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
             f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="{stroke}"/>']
    offset = 0.0
    live = [s for s in segments if s[1] > 0]
    gap = 3.0 if len(live) > 1 else 0.0
    for _, v, color in live:
        length = v / total * circ
        dash = max(length - gap, 0.5)
        parts.append(f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
                     f'stroke-dasharray="{dash:.2f} {circ - dash:.2f}" stroke-dashoffset="{-offset:.2f}" '
                     f'transform="rotate(-90 {cx} {cx})"/>')
        offset += length
    parts.append(f'<text x="{cx}" y="{cx + 2}" text-anchor="middle" font-size="26" font-weight="700" fill="#fff">{escape(center_big)}</text>')
    parts.append(f'<text x="{cx}" y="{cx + 22}" text-anchor="middle" font-size="12.5" fill="{MUTED}">{escape(center_small)}</text></svg>')
    legend = "".join(
        f'<div><b><i class="fa-dot" style="background:{c};margin-right:8px"></i>{escape(l)}</b>'
        f'<span>{v / total:.1%} &nbsp;({fmt_count(v)})</span></div>' for l, v, c in segments)
    return f'<div class="fa-donut">{parts[0]}{"".join(parts[1:])}<div class="fa-dlegend">{legend}</div></div>'


def bar_list(items: list[tuple[str, float]]) -> str:
    if not items:
        return '<div class="fa-note">No alerts yet.</div>'
    top = max(v for _, v in items) or 1.0
    rows = "".join(
        f'<div class="fa-bar"><span>{escape(label)}</span><div class="fa-track"><div class="fa-fill" style="width:{v / top * 100:.0f}%"></div></div>'
        f'<em>{v:.1%}</em></div>' for label, v in items)
    return f'<div class="fa-bars">{rows}</div>'


def alerts_table(rows: list[dict]) -> str:
    if not rows:
        return '<div class="fa-note">No alerts yet.</div>'
    body = "".join(
        f'<tr><td>{escape(r["hour"])}</td><td>{escape(r["type"].replace("_", " ").title())}</td>'
        f'<td>{r["amount"]:,.0f}</td><td>{r["risk"]:.1f}</td><td>{escape(r["reason"] or "-")}</td>'
        f'<td><span class="fa-tag {r["action"]}">{"Blocked" if r["action"] == "block" else "Review"}</span></td>'
        f'<td><span class="fa-tag {"yes" if r["fraud"] else "no"}">{"Fraud" if r["fraud"] else "False alarm"}</span></td></tr>'
        for r in rows)
    head = "".join(f"<th>{h}</th>" for h in ("Time", "Type", "Amount", "Risk", "Main reason", "Decision", "Outcome"))
    return f'<div class="fa-tablewrap"><table class="fa-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def status_card(view: dict, clock: str) -> str:
    drift = view["drift"]
    psi = view["psi"]
    tone, title = {
        "stable": ("ok", "Drift monitor: stable"),
        "watch": ("warn", "Drift monitor: watch"),
        "drift": ("bad", "Drift detected"),
        "warming": ("info", "Drift monitor: warming up"),
    }[drift]
    detail = "Waiting for enough traffic" if psi is None else f"Score shift index {psi:.2f} (last 24 h)"
    cap = view["capacity"]
    used = view["reviews_today"]
    pct = min(used / cap, 1.0) * 100 if cap else 0
    if view["pos"] == 0:
        replay_title, replay_text = "Ready to start", "Press Start to begin the replay"
    elif view["pos"] >= view["total"]:
        replay_title, replay_text = "Replay finished", f'All {view["total"]} hours replayed &middot; updated {escape(clock)}'
    else:
        replay_title, replay_text = "Replay in progress", f'Hour {view["pos"]} of {view["total"]} &middot; updated {escape(clock)}'
    return (
        '<div class="fa-status">'
        f'<div class="fa-srow"><div class="fa-sicon ok">{icon("check", "#4ade80", 22)}</div>'
        '<div class="fa-stext"><b style="color:#4ade80">Risk engine operational</b><span>Scoring every transaction</span></div></div>'
        f'<div class="fa-srow"><div class="fa-sicon {tone}">{icon("pulse", {"ok": "#4ade80", "warn": "#fbbf24", "bad": "#fb7185", "info": "#60a5fa"}[tone], 22)}</div>'
        f'<div class="fa-stext"><b>{title}</b><span>{detail}</span></div></div>'
        f'<div class="fa-srow" style="align-items:flex-start"><div class="fa-sicon info">{icon("list", "#60a5fa", 22)}</div>'
        f'<div class="fa-stext" style="flex:1"><b>Review queue today</b><span>{used} of {cap} reviews used</span>'
        f'<div class="fa-meter"><i style="width:{pct:.0f}%"></i></div></div></div>'
        f'<div class="fa-srow"><div class="fa-sicon info">{icon("clock", "#60a5fa", 22)}</div>'
        f'<div class="fa-stext"><b>{replay_title}</b><span>{replay_text}</span></div></div></div>')


def card(title: str, body: str, ic: str = "bars", legend: str = "") -> str:
    return (f'<div class="fa-card"><div class="fa-ctitle">{icon(ic, "#5aa2ff", 24)}<span>{escape(title)}</span>'
            f'{f"<div class=fa-legend>{legend}</div>" if legend else ""}</div>{body}</div>')


# --------------------------------------------------------------------------- #
# Page pieces
# --------------------------------------------------------------------------- #
def header_html(state: str, period: str, title: str) -> str:
    """Page heading: the mode name on the left, the data window and status on the right."""
    label, dot = {"ready": ("Ready", "idle"), "playing": ("Replay running", "live"),
                  "stopped": ("Stopped", "idle"), "finished": ("Replay finished", "done"),
                  "testing": ("Ready to test", "live"), "live": ("Simulated live", "live"),
                  "cases": ("Stored on disk", "done")}[state]
    return compact(
        f'<div class="fa-root"><div class="fa-head"><div>'
        f'<div class="fa-ptitle">{escape(title)}</div>'
        f'<div class="fa-psub">{escape(period)}</div></div>'
        f'<div class="fa-pills"><div class="fa-pill"><i class="fa-dot {dot}"></i>{label}</div></div></div></div>')


def sidebar_brand() -> str:
    """The product mark at the top of the sidebar."""
    return compact(
        f'<div class="fa-root"><div class="fa-side-brand">'
        f'<div class="fa-side-logo">{icon("shield-check", "#7db8ff", 21)}</div>'
        f'<div><div class="fa-side-name">Fraud Audit</div>'
        f'<div class="fa-side-sub">Risk Monitor</div></div></div></div>')


def page_animation(token: int, slug: str) -> str:
    """Fade the page in once per mode switch.

    The keyframe name carries the token, so it only restarts when the mode changes:
    a live feed that repaints every second keeps the same name and stays still.
    """
    return compact(
        f'<style>@keyframes faIn{token}{{from{{opacity:0;transform:translateY(8px)}}'
        f'to{{opacity:1;transform:none}}}}'
        f'.st-key-page-{slug}{{animation:faIn{token} .28s cubic-bezier(.22,.7,.3,1) both;}}'
        f'@media (prefers-reduced-motion:reduce){{.st-key-page-{slug}{{animation:none;}}}}</style>')


def dashboard_rows(view: dict, empty_hint: str = "Press Start to begin the replay") -> dict:
    """The cards shared by the replay and the live screen, as HTML strings."""
    m = view["metrics"]
    alerts = m["reviewed"] + m["blocked"]
    alert_rate = alerts / m["transactions"] if m["transactions"] else 0.0
    sp = view["recall_series"] * 100

    kpis = "".join([
        kpi_card("Transactions", fmt_count(m["transactions"]), _delta(view["tx_change"]),
                 "card", BLUE, "blue", view["tx_per_hour"], "k1"),
        kpi_card("Alerts raised", fmt_count(alerts),
                 _delta(view["alerts_change"], good_when_up=False), "alert", RED, "red", view["alerts_per_hour"], "k2"),
        kpi_card("Fraud caught", f'{m["recall"]:.1%}',
                 _delta(view["recall_change"], good_when_up=True, points=True), "shield-check", GREEN, "green", sp, "k3"),
        kpi_card("Net benefit", fmt_money(m["net_benefit"]),
                 _delta(view["net_change"], good_when_up=True, money=True), "cash", PURPLE, "purple", view["cum_net"], "k4"),
    ])

    trend = area_chart(view["hours"], [
        {"name": "Transactions", "color": BLUE, "values": view["tx_per_hour"]},
        {"name": "Alerts", "color": RED, "values": view["alerts_per_hour"]},
    ], "t", empty_hint=empty_hint)
    legend = f'<span><i style="background:{BLUE}"></i>Transactions / hour</span><span><i style="background:{RED}"></i>Alerts / hour</span>'
    decisions = donut([("Approved", m["approved"], BLUE), ("Sent to review", m["reviewed"], AMBER),
                       ("Blocked", m["blocked"], RED)], f"{alert_rate:.1%}", "Alert rate")

    money = area_chart(view["hours"], [
        {"name": "Fraud stopped", "color": GREEN, "values": view["cum_saved"]},
        {"name": "Fraud lost", "color": RED, "values": view["cum_lost"]},
    ], "m", height=220, empty_hint=empty_hint)
    money_legend = f'<span><i style="background:{GREEN}"></i>Fraud stopped</span><span><i style="background:{RED}"></i>Fraud lost</span>'
    return {
        "kpis": f'<div class="fa-grid fa-kpis">{kpis}</div>',
        "trends": f'<div class="fa-grid fa-mid">{card("Real-time trends", trend, "bars", legend)}{card("Decisions", decisions, "pie")}</div>',
        "money": ('<div class="fa-grid fa-mid">'
                  f'{card("Money impact (cumulative)", money, "cash", money_legend)}'
                  f'{card("Main reasons for alerts", bar_list(view["reason_shares"]), "bars")}</div>'),
    }


def dashboard_html(view: dict, clock: str) -> str:
    """The full dashboard for one replay state."""
    rows = dashboard_rows(view)
    return compact(
        '<div class="fa-root">'
        f'{rows["kpis"]}{rows["trends"]}{rows["money"]}'
        '<div class="fa-grid fa-bot" style="grid-template-columns:minmax(0,1.8fr) minmax(0,1fr)">'
        f'{card("Latest alerts", alerts_table(view["recent"]), "list")}'
        f'{card("System status", status_card(view, clock), "bell")}</div>'
        '<div class="fa-note">Simulation on PaySim data: legitimate transactions are sampled, so money totals compare settings rather than real losses.</div>'
        '</div>')


def advisor_html(rec, current_block: float, current_capacity: int) -> str:
    """Card with the cost curves and the recommended policy (see decision.recommend_policy)."""
    def block_label(b: float) -> str:
        return "never" if b >= 100 else f"{b:g}"

    blocks = area_chart([block_label(b) for b in rec.block_candidates],
                        [{"name": "Net benefit", "color": PURPLE, "values": list(rec.block_net)}], "ab", height=210)
    caps = area_chart([str(c) for c in rec.capacity_candidates],
                      [{"name": "Net benefit", "color": TEAL, "values": list(rec.capacity_net)}], "ac", height=210)
    gain = rec.best_net - rec.current_net
    if gain <= 1e-6:
        verdict = "Your current settings are already the best of the options tried."
    else:
        verdict = f"Switching to the recommendation would add {fmt_money(gain)} of net benefit."
    summary = (
        f'<div class="fa-note" style="font-size:14px;color:#cfdcf2;margin:0 0 12px">'
        f'<b>Recommended:</b> auto-block at risk {block_label(rec.best_block)} and {rec.best_capacity} reviews per day '
        f'(net benefit {fmt_money(rec.best_net)}). <b>Yours:</b> auto-block {block_label(current_block)}, '
        f'{current_capacity} reviews per day ({fmt_money(rec.current_net)}). {verdict}</div>')
    foot = ('<div class="fa-note">A backtest on the replayed transactions with your costs: it uses the true labels, '
            'so treat it as guidance, not a forecast. Each curve changes one setting and keeps the other as shown.</div>')
    body = (summary + '<div class="fa-grid" style="grid-template-columns:repeat(2,minmax(0,1fr));margin:0">'
            f'<div><div class="fa-klabel" style="margin-bottom:6px">Net benefit by auto-block risk score</div>{blocks}</div>'
            f'<div><div class="fa-klabel" style="margin-bottom:6px">Net benefit by daily review capacity</div>{caps}</div></div>' + foot)
    return compact(f'<div class="fa-root"><div style="margin-top:16px">{card("Policy advisor", body, "cash")}</div></div>')


# --------------------------------------------------------------------------- #
# Transaction testing (mode 3)
# --------------------------------------------------------------------------- #
DECISION_TEXT = {"approve": ("Approve", GREEN), "review": ("Send to review", AMBER), "block": ("Block", RED)}


def mode_head(title: str, text: str) -> str:
    return compact(f'<div class="fa-root fa-modehead"><h2>{escape(title)}</h2><p>{escape(text)}</p></div>')


def _reason_rows(reasons, up: bool) -> str:
    if not reasons:
        return '<div class="fa-note">Nothing notable.</div>'
    top = max(abs(r.contribution) for r in reasons) or 1.0
    cls = "fa-fill" if up else "fa-fill down"
    rows = "".join(
        f'<div class="fa-bar"><span>{escape(r.text)}</span><div class="fa-track">'
        f'<div class="{cls}" style="width:{abs(r.contribution) / top * 100:.0f}%"></div></div>'
        f'<em>{f"{r.share:.0%}" if up else ""}</em></div>' for r in reasons)
    return f'<div class="fa-bars wide">{rows}</div>'


def prediction_html(pred, policy) -> str:
    """Result card for one transaction (see predict.Prediction)."""
    label, color = DECISION_TEXT[pred.decision]
    notes = "".join(f'<div class="fa-note">{escape(n)}</div>' for n in pred.notes)
    if not pred.applicable:
        left = (f'<div class="fa-klabel">Risk score</div><div class="fa-risk-big" style="color:{MUTED}">n/a</div>'
                f'<span class="fa-tag approve">{label}</span>{notes}')
        right = ""
    else:
        marks = ""
        if policy.daily_capacity > 0 and policy.review_min_score <= 100:
            marks += f'<i class="mk" style="left:{policy.review_min_score:.1f}%" title="Review from {policy.review_min_score:g}"></i>'
        if policy.block_threshold <= 100:
            marks += f'<i class="mk" style="left:{policy.block_threshold:.1f}%" title="Block from {policy.block_threshold:g}"></i>'
        left = (f'<div class="fa-klabel">Risk score</div>'
                f'<div class="fa-risk-big" style="color:{color}">{pred.risk:.1f}<small> / 100</small></div>'
                f'<span class="fa-tag {pred.decision}" style="font-size:14px;padding:5px 16px">{label}</span>'
                f'<div class="fa-riskbar">{marks}<span class="pt" style="left:{min(pred.risk, 100):.1f}%"></span></div>'
                f'<div class="fa-note">White lines mark where this policy starts to review and to block.</div>{notes}')
        right = (f'<div class="fa-sub2" style="margin-top:0">What raised the risk</div>{_reason_rows(pred.reasons_up, True)}'
                 f'<div class="fa-sub2">What lowered it</div>{_reason_rows(pred.reasons_down, False)}'
                 '<div class="fa-note" style="margin-top:12px">Reasons show what the engine reacted to. '
                 'They are not proof of fraud.</div>')
    body = f'<div class="fa-pred"><div>{left}</div><div>{right}</div></div>'
    return compact(f'<div class="fa-root">{card("Risk assessment", body, "shield-check")}</div>')


def batch_summary_html(s: dict) -> str:
    """Four summary tiles for a scored file (see processing.summarize_batch)."""
    alert_rate = s["alerts"] / s["scored"] if s["scored"] else 0.0
    cards = [
        kpi_card("Rows scored", fmt_count(s["scored"]),
                 f'of {fmt_count(s["read"])} read' + (f' &middot; {fmt_count(s["rejected"])} rejected' if s["rejected"] else ""),
                 "card", BLUE, "blue", [], "b1"),
        kpi_card("Alerts raised", fmt_count(s["alerts"]),
                 f'{alert_rate:.1%} of scored &middot; {s["blocked"]} blocked', "alert", RED, "red", [], "b2"),
    ]
    if s["labelled"]:
        cards += [
            kpi_card("Fraud caught", f'{s["recall"]:.1%}', f'{s["fraud_caught"]} of {s["fraud_total"]} frauds &middot; '
                     f'precision {s["precision"]:.1%}', "shield-check", GREEN, "green", [], "b3"),
            kpi_card("Net benefit", fmt_money(s["net_benefit"]), "with your costs", "cash", PURPLE, "purple", [], "b4"),
        ]
    else:
        cards += [
            kpi_card("Highest risk", f'{s["max_risk"]:.1f}', "no labels in the file", "pulse", GREEN, "green", [], "b3"),
            kpi_card("Rows rejected", fmt_count(s["rejected"]), "see the list below" if s["rejected"] else "all rows were valid",
                     "list", PURPLE, "purple", [], "b4"),
        ]
    return compact(f'<div class="fa-root"><div class="fa-grid fa-kpis">{"".join(cards)}</div></div>')


def scored_table_html(rows, limit: int = 25) -> str:
    """Highest-risk transactions of a scored file."""
    top = rows.sort_values("risk", ascending=False).head(limit)
    labelled = "isFraud" in rows.columns and rows["isFraud"].notna().any()
    head = ["Row", "Type", "Amount", "Risk", "Main reason", "Decision"] + (["Outcome"] if labelled else [])
    body = []
    for r in top.itertuples():
        label, _ = DECISION_TEXT[r.decision]
        cells = (f'<td>{r.row}</td><td>{escape(r.type.replace("_", " ").title())}</td><td>{r.amount:,.0f}</td>'
                 f'<td>{r.risk:.1f}</td><td>{escape(r.reason or "-")}</td><td><span class="fa-tag {r.decision}">{label}</span></td>')
        if labelled:
            actual = getattr(r, "isFraud")
            cells += ('<td>-</td>' if actual != actual else
                      f'<td><span class="fa-tag {"yes" if actual == 1 else "no"}">{"Fraud" if actual == 1 else "Legitimate"}</span></td>')
        body.append(f"<tr>{cells}</tr>")
    table = (f'<div class="fa-tablewrap"><table class="fa-table"><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}'
             f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    title = (f"Highest-risk transactions (top {len(top)} of {len(rows)})"
             if len(top) < len(rows) else "Scored transactions")
    return compact(f'<div class="fa-root">{card(title, table, "list")}</div>')


# --------------------------------------------------------------------------- #
# Live simulation (mode 2)
# --------------------------------------------------------------------------- #
def live_status_strip(view: dict, clock: str, avg_ms: float, rate: float, processed: int = 0) -> str:
    """Compact system health under the title: clock, performance, drift and review queue."""
    drift_text = {"stable": ("Traffic drift: stable", "live"), "watch": ("Traffic drift: watch", "idle"),
                  "drift": ("Traffic drift: detected", "bad"), "warming": ("Traffic drift: warming up", "done")}[view["drift"]]
    psi = f' ({view["psi"]:.2f})' if view["psi"] is not None else ""
    used, cap = view["reviews_today"], view["capacity"]
    dot = {"bad": '<i class="fa-dot" style="background:#f43f5e;box-shadow:0 0 10px #f43f5e"></i>'}.get(
        drift_text[1], f'<i class="fa-dot {drift_text[1]}"></i>')
    timing = f"{avg_ms:.1f} ms" if avg_ms < 10 else f"{avg_ms:.0f} ms"
    pills = [
        f'<div class="fa-pill">Simulated time <b>{escape(clock)}</b></div>',
        f'<div class="fa-pill" title="Engine time per transaction in the last batches, the rate at which transactions '
        f'are arriving, and the total processed since the session started (in-memory rows are trimmed beyond 100,000)">'
        f'Performance <b>{timing}</b>/txn &middot; <b>{rate:.0f}</b> tx/s &middot; <b>{processed:,}</b> processed</div>',
        f'<div class="fa-pill" title="Population Stability Index of amounts and empty receivers: last 3 simulated hours '
        f'against the first 3 hours of the session. Under 0.10 stable, 0.10-0.25 watch, above 0.25 drift.">'
        f'{dot}{drift_text[0]}{psi}</div>',
        f'<div class="fa-pill">Review queue today <b>{used}</b> of <b>{cap}</b></div>',
    ]
    return compact(f'<div class="fa-root"><div class="fa-pills2">{"".join(pills)}</div></div>')


def live_dashboard_html(view: dict, strip: str) -> str:
    """Status strip, tiles and charts of the live screen (the feed and alerts are drawn separately)."""
    rows = dashboard_rows(view, empty_hint="Waiting for the first simulated transactions&hellip;")
    return compact(f'{strip}<div class="fa-root">{rows["kpis"]}{rows["trends"]}{rows["money"]}</div>')


def feed_html(rows, show_truth: bool) -> str:
    """Every transaction that just arrived (fraud or not), newest first."""
    from src.live import minute_label
    if len(rows) == 0:
        return compact(f'<div class="fa-root">{card("Live transactions", "<div class=fa-note>Waiting for the first transactions...</div>", "list")}</div>')
    head = ["Time", "Type", "Amount", "Risk", "Decision"] + (["True label"] if show_truth else [])
    body = []
    for r in rows.itertuples():
        label, _ = DECISION_TEXT[r.decision]
        risk = f"{r.risk:.1f}" if r.applicable else "not scored"
        cells = (f'<td>{minute_label(int(r.minute))}</td><td>{escape(r.type.replace("_", " ").title())}</td>'
                 f'<td>{r.amount:,.0f}</td><td>{risk}</td><td><span class="fa-tag {r.decision}">{label}</span></td>')
        if show_truth:
            cells += (f'<td><span class="fa-tag fraud">Fraud</span></td>' if r.isFraud == 1
                      else '<td><span class="fa-tag legit">Legitimate</span></td>')
        body.append(f"<tr>{cells}</tr>")
    table = (f'<div class="fa-tablewrap"><table class="fa-table"><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}'
             f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    return compact(f'<div class="fa-root">{card("Live transactions", table, "list")}</div>')


def alert_label(row) -> str:
    """Text of one clickable alert (Streamlit button labels support :color[...] markup)."""
    from src.live import minute_label
    colour = "red" if row.decision == "block" else "orange"
    action = "BLOCKED" if row.decision == "block" else "REVIEW"
    return f":{colour}[**{action}**]  {minute_label(int(row.minute))} · {row.type.replace('_', ' ').title()} · {row.amount:,.0f} · risk {row.risk:.1f}"


def alert_detail_html(row, prediction, policy, show_truth: bool, in_dialog: bool = False) -> str:
    """Everything about one alert: classification, score, reasons and facts.

    in_dialog=True gives a compact two-column layout without a card frame, for the
    pop-up window (which supplies its own title and close button).
    """
    from src.live import minute_label
    label, color = DECISION_TEXT[row.decision]
    marks = ""
    if policy.daily_capacity > 0 and policy.review_min_score <= 100:
        marks += f'<i class="mk" style="left:{policy.review_min_score:.1f}%" title="Review from {policy.review_min_score:g}"></i>'
    if policy.block_threshold <= 100:
        marks += f'<i class="mk" style="left:{policy.block_threshold:.1f}%" title="Block from {policy.block_threshold:g}"></i>'
    facts = [("Transaction", f"#{int(row.id)}"), ("Time", minute_label(int(row.minute))),
             ("Type", row.type.replace("_", " ").title()), ("Amount", f"{row.amount:,.2f}"),
             ("Sender balance before", f"{row.oldbalanceOrg:,.2f}"), ("Receiver balance before", f"{row.oldbalanceDest:,.2f}")]
    kv = "".join(f"<div><span>{k}</span><b>{escape(str(v))}</b></div>" for k, v in facts)
    truth = ""
    if show_truth:
        tag = '<span class="fa-tag fraud">Fraud</span>' if row.isFraud == 1 else '<span class="fa-tag legit">Legitimate</span>'
        extra = f" &middot; {escape(row.pattern)}" if row.isFraud == 1 and row.pattern else ""
        truth = (f'<div class="fa-sub2">True label (simulator only)</div>{tag}{extra}'
                 '<div class="fa-note">Known only to the simulator. The engine never sees it, and in real life it '
                 'arrives days later.</div>')
    pill = f'<span class="fa-tag {row.decision}" style="font-size:14px;padding:5px 16px">{label}</span>'
    reasons = (f'<div class="fa-sub2" style="margin-top:0">What raised the risk</div>{_reason_rows(prediction.reasons_up, True)}'
               f'<div class="fa-sub2">What lowered it</div>{_reason_rows(prediction.reasons_down, False)}')
    note = '<div class="fa-note" style="margin-top:10px">Reasons show what the engine reacted to. They are not proof of fraud.</div>'

    if in_dialog:
        left = (f'<div class="fa-klabel">Risk score</div>'
                f'<div class="fa-risk-big" style="color:{color};font-size:50px">{row.risk:.1f}<small> / 100</small></div>'
                f'{pill}<div class="fa-riskbar">{marks}<span class="pt" style="left:{min(row.risk, 100):.1f}%"></span></div>'
                f'<div class="fa-kv" style="grid-template-columns:1fr">{kv}</div>')
        return compact(f'<div class="fa-root"><div class="fa-pred" style="gap:26px"><div>{left}</div>'
                       f'<div>{reasons}{truth}{note}</div></div></div>')

    body = (
        f'<div class="fa-detail-head"><h3>Alert #{int(row.id)}</h3>{pill}</div>'
        f'<div class="fa-klabel">Risk score</div>'
        f'<div class="fa-risk-big" style="color:{color};font-size:44px">{row.risk:.1f}<small> / 100</small></div>'
        f'<div class="fa-riskbar">{marks}<span class="pt" style="left:{min(row.risk, 100):.1f}%"></span></div>'
        f'<div class="fa-kv">{kv}</div>{reasons}{truth}{note}')
    return compact(f'<div class="fa-root">{card("Alert details", body, "alert")}</div>')


def alert_hint_html() -> str:
    return compact('<div class="fa-root"><div class="fa-note" style="padding:14px 2px">'
                   'Click an alert to see the transaction, its classification, its score and the reasons.</div></div>')


def alerts_title_html(n_alerts: int) -> str:
    return compact(f'<div class="fa-root"><div class="fa-ctitle" style="margin-bottom:4px">{icon("bell", "#5aa2ff", 24)}'
                   f'<span>Alerts</span><div class="fa-legend"><span>{n_alerts:,} raised so far</span></div></div></div>')

# --------------------------------------------------------------------------- #
# Investigation cases
# --------------------------------------------------------------------------- #
_STATUS_LABEL = {"open": "Open", "investigating": "Investigating", "confirmed_fraud": "Confirmed fraud",
                 "false_positive": "False positive", "closed": "Closed"}
_PRIORITY_LABEL = {"low": "Low", "medium": "Medium", "high": "High", "urgent": "Urgent"}


def _stamp(value: str) -> str:
    """UTC timestamp from the database, shown in the analyst's local time."""
    try:
        from datetime import datetime
        return datetime.fromisoformat(value).astimezone().strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return "unknown"


def case_label(case) -> str:
    """One line per case in the list: enough to decide whether to open it."""
    return (f"{case.case_id}  ·  #{int(case.txn_id)}  ·  risk {case.risk:.1f}  ·  "
            f"{case.decision_label}  ·  {_STATUS_LABEL[case.status]}  ·  {_PRIORITY_LABEL[case.priority]}")


def cases_overview_html(counts: dict) -> str:
    """Status counters above the list."""
    order = ("open", "investigating", "confirmed_fraud", "false_positive", "closed")
    pills = "".join(
        f'<div class="fa-pill"><i class="fa-dot" style="background:{_STATUS_DOT[s]}"></i>'
        f'{_STATUS_LABEL[s]} <b>{counts.get(s, 0):,}</b></div>' for s in order)
    pills += f'<div class="fa-pill">All cases <b>{counts.get("all", 0):,}</b></div>'
    return compact(f'<div class="fa-root"><div class="fa-pills2">{pills}</div></div>')


_STATUS_DOT = {"open": "#60a5fa", "investigating": "#fbbf24", "confirmed_fraud": "#fb7185",
               "false_positive": "#4ade80", "closed": "#94a3b8"}


def case_detail_html(case) -> str:
    """Everything known about a case, above the editable fields in the pop-up."""
    facts = [("Case", case.case_id), ("Transaction", f"#{int(case.txn_id)}"),
             ("Amount", f"{case.amount:,.2f}" if case.amount else "-"),
             ("Type", (case.tx_type or "-").replace("_", " ").title()),
             ("Source", case.source.title()), ("Priority", _PRIORITY_LABEL[case.priority]),
             ("Created", _stamp(case.created_at)), ("Updated", _stamp(case.updated_at))]
    kv = "".join(f"<div><span>{k}</span><b>{escape(str(v))}</b></div>" for k, v in facts)
    explanation = escape(case.explanation) if case.explanation else "No explanation was recorded."
    notes = escape(case.notes).replace("\n", "<br>") if case.notes else "No notes yet."
    return compact(
        f'<div class="fa-root"><div class="fa-detail-head"><h3>{case.case_id} &middot; transaction #{int(case.txn_id)}</h3>'
        f'<div class="fa-pills" style="gap:8px"><span class="fa-tag {case.status}">{_STATUS_LABEL[case.status]}</span>'
        f'<span class="fa-tag prio-{case.priority}">{_PRIORITY_LABEL[case.priority]} priority</span></div></div>'
        f'<div class="fa-klabel">Risk score</div>'
        f'<div class="fa-risk-big" style="font-size:38px">{case.risk:.1f}<small> / 100</small></div>'
        f'<div class="fa-riskbar"><span class="pt" style="left:{min(case.risk, 100):.1f}%"></span></div>'
        f'<div class="fa-kv">{kv}</div>'
        f'<div class="fa-sub2">Main explanation</div><div class="fa-case-text">{explanation}</div>'
        f'<div class="fa-sub2">Analyst notes</div><div class="fa-case-text">{notes}</div></div>')


def cases_empty_html() -> str:
    return compact('<div class="fa-root"><div class="fa-card fa-empty">'
                   f'{icon("list", "#5aa2ff", 34)}<h3>No investigation cases yet</h3>'
                   '<p>Open an alert in the live simulation and choose <b>Open investigation case</b>. '
                   'Cases are stored on disk, so they are still here after a restart.</p></div></div>')
