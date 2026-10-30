import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

# Larger default text everywhere
plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 18,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
})

# =====================================================================
# 1. Load data
# =====================================================================
df = pd.read_csv("retail_data.csv")   # <-- adjust path

df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  format="%d/%m/%Y")

# =====================================================================
# 2. Data Enhancement
# =====================================================================
# Shipping time
df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
print(df["Shipping Days"].describe())
print("Negative shipping times:", (df["Shipping Days"] < 0).sum())

# Date parts
df["Year"]         = df["Order Date"].dt.year
df["Quarter"]      = df["Order Date"].dt.quarter
df["Month"]        = df["Order Date"].dt.month
df["Year-Month"]   = df["Order Date"].dt.to_period("M")
df["Year-Quarter"] = df["Order Date"].dt.to_period("Q")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
df["Weekday"] = pd.Categorical(df["Order Date"].dt.day_name(),
                               categories=weekday_order, ordered=True)

# =====================================================================
# 3. KPI definition
# =====================================================================
def kpis(data, by):
    out = data.groupby(by, observed=True).agg(
        Sales=("Sales", "sum"),
        Orders=("Order ID", "nunique"),
    )
    out["AOV"] = out["Sales"] / out["Orders"]
    return out.sort_values("Sales", ascending=False)

orders = df.groupby("Order ID").agg(
    Order_Value=("Sales", "sum"),
    Items=("Product ID", "count"),
    Segment=("Segment", "first"),
    Ship_Mode=("Ship Mode", "first"),
    Shipping_Days=("Shipping Days", "first"),
    Year=("Year", "first"),
)

# Headline KPIs
total_sales  = df["Sales"].sum()
total_orders = df["Order ID"].nunique()
aov          = total_sales / total_orders
avg_ship     = orders["Shipping_Days"].mean()

# Aggregated tables
seg     = kpis(df, "Segment")
reg     = kpis(df, "Region")
cat     = kpis(df, "Category")
monthly = kpis(df, "Year-Month").sort_index()

state_sales = kpis(df, "State")["Sales"]          # sorted, highest first
top_states  = state_sales.head(10)
rest_states = state_sales.iloc[10:]
state_share = top_states.sum() / total_sales

# Shipping reliability: per-order shipping days per mode + 90th percentile
mode_order   = ["Same Day", "First Class", "Second Class", "Standard Class"]
modes        = orders["Ship_Mode"].unique()
mode_list    = ([m for m in mode_order if m in modes] +     # fastest first
                [m for m in modes if m not in mode_order])
ship_groups  = [orders.loc[orders["Ship_Mode"] == m, "Shipping_Days"].values
                for m in mode_list]
ship_p90     = (orders.groupby("Ship_Mode")["Shipping_Days"]
                      .quantile(0.9).reindex(mode_list))

# Growth rates
year_sales = df.groupby("Year")["Sales"].sum()
yoy_total  = year_sales.pct_change() * 100               # % change vs previous year
n_years    = len(year_sales)
cagr = ((year_sales.iloc[-1] / year_sales.iloc[0]) ** (1 / (n_years - 1)) - 1
        if n_years > 1 else np.nan)
print("Months of data per year:", df.groupby("Year")["Month"].nunique().to_dict())


seg_year = df.pivot_table(index="Year", columns="Segment", values="Sales", aggfunc="sum")
seg_yoy  = seg_year.pct_change() * 100                    # YoY growth per segment

# =====================================================================
# 4. Formatting helpers
# =====================================================================
k_usd = lambda v: f"${v/1e3:,.0f}K"
usd   = lambda v: f"${v:,.0f}"
num   = lambda v: f"{v:,.0f}"
days  = lambda v: f"{v:.1f} d"
money_axis = mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K")

# ---- Colour palettes
ROW_PALETTES = {
    0: dict(dark="#252e5c", main="#3a4a8c", mid="#5b6db0", light="#9eaad6", pale="#e3e7f5"),  # indigo
    1: dict(dark="#1d3557", main="#2a6f97", mid="#4a90b8", light="#8fc1dc", pale="#d4e8f2"),  # blue
    2: dict(dark="#16425b", main="#1f6f8b", mid="#3d94a8", light="#86c2cd", pale="#d2eaee"),  # blue-teal
    3: dict(dark="#134e4a", main="#1d7a72", mid="#3fa093", light="#88c9bd", pale="#d3ece6"),  # teal
    4: dict(dark="#1b4332", main="#2d6a4f", mid="#4f9a72", light="#95c9a8", pale="#d8eedf"),  # sea green
}
ACCENT  = "#e76f51"      # shared highlight colour (P90 markers)
NEUTRAL = "#e4e9ed"      # "everything else" areas
GRID    = "#e6e6e6"

def use_palette(row):
    """Return the row's shades, a dark-to-light colormap and a category colour list."""
    p = ROW_PALETTES[row]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"row{row}", [p["pale"], p["light"], p["mid"], p["main"], p["dark"]])
    cats = [p["dark"], p["mid"], p["light"], p["pale"]]
    return p["dark"], p["main"], p["mid"], p["light"], p["pale"], cmap, cats

def text_on(color):
    """White text on dark fills, the row's dark shade on light fills."""
    r, g, b = mcolors.to_rgb(color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else DARK

def style(ax, title):
    ax.set_title(title, fontsize=18, fontweight="bold", loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)

def squarify_rects(values, x, y, w, h):
    """Squarified treemap layout: returns one (x, y, w, h) rectangle per value,
    with areas proportional to the values, filling the box (x, y, w, h)."""
    total = float(sum(values))
    areas = [v * w * h / total for v in values]
    rects = []

    def worst(row, side):
        s = sum(row)
        return max(max(side**2 * r / s**2, s**2 / (side**2 * r)) for r in row)

    def place(row, x, y, w, h):
        s = sum(row)
        if w >= h:                       # column along the left edge
            col_w, yy = s / h, y
            for r in row:
                rects.append((x, yy, col_w, r / col_w)); yy += r / col_w
            return x + col_w, y, w - col_w, h
        row_h, xx = s / w, x             # row along the top edge
        for r in row:
            rects.append((xx, y, r / row_h, row_h)); xx += r / row_h
        return x, y + row_h, w, h - row_h

    row, i = [], 0
    while i < len(areas):
        side = min(w, h)
        if not row or worst(row + [areas[i]], side) <= worst(row, side):
            row.append(areas[i]); i += 1
        else:
            x, y, w, h = place(row, x, y, w, h); row = []
    if row:
        place(row, x, y, w, h)
    return rects

def bar(ax, s, title, fmt, color, rotation=0):
    ax.bar(s.index.astype(str), s.values, color=color)
    ax.bar_label(ax.containers[0], labels=[fmt(v) for v in s.values],
                 padding=3, fontsize=13)
    ax.tick_params(axis="x", rotation=rotation)
    ax.margins(y=0.15)
    style(ax, title)

# =====================================================================
# 5. Dashboard
# =====================================================================
fig = plt.figure(figsize=(26, 30))
gs = GridSpec(5, 12, figure=fig, height_ratios=[0.45, 1.2, 1, 1.05, 1.1],
              hspace=0.6, wspace=0.3,
              top=0.95, bottom=0.03, left=0.045, right=0.965)   # tight outer margins
fig.suptitle("Retail Sales Dashboard", fontsize=34, fontweight="bold", y=0.985)

# ---- Row 0: KPI cards -----------------------------------------------
DARK, MAIN, MID, LIGHT, PALE, SEQ_CMAP, CAT_COLORS = use_palette(0)
cards = [("Total Sales", usd(total_sales)),
         ("Orders", num(total_orders)),
         ("Avg Order Value", f"${aov:,.2f}"),
         ("Avg Shipping Time", days(avg_ship)),
         (f"Sales CAGR {year_sales.index[0]}–{year_sales.index[-1]}", f"{cagr:+.1%}")]
kpi_gs = gs[0, :].subgridspec(1, len(cards), wspace=0.25)
for i, (label, value) in enumerate(cards):
    ax = fig.add_subplot(kpi_gs[0, i])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=PALE, edgecolor=MAIN,
                               lw=2, clip_on=False))
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=32,
            fontweight="bold", color=DARK, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=17,
            color="gray", transform=ax.transAxes)

# ---- Row 1: Monthly Sales & Orders with trend lines -----------------
DARK, MAIN, MID, LIGHT, PALE, SEQ_CMAP, CAT_COLORS = use_palette(1)
ax  = fig.add_subplot(gs[1, :])
ax2 = ax.twinx()
x = monthly.index.to_timestamp()
t = np.arange(len(monthly))

# Orders: light bars in the background (right axis), kept in the lower half
ax2.bar(x, monthly["Orders"], width=20, color=PALE, label="Orders")
slope_o, icpt_o = np.polyfit(t, monthly["Orders"], 1)
ax2.plot(x, slope_o * t + icpt_o, color=MID, lw=2.5, ls=":",
         label=f"Orders trend ({slope_o:+.1f} orders/month)")
ax2.set_ylim(0, monthly["Orders"].max() * 2.6)
ax2.set_ylabel("Orders", color=MID)
ax2.tick_params(axis="y", colors=MID)
ax2.spines["top"].set_visible(False)

# Sales: bold line in front (left axis)
ax.plot(x, monthly["Sales"], color=DARK, lw=3, marker="o", markersize=7, label="Sales")
slope_s, icpt_s = np.polyfit(t, monthly["Sales"], 1)
ax.plot(x, slope_s * t + icpt_s, color=MAIN, lw=2.5, ls=(0, (6, 3)),
        label=f"Sales trend ({slope_s:+,.0f} $/month)")
ax.set_ylim(0, monthly["Sales"].max() * 1.15)
ax.yaxis.set_major_formatter(money_axis)
ax.set_ylabel("Sales", color=DARK)
ax.tick_params(axis="y", colors=DARK)
ax.grid(axis="y", color=GRID)
ax.set_zorder(ax2.get_zorder() + 1)              # draw the sales line above the bars
ax.patch.set_visible(False)

# Quarterly date ticks: "Jan 2015", "Apr 2015", ...
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.margins(x=0.01)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower right", bbox_to_anchor=(1.0, 1.0),
          ncol=4, fontsize=13, frameon=False)
style(ax, "Monthly Sales & Orders")

# ---- Row 2: Category AOV and quarterly sales ---------------------
DARK, MAIN, MID, LIGHT, PALE, SEQ_CMAP, CAT_COLORS = use_palette(2)
ax = fig.add_subplot(gs[2, 0:6])
cat_colors = CAT_COLORS[:len(cat)]
wedges, _, autotexts = ax.pie(
    cat["AOV"].values,
    labels=cat.index,
    colors=cat_colors,
    autopct=lambda p: usd(p * cat["AOV"].sum() / 100),   # show the dollar AOV, not a %
    startangle=90, counterclock=False,
    wedgeprops=dict(edgecolor="white", linewidth=2, width=0.55),
    textprops=dict(fontsize=15), pctdistance=0.72,
)
for txt, col in zip(autotexts, cat_colors):
    txt.set_color(text_on(col)); txt.set_fontweight("bold")
# Keep the axes as wide as its cell (title stays aligned with the treemap below)
# and let the pie sit round and centred inside it
ax.set_autoscale_on(True)
ax.set_aspect("equal", adjustable="datalim")
ax.set_title("AOV by Category", fontsize=18, fontweight="bold", loc="left", pad=12)

ax = fig.add_subplot(gs[2, 7:12])
(df.pivot_table(index="Quarter", columns="Year", values="Sales", aggfunc="sum")
   .plot(kind="bar", ax=ax, rot=0, edgecolor="white",
         color=list(SEQ_CMAP(np.linspace(0.25, 1.0, df["Year"].nunique())))))
ax.yaxis.set_major_formatter(money_axis)
ax.set_ylim(top=ax.get_ylim()[1] * 1.2)                # headroom for the legend
ax.legend(loc="upper right", ncol=4, fontsize=13, frameon=False)
style(ax, "Quarterly Sales by Year")

# ---- Row 3: State treemap and shipping reliability ---------------
DARK, MAIN, MID, LIGHT, PALE, SEQ_CMAP, CAT_COLORS = use_palette(3)
ax = fig.add_subplot(gs[3, 0:6])
ax.set_axis_off()

# Work in the cell's real proportions (inches) so tiles come out roughly square
bbox = ax.get_position()
W = bbox.width  * fig.get_figwidth()
H = bbox.height * fig.get_figheight()

# Leave a band on the right for the remaining states, sized by their share of sales
rest_share = rest_states.sum() / total_sales
top_w      = W * (1 - rest_share)
rects      = squarify_rects(list(top_states.values), 0, 0, top_w, H)

def fit_fontsize(text, rw, rh, max_fs=20):
    """Largest font size (pt) at which the text block fits inside a rw x rh inch tile."""
    lines = text.split("\n")
    by_width  = rw * 72 / (0.75 * max(len(l) for l in lines))
    by_height = rh * 72 / (1.35 * len(lines))
    return min(max_fs, by_width, by_height)

shade = SEQ_CMAP(np.linspace(1.0, 0.3, len(top_states)))   # darker = more sales
for (rx, ry, rw, rh), name, val, col in zip(rects, top_states.index, top_states.values, shade):
    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, facecolor=col, edgecolor="white", lw=2))
    label = f"{name}\n{k_usd(val)}\n{val / total_sales:.0%}"
    fs = fit_fontsize(label, rw, rh)
    if fs < 6:                                  # tiny tile: name only
        label = name
        fs = fit_fontsize(label, rw, rh)
    if fs >= 5:
        ax.text(rx + rw / 2, ry + rh / 2, label, ha="center", va="center",
                fontsize=fs, color=text_on(col), fontweight="bold")

# Right-hand band: the other states as small grey tiles, with one summary label
ox, ow = top_w, W - top_w
for (rx, ry, rw, rh) in squarify_rects(list(rest_states.values), ox, 0, ow, H):
    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, facecolor=NEUTRAL,
                               edgecolor="white", lw=0.8))
ax.text(ox + ow / 2, H / 2,
        f"Other {len(rest_states)}\nstates\n{k_usd(rest_states.sum())}\n{rest_share:.0%}",
        ha="center", va="center", fontsize=15, color=DARK, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.3"))

ax.set_xlim(0, W)
ax.set_ylim(H, 0)                       # first (largest) tile in the top-left corner
ax.set_title(f"Sales by State: Top 10 = {state_share:.0%} of total",
             fontsize=18, fontweight="bold", loc="left", pad=12)

ax = fig.add_subplot(gs[3, 7:12])
bp = ax.boxplot(ship_groups, patch_artist=True, widths=0.55, showmeans=True,
                medianprops=dict(color=DARK, lw=2.5),
                meanprops=dict(marker="o", markerfacecolor="white",
                               markeredgecolor=DARK, markersize=8),
                flierprops=dict(marker=".", markersize=4, alpha=0.4))
for box in bp["boxes"]:
    box.set_facecolor(PALE); box.set_edgecolor(MID)
for k, p90 in enumerate(ship_p90.values, start=1):
    ax.scatter(k, p90, marker="D", color=ACCENT, edgecolor="white", s=90, zorder=5)
    ax.text(k + 0.3, p90, f"P90: {p90:.0f} d", va="center", ha="left", fontsize=13,
            color=ACCENT, fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
ax.set_xticks(range(1, len(mode_list) + 1),
              [f"{m}\n(n={len(g):,})" for m, g in zip(mode_list, ship_groups)])
ax.set_ylabel("Shipping days")
ax.set_xlim(0.5, len(mode_list) + 0.95)               # room for the last P90 label
ax.legend(handles=[
    Line2D([], [], color=DARK, lw=2.5, label="Median"),
    Line2D([], [], marker="o", ls="", markerfacecolor="white",
           markeredgecolor=DARK, label="Mean"),
    Line2D([], [], marker="D", ls="", color=ACCENT, markersize=9, label="90th percentile"),
], loc="upper center", ncol=3, fontsize=13, frameon=False)
ax.set_ylim(top=ax.get_ylim()[1] * 1.3)                 # headroom for the legend
style(ax, "Shipping Reliability by Ship Mode")

# ---- Row 4: Total sales per segment per year (ribbon chart) --------
DARK, MAIN, MID, LIGHT, PALE, SEQ_CMAP, CAT_COLORS = use_palette(4)
ax = fig.add_subplot(gs[4, :])
sales_y = seg_year.fillna(0)
y_labels = sales_y.index.astype(str)
n_y      = len(sales_y)
bar_w    = 0.4

colors = {s: CAT_COLORS[k % len(CAT_COLORS)] for k, s in enumerate(sales_y.columns)}

# In every year, stack segments by rank: highest sales on top
spans = {}                                  # (year index, segment) -> (bottom, top)
for i, (_, row) in enumerate(sales_y.iterrows()):
    y = 0
    for s in row.sort_values().index:
        spans[(i, s)] = (y, y + row[s])
        y += row[s]

# S-shaped curve for the ribbons between neighbouring bars
u = np.linspace(0, 1, 40)
smooth = u**2 * (3 - 2 * u)

for s in sales_y.columns:
    col = colors[s]
    for i in range(n_y):
        lo, hi = spans[(i, s)]
        ax.bar(i, hi - lo, bottom=lo, width=bar_w, color=col, edgecolor="white",
               lw=1, zorder=3, label=s if i == 0 else None)
        if hi - lo > 0:
            g = seg_yoy[s].iloc[i]
            seg_label = k_usd(hi - lo) + ("" if pd.isna(g) else f"\n{g:+.1f}% YoY")
            ax.text(i, (lo + hi) / 2, seg_label, ha="center", va="center",
                    fontsize=14, color=text_on(col), fontweight="bold", zorder=4)
        if i < n_y - 1:
            lo2, hi2 = spans[(i + 1, s)]
            xr = np.linspace(i + bar_w / 2, i + 1 - bar_w / 2, len(u))
            ax.fill_between(xr, lo + (lo2 - lo) * smooth, hi + (hi2 - hi) * smooth,
                            color=col, alpha=0.35, lw=0, zorder=2)

# Yearly total above each stack (stacking is meaningful here: segments add up)
totals = sales_y.sum(axis=1).values
for i, tot in enumerate(totals):
    g = yoy_total.iloc[i]
    tot_label = f"Total {k_usd(tot)}" + ("" if pd.isna(g) else f" ({g:+.1f}%)")
    ax.text(i, tot, tot_label, ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=DARK)

ax.set_xticks(range(n_y), y_labels)
ax.set_xlim(-0.5, n_y - 0.5)
ax.yaxis.set_major_formatter(money_axis)
ax.margins(y=0.1)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(sales_y.columns),
          fontsize=13, frameon=False)       # above the plot, inside the right margin
style(ax, "Total Sales per Segment per Year with YoY Growth (ranked: highest on top)")

# Save before showing: some backends clear the figure once the window closes
fig.savefig("sales_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
