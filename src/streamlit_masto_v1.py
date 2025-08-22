import os
import io
import json
import zipfile
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.font_manager as fm
from PIL import ImageFont

# ---------- Fonts ----------
def choose_font():
    for name in [
        "Noto Sans CJK JP","Noto Sans CJK KR","Noto Sans CJK SC","Noto Sans CJK TC",
        "Noto Sans","Arial Unicode MS","Segoe UI Symbol","DejaVu Sans"
    ]:
        try:
            path = fm.findfont(name, fallback_to_default=False)
            if os.path.isfile(path):
                return path
        except Exception:
            continue
    return fm.findfont("DejaVu Sans")

# ---------- WordCloud optional ----------
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# ---------- App ----------
st.set_page_config(page_title="Mastodon Influencer Analysis", layout="wide")
st.title("📊 Mastodon Influencer Analysis")

# Fixed data directory (controls removed)
data_dir = "."

def load_csv(name, index_col=None):
    path = os.path.join(data_dir, name)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=index_col)
    return None

def load_json(name):
    path = os.path.join(data_dir, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# Load data
spearman = load_csv("spearman_active46.csv", index_col=0)
ridge_avg = load_csv("ridge_avgfavs_active46_coeffs.csv", index_col=0)
ridge_eff = load_csv("ridge_efficiency_active46_coeffs.csv", index_col=0)
hour_hist_global = load_csv("hour_hist_global.csv")
hour_hist_per_account = load_csv("hour_hist_per_account.csv")
posts_win = load_csv("posts_window.csv")
hashtags_counts = load_csv("timewindow_hashtags_counts.csv")
trending_tags = load_csv("trending_tags_7d.csv")

pad = dict(l=50, r=20, t=40, b=40)

# ------------------------------------------------------------
# Active Influencers (46) — summary table BEFORE correlations
# ------------------------------------------------------------
st.subheader("👥 Active Influencers (Top 46 by posting activity)")
st.caption("Scope: **992** accounts analyzed over the **past 7 weeks**; this table shows the **46 most active** posters.")

if posts_win is not None and not posts_win.empty and "handle" in posts_win.columns:
    # Aggregate per handle
    agg = posts_win.groupby("handle").agg(
        posts=("handle", "size"),
        avg_favs=("favs", "mean"),
        avg_boosts=("boosts", "mean"),
        avg_replies=("replies", "mean")
    ).reset_index()

    # Sort by posts desc and take top 46
    top46 = agg.sort_values("posts", ascending=False).head(46)

    # Round for clean presentation
    for col in ["avg_favs", "avg_boosts", "avg_replies"]:
        if col in top46.columns:
            top46[col] = top46[col].round(2)

    st.dataframe(top46, use_container_width=True)
else:
    st.info("`posts_window.csv` not found — cannot compute top 46 active influencers.")

# ------------------------------------------------------------
# Correlations
# ------------------------------------------------------------
st.subheader("📈 Correlations (Spearman)")
if spearman is not None:
    corr_df = spearman.copy()
    if corr_df.shape[0] != corr_df.shape[1]:
        corr_df = corr_df.corr(method="spearman")
    fig = px.imshow(
        corr_df,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="equal",
        labels=dict(color="Spearman ρ"),
    )
    fig.update_layout(
        template="plotly_white",
        margin=pad,
        coloraxis_colorbar=dict(title="ρ"),
        height=700
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Add **spearman_active46.csv** to show the heatmap.")

# ------------------------------------------------------------
# Coefficients
# ------------------------------------------------------------
st.subheader("🧮 Model Coefficients (Standardized)")
cols = st.columns(2)
def show_coef_bar(df, title, col):
    if df is None or df.empty:
        col.info(f"{title}: missing CSV")
        return
    dff = df.copy()
    if "coef_std" not in dff.columns:
        col.warning(f"{title}: no 'coef_std' column found.")
        return
    dff = dff.sort_values("coef_std", key=lambda s: s.abs(), ascending=True).tail(20)
    fig = px.bar(
        dff, x="coef_std", y=dff.index, orientation="h",
        labels=dict(coef_std="Std. Coefficient", y="Feature")
    )
    fig.update_layout(template="plotly_white", margin=pad, title=title)
    col.plotly_chart(fig, use_container_width=True)

show_coef_bar(ridge_avg, "Predicting Avg Favourites — Ridge", cols[0])
show_coef_bar(ridge_eff, "Predicting Efficiency — Ridge", cols[1])

# ------------------------------------------------------------
# Global hourly posting
# ------------------------------------------------------------
st.subheader("⏰ Posting Activity by Hour")
if hour_hist_global is not None and {"hour","count"} <= set(hour_hist_global.columns):
    fig = px.bar(hour_hist_global, x="hour", y="count",
                 labels=dict(hour="Hour (UTC)", count="Posts"),
                 title="Global Hourly Posting Distribution")
    fig.update_layout(template="plotly_white", bargap=0.05, margin=pad)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Provide **hour_hist_global.csv** with columns ['hour','count'].")

# ------------------------------------------------------------
# Daypart charts
# ------------------------------------------------------------
if posts_win is not None and "daypart" in posts_win.columns:
    dcols = st.columns(2)
    # volume by daypart
    vol = (posts_win.groupby("daypart").size().reset_index(name="posts"))
    order_map = {"Night":0,"Morning":1,"Afternoon":2,"Evening":3}
    vol["order"] = vol["daypart"].map(order_map).fillna(99)
    vol = vol.sort_values("order")
    fig1 = px.bar(vol, x="daypart", y="posts",
                  labels=dict(daypart="Daypart (UTC)", posts="Posts"),
                  title="Post volume by daypart (UTC, active)")
    fig1.update_layout(template="plotly_white", margin=pad)
    dcols[0].plotly_chart(fig1, use_container_width=True)

    # weighted avg likes/post by daypart
    if "favs" in posts_win.columns:
        favs = (posts_win.groupby("daypart")["favs"].mean().reset_index(name="avg_likes_per_post"))
        favs["order"] = favs["daypart"].map(order_map).fillna(99)
        favs = favs.sort_values("order")
        fig2 = px.bar(favs, x="daypart", y="avg_likes_per_post",
                      labels=dict(daypart="Daypart (UTC)", avg_likes_per_post="Avg likes/post"),
                      title="Weighted avg likes/post by daypart (UTC, active)")
        fig2.update_layout(template="plotly_white", margin=pad)
        dcols[1].plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# Per-account hourly heatmap
# ------------------------------------------------------------
if hour_hist_per_account is not None and {"handle","hour","count"} <= set(hour_hist_per_account.columns):
    with st.expander("Per-Account Hourly Heatmap"):
        totals = (hour_hist_per_account.groupby("handle")["count"].sum().sort_values(ascending=False))
        top_handles = totals.index.tolist()[:30]
        selected = st.multiselect("Select accounts", top_handles, default=top_handles)
        if selected:
            sub = hour_hist_per_account[hour_hist_per_account["handle"].isin(selected)].copy()
            piv = sub.pivot_table(index="handle", columns="hour", values="count", aggfunc="sum").fillna(0)
            fig = px.imshow(piv, color_continuous_scale="Blues", aspect="auto",
                            labels=dict(x="Hour (UTC)", y="Handle", color="Posts"), text_auto=False)
            fig.update_layout(template="plotly_white", height=400 + 10*len(selected), margin=pad)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Post Explorer
# ------------------------------------------------------------
st.subheader("🔎 Post Explorer")
if posts_win is not None:
    with st.expander("Filter & Explore Posts"):
        cols = st.columns(3)
        langs = sorted([x for x in posts_win["lang"].dropna().unique().tolist() if isinstance(x, str)])
        # default to 'en' if present; else first three
        default_langs = ["en"] if "en" in langs else (langs[:3] if langs else [])
        lang_sel = cols[0].multiselect("Language", langs, default=default_langs)
        with_media = cols[1].selectbox("Has media?", ["Any", "Yes", "No"])
        dayparts = sorted(posts_win["daypart"].dropna().unique().tolist())
        daypart_sel = cols[2].multiselect("Daypart", dayparts, default=dayparts)

        dfp = posts_win.copy()
        if lang_sel:
            dfp = dfp[dfp["lang"].isin(lang_sel)]
        if with_media != "Any" and "has_media" in dfp.columns:
            dfp = dfp[dfp["has_media"] == (with_media == "Yes")]
        if daypart_sel:
            dfp = dfp[dfp["daypart"].isin(daypart_sel)]

        if not dfp.empty:
            color_series = dfp["is_reply"].map({True:"Reply", False:"Original"}) if "is_reply" in dfp.columns else None
            fig = px.scatter(
                dfp, x="boosts", y="favs",
                color=color_series,
                hover_data=[c for c in ["handle","replies","has_media","lang","daypart"] if c in dfp.columns],
                labels=dict(x="Boosts", y="Favourites", color="Type"),
                title="Favourites vs Boosts"
            )
            fig.update_layout(template="plotly_white", margin=pad)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No posts match the filter.")
else:
    st.info("Provide **posts_window.csv** to enable the post explorer.")

# ------------------------------------------------------------
# Interactive Wordcloud + table
# ------------------------------------------------------------
st.subheader("💬 Hashtags")
wc_source = st.radio("Source", ["timewindow_hashtags_counts.csv", "trending_tags_7d.csv"], horizontal=True)
if wc_source == "timewindow_hashtags_counts.csv":
    dfw = hashtags_counts; word_col, count_col = "hashtag", "count"
else:
    dfw = trending_tags; word_col, count_col = "hashtag", "uses_7d"

def plot_interactive_wordcloud(df, word_col, count_col):
    if df is None or df.empty:
        st.info("Hashtag data not available.")
        return

    # frequencies (Unicode-safe)
    freq = {}
    for _, row in df.iterrows():
        w = str(row[word_col]).strip()
        if w.startswith("#"):
            w = w[1:]
        w = unicodedata.normalize("NFKC", w)
        c = float(row[count_col]) if pd.notnull(row[count_col]) else 0.0
        if c > 0:
            freq[w] = freq.get(w, 0.0) + c
    if not freq:
        st.info("No positive counts to display.")
        return

    if not WORDCLOUD_AVAILABLE:
        data = (pd.DataFrame({"Word": list(freq.keys()), "Count": list(freq.values())})
                  .sort_values("Count", ascending=False).reset_index(drop=True))
        st.dataframe(data, use_container_width=True)
        return

    chosen_font = choose_font()
    wc = WordCloud(
        width=1600, height=900, background_color="white",
        prefer_horizontal=0.95, collocations=False,
        font_path=chosen_font
    )
    wc.generate_from_frequencies(freq)
    img = wc.to_array()
    W, H = wc.width, wc.height

    def word_bbox(word, font_size, orientation, pos):
        x, y = pos
        try:
            font = ImageFont.truetype(chosen_font, font_size) if chosen_font else ImageFont.load_default()
            bb = font.getbbox(word)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = font_size * 0.6 * len(word), font_size
        if orientation:
            tw, th = th, tw
        x0, y0 = x, y
        x1, y1 = x + tw, y + th
        return [x0, x1, x1, x0, x0], [H - y0, H - y0, H - y1, H - y1, H - y0]

    fig = go.Figure()
    fig.add_trace(go.Image(z=img))
    for (w, f), font_size, position, orientation, color in wc.layout_:
        xs, ys = word_bbox(w, font_size, orientation, position)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            fill="toself", hoveron="fills",
            line=dict(width=0), opacity=0.01,
            hovertext=f"{w}: {int(freq[w])}",
            hoverinfo="text", showlegend=False
        ))
    fig.update_xaxes(visible=False, range=[0, W])
    fig.update_yaxes(visible=False, range=[H, 0])
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=20, b=10),
        title="Interactive Wordcloud (hover a word to see its count)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # table of words + counts
    table_df = (pd.DataFrame({"Word": list(freq.keys()), "Count": list(freq.values())})
                  .sort_values("Count", ascending=False).reset_index(drop=True))
    st.dataframe(table_df, use_container_width=True)

plot_interactive_wordcloud(dfw, word_col, count_col)

# ------------------------------------------------------------
# Export window
# ------------------------------------------------------------
st.subheader("📦 Export data files")
with st.expander("Create ZIP from current data folder"):
    if not os.path.isdir(data_dir):
        st.warning("Data folder not found.")
    else:
        file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                     if os.path.isfile(os.path.join(data_dir, f)) and any(f.endswith(ext) for ext in [".csv",".json",".png",".jpg",".jpeg",".txt"])]
        st.write(f"Files to include: {len(file_list)}")
        if st.button("Build ZIP archive"):
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fp in file_list:
                    zf.write(fp, arcname=os.path.basename(fp))
            mem.seek(0)
            st.download_button("Download data_export.zip", data=mem, file_name="data_export.zip", mime="application/zip")

st.caption("Built with Streamlit + Plotly. Wordcloud uses Plotly hover over exact word regions and a Unicode-capable font.")