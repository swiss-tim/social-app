import streamlit as st
import requests, datetime as dt, math, time
from collections import Counter, defaultdict
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Data Fetching and Analysis Functions (from user script) ---
INSTANCES = [
    "mastodon.social", "mas.to", "mstdn.social", "hachyderm.io",
    "fosstodon.org", "infosec.exchange", "universeodon.com", "techhub.social"
]
HASHTAG_LIMIT = 20
ACCOUNT_LIMIT = 20
STATUS_LIMIT_PER_PAGE = 40
MAX_STATUSES = 400

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mastodon-Analytics/1.0"})

def get_json(url, params=None):
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def trending_tags(instance):
    url = f"https://{instance}/api/v1/trends/tags"
    return get_json(url, {"limit": HASHTAG_LIMIT})

@st.cache_data(ttl=3600)
def trending_accounts(instance):
    url = f"https://{instance}/api/v1/trends/accounts"
    return get_json(url, {"limit": ACCOUNT_LIMIT})

def lookup_account(instance, handle):
    url = f"https://{instance}/api/v1/accounts/lookup"
    return get_json(url, {"acct": handle})

def fetch_account_by_id(instance, account_id):
    url = f"https://{instance}/api/v1/accounts/{account_id}"
    return get_json(url)

@st.cache_data(ttl=3600)
def fetch_statuses(instance, account_id, max_total=MAX_STATUSES):
    url = f"https://{instance}/api/v1/accounts/{account_id}/statuses"
    all_statuses = []
    max_id = None
    while len(all_statuses) < max_total:
        params = {"limit": STATUS_LIMIT_PER_PAGE, "exclude_replies": False, "exclude_reblogs": False}
        if max_id: params["max_id"] = max_id
        try:
            data = get_json(url, params=params)
            if not data: break
            all_statuses.extend(data)
            max_id = data[-1]["id"]
            time.sleep(0.5)
        except Exception as e:
            st.warning(f"Could not fetch all statuses for account {account_id} from {instance}: {e}")
            break
    return all_statuses[:max_total]

def aggregate_hashtags(instances):
    tag_agg = defaultdict(lambda: {"instances": set(), "uses_7d": 0, "accounts_7d": 0})
    for inst in instances:
        try:
            for tag in trending_tags(inst):
                name = tag["name"].lower()
                hist = tag.get("history", [])
                uses = sum(int(h.get("uses", 0)) for h in hist)
                accts = sum(int(h.get("accounts", 0)) for h in hist)
                tag_agg[name]["instances"].add(inst)
                tag_agg[name]["uses_7d"] += uses
                tag_agg[name]["accounts_7d"] += accts
        except Exception:
            continue
    rows = [{"hashtag": "#" + name, "instances_seen": len(d["instances"]), "uses_7d_total": d["uses_7d"], "accounts_7d_total": d["accounts_7d"]} for name, d in tag_agg.items()]
    df = pd.DataFrame(rows).sort_values(["uses_7d_total", "instances_seen"], ascending=[False, False])
    return df.reset_index(drop=True)

def aggregate_trending_accounts(instances):
    best = {}
    for inst in instances:
        try:
            for acc in trending_accounts(inst):
                handle = f'{acc["acct"]}@{inst}' if "@" not in acc["acct"] else acc["acct"]
                cur = best.get(handle)
                if (not cur) or (acc["followers_count"] > cur["followers_count"]):
                    acc_copy = acc.copy()
                    acc_copy["_home_instance"] = inst
                    acc_copy["_handle"] = handle
                    best[handle] = acc_copy
        except Exception:
            continue
    df = pd.DataFrame(best.values())
    if not df.empty:
        df = df.sort_values("followers_count", ascending=False)
    return df.reset_index(drop=True)

def statuses_to_metrics(statuses):
    if not statuses:
        return {"n_posts": 0, "days_span": 0, "posts_per_day": 0.0, "median_interval_min": None, "hour_hist": {}, "dow_hist": {}, "avg_boosts": 0.0, "avg_favs": 0.0, "avg_replies": 0.0}
    ts = [pd.to_datetime(s["created_at"], utc=True) for s in statuses]
    ts_sorted = sorted(ts)
    span_days = max(1, (ts_sorted[-1] - ts_sorted[0]).total_seconds() / 86400)
    intervals_min = [(t2 - t1).total_seconds() / 60.0 for t1, t2 in zip(ts_sorted, ts_sorted[1:])]
    median_interval = float(pd.Series(intervals_min).median()) if intervals_min else None
    hour_hist = pd.Series([t.hour for t in ts]).value_counts().sort_index().to_dict()
    dow_hist = pd.Series([t.dayofweek for t in ts]).value_counts().sort_index().to_dict()
    boosts = pd.Series([s.get("reblogs_count", 0) for s in statuses]).mean()
    favs = pd.Series([s.get("favourites_count", 0) for s in statuses]).mean()
    replies = pd.Series([s.get("replies_count", 0) for s in statuses]).mean()
    return {
        "n_posts": len(statuses), "days_span": round(span_days, 1), "posts_per_day": round(len(statuses) / span_days, 2),
        "median_interval_min": round(median_interval, 1) if median_interval is not None else None,
        "hour_hist": hour_hist, "dow_hist": dow_hist,
        "avg_boosts": round(float(boosts), 2), "avg_favs": round(float(favs), 2), "avg_replies": round(float(replies), 2),
    }

@st.cache_data(ttl=3600)
def analyze_accounts(handles, default_instance="mastodon.social"):
    out = []
    for handle in handles:
        user, _, domain = handle.partition("@")
        domain = domain or default_instance
        try:
            acc = lookup_account(domain, handle)
            acct_id, home = acc["id"], domain
            profile = fetch_account_by_id(home, acct_id)
            statuses = fetch_statuses(home, acct_id)
            metrics = statuses_to_metrics(statuses)
            out.append({
                "handle": f'{acc["acct"]}@{home}', "id": acct_id, "followers": profile.get("followers_count"), "following": profile.get("following_count"),
                "statuses_total": profile.get("statuses_count"), "created_at": profile.get("created_at"), "last_status_at": profile.get("last_status_at"),
                **metrics
            })
        except Exception as e:
            st.error(f"Failed to analyze {handle}: {e}")
            continue
    return pd.DataFrame(out)

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Mastodon User Analytics Dashboard")
st.write("This app analyzes trends and user activity on Mastodon.")

# --- Sidebar for Controls ---
st.sidebar.header("Controls")
num_prominent_users = st.sidebar.slider("Number of Prominent Users to Analyze", 1, 10, 3)
influencer_handle = st.sidebar.text_input("Influencer Handle for Comparison", "Gargron@mastodon.social")

# --- Main Data Loading and Processing ---
with st.spinner("Gathering prevalent hashtags and trending accounts..."):
    df_tags = aggregate_hashtags(INSTANCES)
    df_acc = aggregate_trending_accounts(INSTANCES)

# --- Display Trending Data ---
st.header("Mastodon Trends")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trending Hashtags (Last 7 Days)")
    st.dataframe(df_tags.head(25), height=400)

with col2:
    st.subheader("Trending Accounts")
    if not df_acc.empty:
        cols = ["_handle", "followers_count", "statuses_count", "last_status_at", "_home_instance"]
        st.dataframe(df_acc[cols].head(25), height=400)
    else:
        st.warning("No trending accounts returned. Some instances may have disabled public trends.")

# --- Analyze and Display Behavioral Metrics ---
st.header("User Behavior Analysis")
if not df_acc.empty:
    top_handles = []
    for h in df_acc["_handle"].head(num_prominent_users).tolist():
        if "@" not in h:
            h += "@" + df_acc[df_acc["_handle"] == h]["_home_instance"].iloc[0]
        top_handles.append(h)

    handles_to_analyze = list(dict.fromkeys(top_handles + [influencer_handle]))

    with st.spinner(f"Performing deep-dive analysis on {len(handles_to_analyze)} accounts..."):
        df_beh = analyze_accounts(handles_to_analyze)

    if not df_beh.empty:
        st.subheader("Comparative Metrics")
        show_cols = ["handle", "followers", "posts_per_day", "median_interval_min", "avg_boosts", "avg_favs", "avg_replies"]
        st.dataframe(df_beh[show_cols])

        st.subheader("Engagement Windows (UTC)")
        for i, row in df_beh.iterrows():
            st.markdown(f"#### Activity for `{row['handle']}`")
            
            # Prepare data for charts
            hour_data = pd.DataFrame(list(row['hour_hist'].items()), columns=['Hour (UTC)', 'Posts'])
            dow_data = pd.DataFrame(list(row['dow_hist'].items()), columns=['Day of Week', 'Posts'])
            dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            dow_data['Day of Week'] = dow_data['Day of Week'].map(dow_map)

            # Hour of Day Chart
            chart_hour = alt.Chart(hour_data).mark_bar().encode(
                x=alt.X('Hour (UTC):O', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Posts:Q', title='Number of Posts'),
                tooltip=['Hour (UTC)', 'Posts']
            ).properties(
                title='Posting Activity by Hour of Day'
            )
            
            # Day of Week Chart
            chart_dow = alt.Chart(dow_data).mark_bar().encode(
                x=alt.X('Day of Week:N', sort=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
                y=alt.Y('Posts:Q', title='Number of Posts'),
                tooltip=['Day of Week', 'Posts']
            ).properties(
                title='Posting Activity by Day of Week'
            )

            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(chart_hour, use_container_width=True)
            with c2:
                st.altair_chart(chart_dow, use_container_width=True)
    else:
        st.error("Could not retrieve behavioral data for the selected accounts.")
else:
    st.info("Skipping behavioral analysis as no trending accounts were found.")

# --- Display Trending Data ---
st.header("Mastodon Trends")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trending Hashtags (Last 7 Days)")
    st.dataframe(df_tags.head(25), height=400)

    # --- Hashtag Wordcloud ---
    st.subheader("Hashtag Wordcloud")
    hashtag_freq = dict(zip(df_tags["hashtag"], df_tags["uses_7d_total"]))
    if hashtag_freq:
        wc = WordCloud(width=600, height=300, background_color="white").generate_from_frequencies(hashtag_freq)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No hashtag data available for wordcloud.")

with col2:
    st.subheader("Trending Accounts")
    if not df_acc.empty:
        cols = ["_handle", "followers_count", "statuses_count", "last_status_at", "_home_instance"]
        st.dataframe(df_acc[cols].head(25), height=400)
    else:
        st.warning("No trending accounts returned. Some instances may have disabled public trends.")

st.sidebar.info(f"Data is cached for one hour. Last refreshed at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
