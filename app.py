import streamlit as st
import json
import os
import re
from collections import Counter
from math import log

# ─── Page config ───
st.set_page_config(
    page_title="Wishlist → Purchase Discovery Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───
OPPORTUNITY_AREAS = {
    "size_fit": {
        "name": "Size & Fit Uncertainty",
        "impact": 9.2,
        "color": "#E74C3C",
        "description": "Users can't verify how a product will fit without trying it on. Size charts are inconsistent across brands, and past return experiences compound hesitation.",
    },
    "wishlist_ux": {
        "name": "Wishlist Clutter & Decision Fatigue",
        "impact": 8.1,
        "color": "#9B59B6",
        "description": "Wishlists grow to 50–200+ items with no filters, sorting, or organization. Users can't find what they actually want to buy from the noise.",
    },
    "price_timing": {
        "name": "Price Watching & Sale Waiting",
        "impact": 7.8,
        "color": "#F39C12",
        "description": "Users wishlist items at full price and wait for sales. Without native price tracking, they rely on third-party apps or simply forget.",
    },
    "return_anxiety": {
        "name": "Return Process Anxiety",
        "impact": 7.5,
        "color": "#E67E22",
        "description": "Past bad return experiences (failed quality checks, wrong reshipped items, refund delays) create forward hesitation.",
    },
    "social_validation": {
        "name": "Social & Styling Validation Gap",
        "impact": 7.2,
        "color": "#3498DB",
        "description": "Users want to see real-life looks on real bodies and get peer opinions before committing to a purchase.",
    },
    "occasion_mismatch": {
        "name": "Occasion-Driven Postponement",
        "impact": 6.8,
        "color": "#1ABC9C",
        "description": "Users wishlist items for future occasions (weddings, festivals, office) but have no trigger to revisit when the occasion nears.",
    },
    "comparison_gap": {
        "name": "No Comparison Infrastructure",
        "impact": 6.5,
        "color": "#2ECC71",
        "description": "Users wishlist similar items to compare but lack side-by-side comparison tools for reviews, sizing, and material.",
    },
    "trust_authenticity": {
        "name": "Product Authenticity & Quality Doubts",
        "impact": 6.3,
        "color": "#95A5A6",
        "description": "Concerns about fake/different products, inconsistent quality across sellers, and review credibility.",
    },
}

RESEARCH_QUESTIONS = [
    ("RQ1", "Why do users add fashion products to their wishlist?"),
    ("RQ2", "What prevents wishlisted products from eventually being purchased?"),
    ("RQ3", "What uncertainties remain after users have identified a product they like?"),
    ("RQ4", "What causes users to postpone a purchase?"),
    ("RQ5", "How do users compare multiple shortlisted products?"),
    ("RQ6", "What information do users seek outside Myntra before purchasing?"),
    ("RQ7", "What role do fit, size, styling, price, reviews, occasion and social validation play?"),
    ("RQ8", "When is wishlist genuine purchase intent vs bookmarking?"),
    ("RQ9", "How do behaviors differ across user segments?"),
    ("RQ10", "What unmet needs emerge consistently across user conversations?"),
]

RQ_TO_THEMES = {
    "RQ1": ["price_timing", "occasion_mismatch", "wishlist_ux"],
    "RQ2": ["size_fit", "return_anxiety", "trust_authenticity", "wishlist_ux"],
    "RQ3": ["size_fit", "social_validation", "trust_authenticity"],
    "RQ4": ["price_timing", "occasion_mismatch", "return_anxiety", "size_fit"],
    "RQ5": ["comparison_gap", "wishlist_ux", "social_validation"],
    "RQ6": ["social_validation", "trust_authenticity", "price_timing", "comparison_gap"],
    "RQ7": ["size_fit", "social_validation", "price_timing", "occasion_mismatch"],
    "RQ8": ["wishlist_ux", "occasion_mismatch", "price_timing"],
    "RQ9": ["occasion_mismatch", "price_timing", "size_fit"],
    "RQ10": ["size_fit", "wishlist_ux", "comparison_gap", "social_validation"],
}


# ─── Data loading ───
@st.cache_data
def load_reviews():
    path = os.path.join(os.path.dirname(__file__), "data", "reviews.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Simple TF-IDF search engine (no sklearn dependency) ───
class MiniSearch:
    """Lightweight keyword search with TF-IDF ranking."""

    def __init__(self, docs: list[dict], field: str = "text"):
        self.docs = docs
        self.field = field
        self.n = len(docs)
        # Tokenize
        self.tokens = []
        self.df = Counter()
        for doc in docs:
            toks = set(self._tokenize(doc[field]))
            self.tokens.append(toks)
            for t in toks:
                self.df[t] += 1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        q_toks = set(self._tokenize(query))
        scores = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for t in q_toks:
                if t in self.tokens[i] and self.df[t] > 0:
                    idf = log(self.n / self.df[t])
                    score += idf
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:top_k]]


# ─── Claude API helper ───
def call_claude(system_prompt: str, user_msg: str, api_key: str) -> str | None:
    """Call Anthropic Messages API. Returns the text response or None on error."""
    import urllib.request

    body = json.dumps(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_msg}],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return "".join(b.get("text", "") for b in data.get("content", []))
    except Exception as e:
        return f"__ERROR__: {e}"


# ─── Custom CSS ───
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; }
        [data-testid="stToolbar"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    .metric-card {
        background: #161920;
        border: 1px solid #252830;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #E8E9EC; }
    .metric-label { font-size: 12px; color: #8B8F9A; text-transform: uppercase; letter-spacing: 0.06em; }
    .opp-card {
        background: #161920;
        border: 1px solid #252830;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .review-card {
        background: #161920;
        border: 1px solid #252830;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 6px;
    }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
    }
        div[data-testid="stSidebar"] { background: #0D0F12; }
    /* Radio buttons as rectangular nav items */
    div[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
    div[data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent;
        border-radius: 8px;
        padding: 8px 12px !important;
        cursor: pointer;
        transition: background 0.15s;
    }
    div[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #1C2029;
    }
    div[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background: rgba(108,142,239,0.15);
        border: 1px solid rgba(74,107,196,0.4);
    }
    /* Hide the radio circles */
    div[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] { display: none; }
    div[data-testid="stSidebar"] [role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
        font-size: 14px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Sidebar ───
with st.sidebar:
    st.title("🔍 Discovery Engine")
    st.caption("Myntra Wishlist → Purchase")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "💬 Ask the Data",
            "📑 Evidence Explorer",
            "🗺️ RQ Mapping",
            "⚖️ Comparison Matrix",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Data Corpus**")
    reviews = load_reviews()
    st.metric("Total Reviews", len(reviews))
    st.metric("Platforms", len(set(r["platform"] for r in reviews)))
    st.metric("Themes Tracked", len(OPPORTUNITY_AREAS))
    st.divider()
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Required for 'Ask the Data'. Get one at console.anthropic.com",
    )
    st.caption("Key stays in your browser session and is never stored.")


# ═══════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════
if page == "📊 Dashboard":
    st.header("Discovery Dashboard")
    st.caption(
        "AI-powered analysis of user feedback across 8 platforms to understand why Myntra users wishlist but don't purchase."
    )

    # ── Top metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Evidence Points", f"{len(reviews)}")
    col2.metric("Opportunity Areas", len(OPPORTUNITY_AREAS))
    col3.metric("Platforms", len(set(r["platform"] for r in reviews)))
    col4.metric("Research Qs Covered", "10 / 10")

    st.subheader("Opportunity Areas — Ranked by Impact")

    sorted_opps = sorted(OPPORTUNITY_AREAS.items(), key=lambda x: x[1]["impact"], reverse=True)
    for key, opp in sorted_opps:
        count = sum(1 for r in reviews if key in r["themes"])
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.markdown(
                f"**{opp['name']}** &nbsp; `{opp['impact']}/10 impact` &nbsp; · &nbsp; {count} reviews"
            )
            st.progress(opp["impact"] / 10)
            st.caption(opp["description"])
        with col_b:
            # Top platforms for this theme
            plats = Counter(r["platform"] for r in reviews if key in r["themes"])
            for p, c in plats.most_common(3):
                st.caption(f"{p}: {c}")
        st.divider()

    # ── Source breakdown ──
    st.subheader("Reviews by Platform")
    plat_counts = Counter(r["platform"] for r in reviews)
    cols = st.columns(len(plat_counts))
    for i, (plat, cnt) in enumerate(plat_counts.most_common()):
        with cols[i]:
            st.metric(plat, cnt)

    # ── Sentiment ──
    st.subheader("Sentiment Distribution")
    sent_counts = Counter(r["sentiment"] for r in reviews)
    cols2 = st.columns(len(sent_counts))
    for i, (sent, cnt) in enumerate(sent_counts.most_common()):
        with cols2[i]:
            emoji = {"negative": "🔴", "mixed": "🟡", "neutral": "⚪", "positive": "🟢"}.get(sent, "⚪")
            st.metric(f"{emoji} {sent.title()}", cnt)

    # ── User segments ──
    st.subheader("User Segments Identified")
    seg_counts = Counter(r["user_segment"] for r in reviews)
    for seg, cnt in seg_counts.most_common(10):
        st.markdown(f"- **{seg}** — {cnt} reviews")


# ═══════════════════════════════════════
#  ASK THE DATA  (RAG search)
# ═══════════════════════════════════════
elif page == "💬 Ask the Data":
    st.header("Ask the Data")
    st.caption(
        "Ask any question about Myntra wishlist behavior. The engine retrieves relevant user reviews, "
        "then uses Claude to synthesize an evidence-grounded answer."
    )

    # Example questions
    with st.expander("💡 Example questions you can ask"):
        examples = [
            "Why do users add items to their wishlist but never buy them?",
            "What are the main reasons users hesitate about size and fit?",
            "How do users use the wishlist as a bookmarking tool vs purchase intent?",
            "What information do shoppers seek on YouTube before buying from Myntra?",
            "How does return policy anxiety affect purchase decisions?",
            "What do price-sensitive users do with their wishlists?",
            "Why do users maintain large wishlists with 100+ items?",
            "What role does social validation play in fashion purchase decisions?",
            "How do different user segments use the wishlist differently?",
            "What are the most common unmet needs across user feedback?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:30]}"):
                st.session_state["ask_query"] = ex

    query = st.text_input(
        "Your question",
        value=st.session_state.get("ask_query", ""),
        placeholder="e.g., Why do users postpone purchases from their wishlist?",
    )

    if st.button("🔍 Search & Analyze", type="primary", disabled=not query.strip()):
        search_engine = MiniSearch(reviews)
        results = search_engine.search(query, top_k=15)

        if not results:
            st.warning("No relevant reviews found. Try rephrasing your question.")
        else:
            # ── Show retrieved reviews ──
            st.subheader(f"📎 Retrieved {len(results)} Relevant Reviews")

            for r in results:
                themes_str = ", ".join(
                    OPPORTUNITY_AREAS[t]["name"] for t in r["themes"] if t in OPPORTUNITY_AREAS
                )
                st.markdown(
                    f"""<div class="review-card">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="font-size:12px;color:#6C8EEF;font-weight:600;">{r['platform']}</span>
                        <span style="font-size:11px;color:#5A5E6A;">{r.get('date','')}</span>
                    </div>
                    <div style="font-size:13px;color:#E8E9EC;line-height:1.6;">"{r['text']}"</div>
                    <div style="margin-top:8px;font-size:11px;color:#8B8F9A;">
                        Themes: {themes_str} &nbsp;·&nbsp; Sentiment: {r['sentiment']} &nbsp;·&nbsp; Segment: {r.get('user_segment','-')}
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # ── Claude synthesis ──
            st.subheader("🧠 AI-Synthesized Answer")

            if not api_key:
                st.info(
                    "Enter your Anthropic API key in the sidebar to get an AI-synthesized answer. "
                    "Without it, you can still browse the retrieved reviews above."
                )
            else:
                with st.spinner("Claude is analyzing the evidence..."):
                    review_texts = "\n---\n".join(
                        f"[{r['platform']}] [{r['sentiment']}] [{r.get('user_segment','')}] {r['text']}"
                        for r in results
                    )
                    system = (
                        "You are a senior user researcher analyzing Myntra (India's top fashion e-commerce) "
                        "wishlist-to-purchase conversion. You're given retrieved user reviews as evidence. "
                        "Answer the user's question grounded ONLY in the provided reviews. "
                        "Structure your answer as:\n"
                        "1. **Direct Answer** (2-3 sentences)\n"
                        "2. **Key Themes** (list the 3-5 most prominent patterns with evidence)\n"
                        "3. **User Segments Affected** (which types of users are most impacted)\n"
                        "4. **Opportunity for Intervention** (what could Myntra do, excluding monetary incentives)\n"
                        "5. **Evidence Strength** (how confident are you based on the review volume and consistency)\n\n"
                        "Quote specific reviews as evidence (use brief excerpts). Be specific, not generic."
                    )
                    user_msg = f"Question: {query}\n\nRetrieved user reviews:\n{review_texts}"
                    answer = call_claude(system, user_msg, api_key)

                if answer and not answer.startswith("__ERROR__"):
                    st.markdown(answer)
                elif answer:
                    st.error(answer.replace("__ERROR__: ", ""))

            # ── Theme distribution in results ──
            st.subheader("Theme Distribution in Retrieved Reviews")
            theme_counts = Counter()
            for r in results:
                for t in r["themes"]:
                    if t in OPPORTUNITY_AREAS:
                        theme_counts[t] += 1
            if theme_counts:
                for t, cnt in theme_counts.most_common():
                    opp = OPPORTUNITY_AREAS[t]
                    st.markdown(f"**{opp['name']}** — {cnt} / {len(results)} reviews")
                    st.progress(cnt / len(results))


# ═══════════════════════════════════════
#  EVIDENCE EXPLORER
# ═══════════════════════════════════════
elif page == "📑 Evidence Explorer":
    st.header("Evidence Explorer")
    st.caption("Browse and filter the review corpus by platform, theme, sentiment, and user segment.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plat_filter = st.multiselect(
            "Platform",
            sorted(set(r["platform"] for r in reviews)),
            default=[],
        )
    with col2:
        theme_filter = st.multiselect(
            "Theme",
            [(k, v["name"]) for k, v in OPPORTUNITY_AREAS.items()],
            format_func=lambda x: x[1],
            default=[],
        )
    with col3:
        sent_filter = st.multiselect(
            "Sentiment",
            sorted(set(r["sentiment"] for r in reviews)),
            default=[],
        )
    with col4:
        seg_filter = st.multiselect(
            "User Segment",
            sorted(set(r["user_segment"] for r in reviews)),
            default=[],
        )

    # Filter
    filtered = reviews
    if plat_filter:
        filtered = [r for r in filtered if r["platform"] in plat_filter]
    if theme_filter:
        theme_keys = [t[0] for t in theme_filter]
        filtered = [r for r in filtered if any(t in theme_keys for t in r["themes"])]
    if sent_filter:
        filtered = [r for r in filtered if r["sentiment"] in sent_filter]
    if seg_filter:
        filtered = [r for r in filtered if r.get("user_segment") in seg_filter]

    st.markdown(f"**Showing {len(filtered)} of {len(reviews)} reviews**")

    # Quick stats
    if filtered:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Theme Distribution**")
            tc = Counter()
            for r in filtered:
                for t in r["themes"]:
                    if t in OPPORTUNITY_AREAS:
                        tc[t] += 1
            for t, c in tc.most_common():
                st.caption(f"{OPPORTUNITY_AREAS[t]['name']}: {c}")
        with col_b:
            st.markdown("**Sentiment Split**")
            sc = Counter(r["sentiment"] for r in filtered)
            for s, c in sc.most_common():
                st.caption(f"{s}: {c}")

    st.divider()

    # Review list
    for r in filtered[:50]:
        themes_str = " · ".join(
            OPPORTUNITY_AREAS[t]["name"] for t in r["themes"] if t in OPPORTUNITY_AREAS
        )
        rating_str = f"⭐ {r['rating']}/5" if r.get("rating") else ""
        st.markdown(
            f"""<div class="review-card">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-size:12px;color:#6C8EEF;font-weight:600;">{r['platform']} {rating_str}</span>
                <span style="font-size:11px;color:#5A5E6A;">{r.get('date','')} · {r.get('user_segment','')}</span>
            </div>
            <div style="font-size:13px;color:#E8E9EC;line-height:1.6;">"{r['text']}"</div>
            <div style="margin-top:8px;font-size:11px;color:#8B8F9A;">Themes: {themes_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    if len(filtered) > 50:
        st.info(f"Showing first 50 of {len(filtered)} reviews. Use filters to narrow down.")


# ═══════════════════════════════════════
#  RQ MAPPING
# ═══════════════════════════════════════
elif page == "🗺️ RQ Mapping":
    st.header("Research Question ↔ Evidence Mapping")
    st.caption(
        "Each research question from the problem statement, mapped to opportunity areas "
        "and sample evidence from the review corpus."
    )

    for rq_id, rq_text in RESEARCH_QUESTIONS:
        with st.expander(f"**{rq_id}**: {rq_text}"):
            themes = RQ_TO_THEMES.get(rq_id, [])
            st.markdown("**Related Opportunity Areas:**")
            for t in themes:
                opp = OPPORTUNITY_AREAS[t]
                count = sum(1 for r in reviews if t in r["themes"])
                st.markdown(f"- {opp['name']} — Impact: `{opp['impact']}/10` · {count} reviews")

            # Sample reviews
            relevant = [
                r for r in reviews if any(t in r["themes"] for t in themes)
            ]
            st.markdown(f"**Sample Evidence** ({len(relevant)} total reviews)")
            for r in relevant[:5]:
                st.markdown(
                    f"> *\"{r['text']}\"*\n>\n> — {r['platform']}, {r.get('user_segment','')}"
                )


# ═══════════════════════════════════════
#  COMPARISON MATRIX
# ═══════════════════════════════════════
elif page == "⚖️ Comparison Matrix":
    st.header("Opportunity Comparison Matrix")
    st.caption(
        "Side-by-side comparison of all opportunity areas by impact, evidence volume, "
        "platform coverage, and research question relevance."
    )

    # Build comparison data
    rows = []
    for key, opp in OPPORTUNITY_AREAS.items():
        count = sum(1 for r in reviews if key in r["themes"])
        platforms = set(r["platform"] for r in reviews if key in r["themes"])
        segments = set(r.get("user_segment", "") for r in reviews if key in r["themes"])
        rqs = [rq_id for rq_id, themes in RQ_TO_THEMES.items() if key in themes]
        sents = Counter(r["sentiment"] for r in reviews if key in r["themes"])
        neg_pct = round(sents.get("negative", 0) / max(count, 1) * 100)
        rows.append(
            {
                "Opportunity Area": opp["name"],
                "Impact (0-10)": opp["impact"],
                "Evidence Count": count,
                "Platforms": len(platforms),
                "RQs Addressed": len(rqs),
                "Segments Affected": len(segments),
                "% Negative Sentiment": neg_pct,
            }
        )

    rows.sort(key=lambda x: x["Impact (0-10)"], reverse=True)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Visual comparison
    st.subheader("Impact vs Evidence Volume")
    import_ok = True
    try:
        import altair  # noqa: F401 – just checking availability
    except ImportError:
        import_ok = False

    chart_rows = []
    for key, opp in OPPORTUNITY_AREAS.items():
        count = sum(1 for r in reviews if key in r["themes"])
        chart_rows.append(
            {"name": opp["name"], "impact": opp["impact"], "evidence": count}
        )

    if import_ok:
        import altair as alt
        import pandas as pd

        df = pd.DataFrame(chart_rows)
        chart = (
            alt.Chart(df)
            .mark_circle(size=200)
            .encode(
                x=alt.X("evidence:Q", title="Evidence Count (# reviews)"),
                y=alt.Y("impact:Q", title="Impact Score (0-10)", scale=alt.Scale(domain=[5, 10])),
                tooltip=["name", "impact", "evidence"],
                color=alt.Color("name:N", legend=None),
            )
            .properties(height=400)
            .interactive()
        )
        text = (
            alt.Chart(df)
            .mark_text(dy=-15, fontSize=11)
            .encode(
                x="evidence:Q",
                y="impact:Q",
                text="name:N",
            )
        )
        st.altair_chart(chart + text, use_container_width=True)
    else:
        st.markdown("| Area | Impact | Evidence |")
        st.markdown("|---|---|---|")
        for r in chart_rows:
            st.markdown(f"| {r['name']} | {r['impact']} | {r['evidence']} |")

    # ── Synthesis ──
    st.subheader("Synthesis for Part 2")
    st.success(
        "**Top 3 non-monetary intervention opportunities:**\n\n"
        "1. **Size & Fit Uncertainty** (Impact 9.2) — The single largest barrier. Users report inconsistent "
        "size charts across brands, wrong sizes delivered, and a lack of body-type representation. "
        "This problem is *compounded* by return anxiety (if it doesn't fit, dealing with returns is painful).\n\n"
        "2. **Wishlist Clutter & Decision Fatigue** (Impact 8.1) — Wishlists become dumping grounds with "
        "no organization tools. Users can't filter by size availability, price, or category. The wishlist "
        "designed to aid purchase decisions instead creates decision paralysis.\n\n"
        "3. **Price Watching & Sale Waiting** (Impact 7.8) — A large segment explicitly uses wishlists as "
        "price-watch lists, waiting for EORS / Big Fashion Festival. While the constraint says no monetary "
        "incentives, solving the *information gap* (showing price history, notifying at right time) is still valid.\n\n"
        "**The discovered user problem**: Users have identified products they want, but the platform fails to "
        "resolve the remaining uncertainties (Will it fit me? Is it worth this price? What does it really "
        "look like? How do I choose from 5 similar options?) — and the wishlist itself compounds the problem "
        "by becoming an unmanageable, unstructured backlog that erodes purchase intent over time."
    )
