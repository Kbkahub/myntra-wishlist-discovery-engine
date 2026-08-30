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
        "color": "#FF3F6C",
        "description": "Users can't verify how a product will fit without trying it on. Size charts are inconsistent across brands, and past return experiences compound hesitation.",
    },
    "wishlist_ux": {
        "name": "Wishlist Clutter & Decision Fatigue",
        "impact": 8.1,
        "color": "#A0369B",
        "description": "Wishlists grow to 50–200+ items with no filters, sorting, or organization. Users can't find what they actually want to buy from the noise.",
    },
    "price_timing": {
        "name": "Price Watching & Sale Waiting",
        "impact": 7.8,
        "color": "#F26A38",
        "description": "Users wishlist items at full price and wait for sales. Without native price tracking, they rely on third-party apps or simply forget.",
    },
    "return_anxiety": {
        "name": "Return Process Anxiety",
        "impact": 7.5,
        "color": "#D4532D",
        "description": "Past bad return experiences (failed quality checks, wrong reshipped items, refund delays) create forward hesitation.",
    },
    "social_validation": {
        "name": "Social & Styling Validation Gap",
        "impact": 7.2,
        "color": "#3B82F6",
        "description": "Users want to see real-life looks on real bodies and get peer opinions before committing to a purchase.",
    },
    "occasion_mismatch": {
        "name": "Occasion-Driven Postponement",
        "impact": 6.8,
        "color": "#14B8A6",
        "description": "Users wishlist items for future occasions (weddings, festivals, office) but have no trigger to revisit when the occasion nears.",
    },
    "comparison_gap": {
        "name": "No Comparison Infrastructure",
        "impact": 6.5,
        "color": "#22C55E",
        "description": "Users wishlist similar items to compare but lack side-by-side comparison tools for reviews, sizing, and material.",
    },
    "trust_authenticity": {
        "name": "Product Authenticity & Quality Doubts",
        "impact": 6.3,
        "color": "#8B8F9A",
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


@st.cache_data
def load_survey():
    path = os.path.join(os.path.dirname(__file__), "data", "survey.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Simple TF-IDF search engine ───
class MiniSearch:
    """Lightweight keyword search with TF-IDF ranking."""

    def __init__(self, docs: list, field: str = "text"):
        self.docs = docs
        self.field = field
        self.n = len(docs)
        self.tokens = []
        self.df = Counter()
        for doc in docs:
            toks = set(self._tokenize(doc.get(field, "")))
            self.tokens.append(toks)
            for t in toks:
                self.df[t] += 1

    @staticmethod
    def _tokenize(text: str) -> list:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 15) -> list:
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


# ─── Groq API helper ───
def call_groq(system_prompt: str, user_msg: str, api_key: str) -> str:
    """Call Groq API (OpenAI-compatible). Returns text response or error string."""
    import urllib.request

    body = json.dumps({
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 1500,
        "temperature": 0.4,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERROR__: {e}"


# ─── Custom CSS (Myntra themed) ───
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Hide Streamlit chrome */
    [data-testid="stToolbar"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }

    .block-container { padding-top: 1.5rem; font-family: 'Inter', sans-serif; }

    /* Sidebar */
    div[data-testid="stSidebar"] { background: #282C3F; }
    div[data-testid="stSidebar"] * { color: #FFFFFF !important; }

    /* Sidebar nav: hide radio circles, rectangular highlight */
    div[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin: 2px 0 !important;
        border: 1px solid transparent !important;
        transition: background 0.15s;
    }
    div[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(255,255,255,0.08) !important;
    }
    div[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[aria-checked="true"] {
        background: rgba(255,63,108,0.2) !important;
        border: 1px solid rgba(255,63,108,0.5) !important;
    }
    div[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(40,44,63,0.06);
    }
    [data-testid="stMetricValue"] { color: #282C3F !important; }
    [data-testid="stMetricLabel"] { color: #535766 !important; }

    /* Review cards */
    .review-card {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(40,44,63,0.06);
    }
    .survey-card {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-left: 4px solid #FF3F6C;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(40,44,63,0.06);
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: #FF3F6C !important;
        border-color: #FF3F6C !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #E0355E !important;
        border-color: #E0355E !important;
    }

    /* Progress bars */
    .stProgress > div > div > div { background: #FF3F6C !important; }

    /* Expander headers */
    .streamlit-expanderHeader { font-weight: 600; color: #282C3F; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Sidebar ───
with st.sidebar:
    st.markdown(
        "<h2 style='margin:0;font-size:20px;'>🔍 Discovery Engine</h2>"
        "<p style='margin:2px 0 0;font-size:12px;opacity:0.7;'>Myntra Wishlist → Purchase</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "💬 Ask Insights from Public Reviews",
            "🎯 Ask Insights from User Survey",
            "📑 Evidence Explorer",
            "🗺️ RQ Mapping",
            "⚖️ Comparison Matrix",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Data Corpus**")
    reviews = load_reviews()
    survey = load_survey()
    st.metric("Public Reviews", len(reviews))
    st.metric("Survey Responses", len(survey))
    st.metric("Platforms", len(set(r["platform"] for r in reviews)))
    st.divider()
    # API key
    default_key = ""
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        default_key = st.secrets["GROQ_API_KEY"]
    api_key = st.text_input(
        "Groq API Key (free)",
        value=default_key,
        type="password",
        help="Free at console.groq.com. Powers AI synthesis.",
    )
    if default_key:
        st.caption("✅ Key loaded from secrets.")
    else:
        st.caption("Get a free key at console.groq.com")


# ═══════════════════════════════════════
#  System prompts
# ═══════════════════════════════════════
REVIEW_SYSTEM = (
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

SURVEY_SYSTEM = (
    "You are a senior user researcher analyzing results from a primary user survey about "
    "Myntra wishlist behavior. The survey was conducted with real users and contains both "
    "quantitative (Likert scale 1-5) and qualitative (open text) responses. "
    "Answer the user's question grounded ONLY in the survey data provided. "
    "Structure your answer as:\n"
    "1. **Key Finding** (2-3 sentences summarizing the insight)\n"
    "2. **Supporting Data** (reference specific survey responses, quote verbatim where useful)\n"
    "3. **Demographic Patterns** (any differences by gender, age, city, or employment status)\n"
    "4. **Quantitative Highlights** (reference barrier ratings where relevant — scale is 1=not a problem to 5=major problem)\n"
    "5. **Implications** (what this means for product decisions, no monetary incentives allowed)\n\n"
    "Be specific. Reference respondent demographics when quoting."
)


# ═══════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("<h1 style='color:#282C3F;margin-bottom:4px;'>Discovery Dashboard</h1>", unsafe_allow_html=True)
    st.caption(
        "AI-powered analysis of user feedback across 8 platforms + primary user survey "
        "to understand why Myntra users wishlist but don't purchase."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Public Reviews", f"{len(reviews)}")
    col2.metric("Survey Responses", f"{len(survey)}")
    col3.metric("Opportunity Areas", len(OPPORTUNITY_AREAS))
    col4.metric("Research Qs Covered", "10 / 10")

    st.subheader("Opportunity Areas — Ranked by Impact")

    sorted_opps = sorted(OPPORTUNITY_AREAS.items(), key=lambda x: x[1]["impact"], reverse=True)
    for key, opp in sorted_opps:
        count = sum(1 for r in reviews if key in r["themes"])
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.markdown(f"**{opp['name']}** &nbsp; `{opp['impact']}/10 impact` &nbsp; · &nbsp; {count} reviews")
            st.progress(opp["impact"] / 10)
            st.caption(opp["description"])
        with col_b:
            plats = Counter(r["platform"] for r in reviews if key in r["themes"])
            for p, c in plats.most_common(3):
                st.caption(f"{p}: {c}")
        st.divider()

    # Source breakdown
    st.subheader("Reviews by Platform")
    plat_counts = Counter(r["platform"] for r in reviews)
    cols = st.columns(len(plat_counts))
    for i, (plat, cnt) in enumerate(plat_counts.most_common()):
        with cols[i]:
            st.metric(plat, cnt)

    # Sentiment
    st.subheader("Sentiment Distribution")
    sent_counts = Counter(r["sentiment"] for r in reviews)
    cols2 = st.columns(len(sent_counts))
    emojis = {"negative": "🔴", "mixed": "🟡", "neutral": "⚪", "positive": "🟢"}
    for i, (sent, cnt) in enumerate(sent_counts.most_common()):
        with cols2[i]:
            st.metric(f"{emojis.get(sent, '⚪')} {sent.title()}", cnt)

    # User segments
    st.subheader("User Segments Identified")
    seg_counts = Counter(r["user_segment"] for r in reviews)
    for seg, cnt in seg_counts.most_common(10):
        st.markdown(f"- **{seg}** — {cnt} reviews")


# ═══════════════════════════════════════
#  ASK FROM PUBLIC REVIEWS
# ═══════════════════════════════════════
elif page == "💬 Ask Insights from Public Reviews":
    st.markdown("<h1 style='color:#282C3F;'>Ask Insights from Public Reviews</h1>", unsafe_allow_html=True)
    st.caption(
        "Ask any question about wishlist behavior. The engine retrieves relevant reviews "
        "from App Store, Play Store, Reddit, YouTube, Trustpilot & more, then uses AI to synthesize an answer."
    )

    with st.expander("💡 Example questions"):
        examples = [
            "Why do users add items to their wishlist but never buy them?",
            "What are the main reasons users hesitate about size and fit?",
            "How do users use the wishlist as a bookmarking tool?",
            "What information do shoppers seek on YouTube before buying?",
            "How does return policy anxiety affect purchase decisions?",
            "What do price-sensitive users do with their wishlists?",
            "What role does social validation play in fashion purchases?",
            "What are the most common unmet needs?",
        ]
        for ex in examples:
            if st.button(ex, key=f"rev_{ex[:25]}"):
                st.session_state["rev_query"] = ex

    query = st.text_input(
        "Your question",
        value=st.session_state.get("rev_query", ""),
        placeholder="e.g., Why do users postpone purchases from their wishlist?",
        key="rev_input",
    )

    if st.button("🔍 Search & Analyze", type="primary", disabled=not query.strip(), key="rev_btn"):
        engine = MiniSearch(reviews, field="text")
        results = engine.search(query, top_k=15)

        if not results:
            st.warning("No relevant reviews found. Try rephrasing your question.")
        else:
            st.subheader(f"📎 Retrieved {len(results)} Relevant Reviews")
            for r in results:
                themes_str = ", ".join(
                    OPPORTUNITY_AREAS[t]["name"] for t in r["themes"] if t in OPPORTUNITY_AREAS
                )
                rating_str = f" ⭐ {r['rating']}/5" if r.get("rating") else ""
                st.markdown(
                    f"""<div class="review-card">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="font-size:12px;color:#FF3F6C;font-weight:600;">{r['platform']}{rating_str}</span>
                        <span style="font-size:11px;color:#94969F;">{r.get('date','')}</span>
                    </div>
                    <div style="font-size:13px;color:#282C3F;line-height:1.6;">"{r['text']}"</div>
                    <div style="margin-top:8px;font-size:11px;color:#535766;">
                        Themes: {themes_str} · Sentiment: {r['sentiment']} · Segment: {r.get('user_segment','-')}
                    </div></div>""",
                    unsafe_allow_html=True,
                )

            # AI synthesis
            st.subheader("🧠 AI-Synthesized Answer")
            if not api_key:
                st.info("Enter your free Groq API key in the sidebar to get AI-synthesized answers.")
            else:
                with st.spinner("Analyzing evidence..."):
                    review_texts = "\n---\n".join(
                        f"[{r['platform']}] [{r['sentiment']}] [{r.get('user_segment','')}] {r['text']}"
                        for r in results
                    )
                    answer = call_groq(REVIEW_SYSTEM, f"Question: {query}\n\nRetrieved user reviews:\n{review_texts}", api_key)
                if answer and not answer.startswith("__ERROR__"):
                    st.markdown(answer)
                elif answer:
                    st.error(answer.replace("__ERROR__: ", ""))

            # Theme distribution
            st.subheader("Theme Distribution in Results")
            theme_counts = Counter()
            for r in results:
                for t in r["themes"]:
                    if t in OPPORTUNITY_AREAS:
                        theme_counts[t] += 1
            for t, cnt in theme_counts.most_common():
                opp = OPPORTUNITY_AREAS[t]
                st.markdown(f"**{opp['name']}** — {cnt} / {len(results)} reviews")
                st.progress(cnt / len(results))


# ═══════════════════════════════════════
#  ASK FROM USER SURVEY
# ═══════════════════════════════════════
elif page == "🎯 Ask Insights from User Survey":
    st.markdown("<h1 style='color:#282C3F;'>Ask Insights from User Survey</h1>", unsafe_allow_html=True)
    st.caption(
        f"Ask questions about primary research data from {len(survey)} survey respondents. "
        "The engine retrieves matching responses and uses AI to synthesize insights."
    )

    # Survey overview
    with st.expander("📋 Survey Overview"):
        genders = Counter(s["demographics"]["gender"] for s in survey if s["demographics"]["gender"])
        ages = Counter(s["demographics"]["age_group"] for s in survey if s["demographics"]["age_group"])
        situations = Counter(s["demographics"]["situation"] for s in survey if s["demographics"]["situation"])
        cities = Counter(s["demographics"]["city"] for s in survey if s["demographics"]["city"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Gender**")
            for g, c in genders.most_common():
                st.caption(f"{g}: {c}")
        with c2:
            st.markdown("**Age Group**")
            for a, c in ages.most_common():
                st.caption(f"{a}: {c}")
        with c3:
            st.markdown("**Situation**")
            for s_val, c in situations.most_common():
                st.caption(f"{s_val}: {c}")
        with c4:
            st.markdown("**City**")
            for ci, c in cities.most_common():
                st.caption(f"{ci}: {c}")

        # Average barrier ratings
        st.markdown("---")
        st.markdown("**Average Barrier Ratings (1 = not a problem, 5 = major problem)**")
        barrier_labels = {
            "unsure_size_fit": "Unsure about size/fit",
            "too_many_items_overwhelmed": "Too many items, overwhelmed",
            "dont_know_styling": "Don't know how to style it",
            "not_enough_reviews": "Not enough reviews/photos",
            "forgotten_saved_items": "Forgotten saved items",
            "impulse_save_excitement_faded": "Impulse save, excitement faded",
            "cant_decide_similar_items": "Can't decide between similar items",
            "not_confident_product_like_photos": "Product won't look like photos",
            "out_of_stock": "Out of stock / size unavailable",
            "saving_for_future_event": "Saving for a future event",
            "dont_trust_returns": "Don't trust return process",
            "realized_dont_need": "Realized don't need it",
        }
        avgs = {}
        for key, label in barrier_labels.items():
            vals = [s["barriers_ratings"][key] for s in survey if s["barriers_ratings"].get(key) is not None]
            if vals:
                avg = sum(vals) / len(vals)
                avgs[label] = avg
        for label, avg in sorted(avgs.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{label}** — {avg:.1f}/5")
            st.progress(avg / 5)

    with st.expander("💡 Example questions"):
        survey_examples = [
            "What does the wishlist mean to users?",
            "What are the top barriers to buying from the wishlist?",
            "How do users describe times they wanted to buy but didn't?",
            "What magic feature do users wish the app had?",
            "Do users use wishlist as purchase intent or bookmarking?",
            "How do different age groups use the wishlist?",
            "What triggers users to revisit their wishlist?",
            "What made users finally complete a wishlist purchase?",
        ]
        for ex in survey_examples:
            if st.button(ex, key=f"srv_{ex[:25]}"):
                st.session_state["srv_query"] = ex

    query = st.text_input(
        "Your question",
        value=st.session_state.get("srv_query", ""),
        placeholder="e.g., What stops users from buying items they've wishlisted?",
        key="srv_input",
    )

    if st.button("🔍 Search & Analyze", type="primary", disabled=not query.strip(), key="srv_btn"):
        engine = MiniSearch(survey, field="searchable_text")
        results = engine.search(query, top_k=10)

        if not results:
            st.warning("No matching survey responses found. Try broader terms.")
        else:
            st.subheader(f"📎 Retrieved {len(results)} Matching Responses")
            for r in results:
                demo = r["demographics"]
                demo_str = f"{demo.get('gender', '')} · {demo.get('age_group', '')} · {demo.get('situation', '')} · {demo.get('city', '')}"

                # Pull key text answers
                highlights = []
                for field, label in [
                    ("wishlist_meaning", "Wishlist means"),
                    ("good_purchase_story", "Good purchase"),
                    ("didnt_buy_story", "Didn't buy because"),
                    ("magic_feature", "Magic feature"),
                    ("wishlist_change_request", "Would change"),
                    ("top_3_blockers", "Top blockers"),
                    ("scenario_40_items", "40-item scenario"),
                    ("scenario_unsure_top", "Unsure about item"),
                ]:
                    val = r.get(field)
                    if val and str(val).strip() not in ("None", "", "nan"):
                        highlights.append(f"<b>{label}:</b> {val}")

                highlight_html = "<br>".join(highlights) if highlights else "<i>Mostly quantitative responses</i>"

                st.markdown(
                    f"""<div class="survey-card">
                    <div style="font-size:12px;color:#FF3F6C;font-weight:600;margin-bottom:6px;">{demo_str}</div>
                    <div style="font-size:13px;color:#282C3F;line-height:1.7;">{highlight_html}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # AI synthesis
            st.subheader("🧠 AI-Synthesized Insight")
            if not api_key:
                st.info("Enter your free Groq API key in the sidebar to get AI-synthesized insights.")
            else:
                with st.spinner("Analyzing survey data..."):
                    survey_texts = "\n---\n".join(
                        f"[Respondent {r['id']}] [{r['demographics'].get('gender','')}] "
                        f"[{r['demographics'].get('age_group','')}] [{r['demographics'].get('city','')}]\n"
                        f"Wishlist meaning: {r.get('wishlist_meaning','N/A')}\n"
                        f"Top blockers: {r.get('top_3_blockers','N/A')}\n"
                        f"Decision factors: {r.get('top_3_decision_factors','N/A')}\n"
                        f"Good purchase story: {r.get('good_purchase_story','N/A')}\n"
                        f"Didn't buy story: {r.get('didnt_buy_story','N/A')}\n"
                        f"Magic feature: {r.get('magic_feature','N/A')}\n"
                        f"Wishlist items: {r.get('wishlist_usage',{}).get('items_in_wishlist','N/A')}\n"
                        f"Buy likelihood: {r.get('wishlist_usage',{}).get('perceived_buy_likelihood','N/A')}\n"
                        f"Thinking when adding: {r.get('wishlist_usage',{}).get('thinking_when_adding','N/A')}\n"
                        f"Barrier ratings: {json.dumps(r.get('barriers_ratings',{}))}"
                        for r in results
                    )
                    answer = call_groq(SURVEY_SYSTEM, f"Question: {query}\n\nSurvey responses:\n{survey_texts}", api_key)
                if answer and not answer.startswith("__ERROR__"):
                    st.markdown(answer)
                elif answer:
                    st.error(answer.replace("__ERROR__: ", ""))


# ═══════════════════════════════════════
#  EVIDENCE EXPLORER
# ═══════════════════════════════════════
elif page == "📑 Evidence Explorer":
    st.markdown("<h1 style='color:#282C3F;'>Evidence Explorer</h1>", unsafe_allow_html=True)
    st.caption("Browse and filter the review corpus by platform, theme, sentiment, and user segment.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plat_filter = st.multiselect("Platform", sorted(set(r["platform"] for r in reviews)), default=[])
    with col2:
        theme_filter = st.multiselect(
            "Theme",
            [(k, v["name"]) for k, v in OPPORTUNITY_AREAS.items()],
            format_func=lambda x: x[1],
            default=[],
        )
    with col3:
        sent_filter = st.multiselect("Sentiment", sorted(set(r["sentiment"] for r in reviews)), default=[])
    with col4:
        seg_filter = st.multiselect("User Segment", sorted(set(r["user_segment"] for r in reviews)), default=[])

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

    for r in filtered[:50]:
        themes_str = " · ".join(
            OPPORTUNITY_AREAS[t]["name"] for t in r["themes"] if t in OPPORTUNITY_AREAS
        )
        rating_str = f" ⭐ {r['rating']}/5" if r.get("rating") else ""
        st.markdown(
            f"""<div class="review-card">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-size:12px;color:#FF3F6C;font-weight:600;">{r['platform']}{rating_str}</span>
                <span style="font-size:11px;color:#94969F;">{r.get('date','')} · {r.get('user_segment','')}</span>
            </div>
            <div style="font-size:13px;color:#282C3F;line-height:1.6;">"{r['text']}"</div>
            <div style="margin-top:8px;font-size:11px;color:#535766;">Themes: {themes_str}</div></div>""",
            unsafe_allow_html=True,
        )
    if len(filtered) > 50:
        st.info(f"Showing first 50 of {len(filtered)} reviews. Use filters to narrow down.")


# ═══════════════════════════════════════
#  RQ MAPPING
# ═══════════════════════════════════════
elif page == "🗺️ RQ Mapping":
    st.markdown("<h1 style='color:#282C3F;'>Research Question ↔ Evidence Mapping</h1>", unsafe_allow_html=True)
    st.caption("Each research question mapped to opportunity areas with sample evidence.")

    for rq_id, rq_text in RESEARCH_QUESTIONS:
        with st.expander(f"**{rq_id}**: {rq_text}"):
            themes = RQ_TO_THEMES.get(rq_id, [])
            st.markdown("**Related Opportunity Areas:**")
            for t in themes:
                opp = OPPORTUNITY_AREAS[t]
                count = sum(1 for r in reviews if t in r["themes"])
                st.markdown(f"- {opp['name']} — Impact: `{opp['impact']}/10` · {count} reviews")

            relevant = [r for r in reviews if any(t in r["themes"] for t in themes)]
            st.markdown(f"**Sample Evidence** ({len(relevant)} total reviews)")
            for r in relevant[:5]:
                st.markdown(f"> *\"{r['text']}\"*\n>\n> — {r['platform']}, {r.get('user_segment','')}")


# ═══════════════════════════════════════
#  COMPARISON MATRIX
# ═══════════════════════════════════════
elif page == "⚖️ Comparison Matrix":
    st.markdown("<h1 style='color:#282C3F;'>Opportunity Comparison Matrix</h1>", unsafe_allow_html=True)
    st.caption("Side-by-side comparison by impact, evidence, platform coverage, and RQ relevance.")

    rows = []
    for key, opp in OPPORTUNITY_AREAS.items():
        count = sum(1 for r in reviews if key in r["themes"])
        platforms = set(r["platform"] for r in reviews if key in r["themes"])
        rqs = [rq_id for rq_id, themes in RQ_TO_THEMES.items() if key in themes]
        sents = Counter(r["sentiment"] for r in reviews if key in r["themes"])
        neg_pct = round(sents.get("negative", 0) / max(count, 1) * 100)
        rows.append({
            "Opportunity Area": opp["name"],
            "Impact (0-10)": opp["impact"],
            "Evidence Count": count,
            "Platforms": len(platforms),
            "RQs Addressed": len(rqs),
            "% Negative": neg_pct,
        })
    rows.sort(key=lambda x: x["Impact (0-10)"], reverse=True)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Chart
    st.subheader("Impact vs Evidence Volume")
    try:
        import altair as alt
        import pandas as pd

        chart_rows = []
        for key, opp in OPPORTUNITY_AREAS.items():
            count = sum(1 for r in reviews if key in r["themes"])
            chart_rows.append({"name": opp["name"], "impact": opp["impact"], "evidence": count})
        df = pd.DataFrame(chart_rows)
        chart = (
            alt.Chart(df)
            .mark_circle(size=200, color="#FF3F6C")
            .encode(
                x=alt.X("evidence:Q", title="Evidence Count"),
                y=alt.Y("impact:Q", title="Impact Score", scale=alt.Scale(domain=[5, 10])),
                tooltip=["name", "impact", "evidence"],
            )
            .properties(height=400)
            .interactive()
        )
        text = alt.Chart(df).mark_text(dy=-15, fontSize=11).encode(x="evidence:Q", y="impact:Q", text="name:N")
        st.altair_chart(chart + text, use_container_width=True)
    except ImportError:
        st.info("Install altair for the interactive chart.")

    # Synthesis
    st.subheader("Synthesis for Part 2")
    st.success(
        "**Top 3 non-monetary intervention opportunities:**\n\n"
        "1. **Size & Fit Uncertainty** (Impact 9.2) — The single largest barrier across both public reviews "
        "AND survey data. Users report inconsistent size charts, wrong sizes delivered, and no body-type "
        "representation. Survey respondents rated this barrier highest on average.\n\n"
        "2. **Wishlist Clutter & Decision Fatigue** (Impact 8.1) — Wishlists become dumping grounds. "
        "No filters, no categories, no comparison tools. Survey confirms users feel overwhelmed.\n\n"
        "3. **Price Watching & Sale Waiting** (Impact 7.8) — Users explicitly use wishlists as price-watch "
        "lists. Solving the information gap (price history, smart notifications) is valid without discounts.\n\n"
        "**The discovered user problem**: Users have identified products they want, but the platform fails to "
        "resolve remaining uncertainties (fit, quality, styling, comparison) — and the wishlist compounds "
        "the problem by becoming an unmanageable backlog that erodes purchase intent over time."
    )
