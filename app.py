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
    def __init__(self, docs, field="text"):
        self.docs = docs
        self.n = len(docs)
        self.tokens = []
        self.df = Counter()
        for doc in docs:
            toks = set(re.findall(r"[a-z0-9]+", doc.get(field, "").lower()))
            self.tokens.append(toks)
            for t in toks:
                self.df[t] += 1

    def search(self, query, top_k=15):
        q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        scores = []
        for i, doc in enumerate(self.docs):
            score = sum(log(self.n / self.df[t]) for t in q_toks if t in self.tokens[i] and self.df[t] > 0)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:top_k]]


# ─── Groq API with fallback models ───
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]


def call_groq(system_prompt, user_msg, api_key):
    import urllib.request
    import urllib.error

    last_error = ""
    for model in GROQ_MODELS:
        body = json.dumps({
            "model": model,
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
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_error = f"Model {model}: HTTP {e.code}"
            continue
        except Exception as e:
            last_error = f"Model {model}: {e}"
            continue

    return f"__ERROR__: All models failed. Last error: {last_error}. Check your Groq API key at console.groq.com."


# ─── Custom CSS (Myntra themed) ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    [data-testid="stToolbar"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    .block-container { padding-top: 1.5rem; font-family: 'Inter', sans-serif; }

    /* Force sidebar open */
    [data-testid="stSidebar"] { min-width: 300px !important; width: 300px !important; }
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 300px !important; width: 300px !important;
        transform: none !important; display: block !important;
    }
    /* Make sidebar toggle arrow visible */
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        color: #FF3F6C !important;
    }

    /* Sidebar — Myntra pink */
    div[data-testid="stSidebar"] { background: #FF3F6C !important; }
    div[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    div[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

    /* Sidebar nav buttons — bold, uppercase, no circles */
    div[data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #FFFFFF !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        text-align: left !important;
        padding: 10px 14px !important;
        margin: 2px 0 !important;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.15) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    div[data-testid="stSidebar"] .stButton > button:focus {
        background: rgba(255,255,255,0.25) !important;
        border: 1px solid rgba(255,255,255,0.5) !important;
        box-shadow: none !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E8E8E8;
        border-radius: 10px; padding: 12px;
        box-shadow: 0 1px 3px rgba(40,44,63,0.06);
    }
    [data-testid="stMetricValue"] { color: #282C3F !important; }
    [data-testid="stMetricLabel"] { color: #535766 !important; }

    .review-card {
        background: #FFFFFF; border: 1px solid #E8E8E8;
        border-radius: 10px; padding: 16px; margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(40,44,63,0.06);
    }
    .survey-card {
        background: #FFFFFF; border: 1px solid #E8E8E8;
        border-left: 4px solid #FF3F6C;
        border-radius: 10px; padding: 16px; margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(40,44,63,0.06);
    }
    .ai-answer {
        background: #FFF5F7; border: 1px solid #FFD6DF;
        border-left: 4px solid #FF3F6C;
        border-radius: 10px; padding: 20px; margin-bottom: 20px;
    }

    .stButton > button[kind="primary"] {
        background: #FF3F6C !important; border-color: #FF3F6C !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #E0355E !important; border-color: #E0355E !important;
    }
    .stProgress > div > div > div { background: #FF3F6C !important; }
    .streamlit-expanderHeader { font-weight: 600; color: #282C3F; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ───
NAV_ITEMS = [
    ("dashboard", "📊 DASHBOARD"),
    ("public_reviews", "💬 ASK INSIGHTS FROM PUBLIC REVIEWS"),
    ("user_survey", "🎯 ASK INSIGHTS FROM USER SURVEY"),
    ("explorer", "📑 EVIDENCE EXPLORER"),
    ("rq_mapping", "🗺️ RQ MAPPING"),
    ("comparison", "⚖️ COMPARISON MATRIX"),
]

if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"

with st.sidebar:
    st.markdown(
        "<h2 style='margin:0;font-size:20px;'>🔍 Discovery Engine</h2>"
        "<p style='margin:2px 0 0;font-size:12px;opacity:0.7;'>Myntra Wishlist → Purchase</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Navigation buttons — active page highlighted in pink
    for key, label in NAV_ITEMS:
        is_active = st.session_state["page"] == key
        if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state["page"] = key
            st.rerun()
    st.divider()
    st.markdown("**Data Corpus**")
    reviews = load_reviews()
    survey = load_survey()
    st.metric("Public Reviews", len(reviews))
    st.metric("Survey Responses", len(survey))
    st.metric("Platforms", len(set(r["platform"] for r in reviews)))

# Silent API key from secrets only
api_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""


# ─── System prompts ───
REVIEW_SYSTEM = (
    "You are a senior user researcher analyzing Myntra (India's top fashion e-commerce) "
    "wishlist-to-purchase conversion. You're given retrieved user reviews as evidence. "
    "Answer the user's question grounded ONLY in the provided reviews. "
    "Structure your answer as:\n"
    "1. **Direct Answer** (2-3 sentences)\n"
    "2. **Key Themes** (3-5 prominent patterns with evidence)\n"
    "3. **User Segments Affected**\n"
    "4. **Opportunity for Intervention** (no monetary incentives)\n"
    "5. **Evidence Strength**\n\n"
    "Quote specific reviews. Be specific, not generic."
)

SURVEY_SYSTEM = (
    "You are a senior user researcher analyzing a primary user survey about "
    "Myntra wishlist behavior. Survey has quantitative (Likert 1-5) and qualitative responses. "
    "Answer grounded ONLY in the provided survey data. Structure:\n"
    "1. **Key Finding** (2-3 sentences)\n"
    "2. **Supporting Data** (reference specific responses, quote verbatim)\n"
    "3. **Demographic Patterns** (differences by gender, age, city)\n"
    "4. **Quantitative Highlights** (barrier ratings: 1=no problem, 5=major problem)\n"
    "5. **Implications** (product decisions, no monetary incentives)\n\n"
    "Reference respondent demographics when quoting. Be specific."
)


# ═══════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════
if st.session_state["page"] == "dashboard":
    st.markdown("<h1 style='color:#282C3F;'>Discovery Dashboard</h1>", unsafe_allow_html=True)
    st.caption("AI-powered analysis across 8 platforms + primary user survey.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Public Reviews", len(reviews))
    col2.metric("Survey Responses", len(survey))
    col3.metric("Opportunity Areas", len(OPPORTUNITY_AREAS))
    col4.metric("Research Qs Covered", "10 / 10")

    st.subheader("Opportunity Areas — Ranked by Impact")
    for key, opp in sorted(OPPORTUNITY_AREAS.items(), key=lambda x: x[1]["impact"], reverse=True):
        count = sum(1 for r in reviews if key in r["themes"])
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.markdown(f"**{opp['name']}** &nbsp; `{opp['impact']}/10` &nbsp; · &nbsp; {count} reviews")
            st.progress(opp["impact"] / 10)
            st.caption(opp["description"])
        with col_b:
            for p, c in Counter(r["platform"] for r in reviews if key in r["themes"]).most_common(3):
                st.caption(f"{p}: {c}")
        st.divider()

    st.subheader("Reviews by Platform")
    cols = st.columns(len(set(r["platform"] for r in reviews)))
    for i, (plat, cnt) in enumerate(Counter(r["platform"] for r in reviews).most_common()):
        with cols[i]:
            st.metric(plat, cnt)

    st.subheader("Sentiment Distribution")
    emojis = {"negative": "🔴", "mixed": "🟡", "neutral": "⚪", "positive": "🟢"}
    sent_counts = Counter(r["sentiment"] for r in reviews)
    cols2 = st.columns(len(sent_counts))
    for i, (sent, cnt) in enumerate(sent_counts.most_common()):
        with cols2[i]:
            st.metric(f"{emojis.get(sent, '⚪')} {sent.title()}", cnt)

    st.subheader("User Segments Identified")
    for seg, cnt in Counter(r["user_segment"] for r in reviews).most_common(10):
        st.markdown(f"- **{seg}** — {cnt} reviews")


# ═══════════════════════════════════════
#  ASK INSIGHTS FROM PUBLIC REVIEWS
# ═══════════════════════════════════════
elif st.session_state["page"] == "public_reviews":
    st.markdown("<h1 style='color:#282C3F;'>Ask Insights from Public Reviews</h1>", unsafe_allow_html=True)
    st.caption("Ask any question. The engine retrieves relevant reviews and uses AI to synthesize an answer.")

    query = st.text_input(
        "Your question",
        placeholder="e.g., Why do users add items to wishlist but never buy them?",
        key="rev_input",
    )

    if st.button("🔍 Search & Analyze", type="primary", disabled=not query.strip(), key="rev_btn"):
        results = MiniSearch(reviews, field="text").search(query, top_k=15)

        if not results:
            st.warning("No relevant reviews found. Try rephrasing.")
        else:
            # ── AI ANSWER FIRST ──
            st.subheader("🧠 AI-Synthesized Answer")
            if not api_key:
                st.warning("AI synthesis unavailable. Add GROQ_API_KEY to Streamlit secrets.")
            else:
                with st.spinner("Analyzing evidence..."):
                    review_texts = "\n---\n".join(
                        f"[{r['platform']}] [{r['sentiment']}] [{r.get('user_segment','')}] {r['text']}"
                        for r in results
                    )
                    answer = call_groq(REVIEW_SYSTEM, f"Question: {query}\n\nReviews:\n{review_texts}", api_key)
                if answer and not answer.startswith("__ERROR__"):
                    st.markdown(f'<div class="ai-answer">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.error(answer.replace("__ERROR__: ", ""))

            # ── THEN REVIEWS BELOW ──
            st.subheader(f"📎 Source Reviews ({len(results)} retrieved)")
            for r in results:
                themes_str = ", ".join(OPPORTUNITY_AREAS[t]["name"] for t in r["themes"] if t in OPPORTUNITY_AREAS)
                rating_str = f" ⭐ {r['rating']}/5" if r.get("rating") else ""
                st.markdown(
                    f"""<div class="review-card">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <span style="font-size:12px;color:#FF3F6C;font-weight:600;">{r['platform']}{rating_str}</span>
                        <span style="font-size:11px;color:#94969F;">{r.get('date','')}</span>
                    </div>
                    <div style="font-size:13px;color:#282C3F;line-height:1.6;">"{r['text']}"</div>
                    <div style="margin-top:8px;font-size:11px;color:#535766;">
                        Themes: {themes_str} · {r['sentiment']} · {r.get('user_segment','-')}
                    </div></div>""",
                    unsafe_allow_html=True,
                )

            # Theme distribution
            st.subheader("Theme Distribution in Results")
            theme_counts = Counter()
            for r in results:
                for t in r["themes"]:
                    if t in OPPORTUNITY_AREAS:
                        theme_counts[t] += 1
            for t, cnt in theme_counts.most_common():
                st.markdown(f"**{OPPORTUNITY_AREAS[t]['name']}** — {cnt}/{len(results)}")
                st.progress(cnt / len(results))


# ═══════════════════════════════════════
#  ASK INSIGHTS FROM USER SURVEY
# ═══════════════════════════════════════
elif st.session_state["page"] == "user_survey":
    st.markdown("<h1 style='color:#282C3F;'>Ask Insights from User Survey</h1>", unsafe_allow_html=True)
    st.caption(f"Insights from {len(survey)} survey respondents. AI synthesizes patterns from primary research.")

    # Survey overview
    with st.expander("📋 Survey Overview"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Gender**")
            for g, c in Counter(s["demographics"]["gender"] for s in survey if s["demographics"]["gender"]).most_common():
                st.caption(f"{g}: {c}")
        with c2:
            st.markdown("**Age Group**")
            for a, c in Counter(s["demographics"]["age_group"] for s in survey if s["demographics"]["age_group"]).most_common():
                st.caption(f"{a}: {c}")
        with c3:
            st.markdown("**Situation**")
            for sv, c in Counter(s["demographics"]["situation"] for s in survey if s["demographics"]["situation"]).most_common():
                st.caption(f"{sv}: {c}")
        with c4:
            st.markdown("**City**")
            for ci, c in Counter(s["demographics"]["city"] for s in survey if s["demographics"]["city"]).most_common():
                st.caption(f"{ci}: {c}")

        st.markdown("---")
        st.markdown("**Average Barrier Ratings (1=no problem, 5=major)**")
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
                avgs[label] = sum(vals) / len(vals)
        for label, avg in sorted(avgs.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{label}** — {avg:.1f}/5")
            st.progress(avg / 5)

    query = st.text_input(
        "Your question",
        placeholder="e.g., What stops users from buying items they've wishlisted?",
        key="srv_input",
    )

    if st.button("🔍 Search & Analyze", type="primary", disabled=not query.strip(), key="srv_btn"):
        results = MiniSearch(survey, field="searchable_text").search(query, top_k=10)

        if not results:
            st.warning("No matching responses. Try broader terms.")
        else:
            # ── AI ANSWER FIRST ──
            st.subheader("🧠 AI-Synthesized Insight")
            if not api_key:
                st.warning("AI synthesis unavailable. Add GROQ_API_KEY to Streamlit secrets.")
            else:
                with st.spinner("Analyzing survey data..."):
                    survey_texts = "\n---\n".join(
                        f"[Respondent {r['id']}] [{r['demographics'].get('gender','')}] "
                        f"[{r['demographics'].get('age_group','')}] [{r['demographics'].get('city','')}]\n"
                        f"Wishlist meaning: {r.get('wishlist_meaning','N/A')}\n"
                        f"Top blockers: {r.get('top_3_blockers','N/A')}\n"
                        f"Decision factors: {r.get('top_3_decision_factors','N/A')}\n"
                        f"Good purchase: {r.get('good_purchase_story','N/A')}\n"
                        f"Didn't buy: {r.get('didnt_buy_story','N/A')}\n"
                        f"Magic feature: {r.get('magic_feature','N/A')}\n"
                        f"Items in wishlist: {r.get('wishlist_usage',{}).get('items_in_wishlist','N/A')}\n"
                        f"Buy likelihood: {r.get('wishlist_usage',{}).get('perceived_buy_likelihood','N/A')}\n"
                        f"Thinking when adding: {r.get('wishlist_usage',{}).get('thinking_when_adding','N/A')}\n"
                        f"Barriers: {json.dumps(r.get('barriers_ratings',{}))}"
                        for r in results
                    )
                    answer = call_groq(SURVEY_SYSTEM, f"Question: {query}\n\nSurvey data:\n{survey_texts}", api_key)
                if answer and not answer.startswith("__ERROR__"):
                    st.markdown(f'<div class="ai-answer">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.error(answer.replace("__ERROR__: ", ""))

            # ── THEN SURVEY RESPONSES BELOW ──
            st.subheader(f"📎 Source Responses ({len(results)} retrieved)")
            for r in results:
                demo = r["demographics"]
                demo_str = f"{demo.get('gender','')} · {demo.get('age_group','')} · {demo.get('situation','')} · {demo.get('city','')}"
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


# ═══════════════════════════════════════
#  EVIDENCE EXPLORER
# ═══════════════════════════════════════
elif st.session_state["page"] == "explorer":
    st.markdown("<h1 style='color:#282C3F;'>Evidence Explorer</h1>", unsafe_allow_html=True)
    st.caption("Browse and filter the review corpus.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plat_filter = st.multiselect("Platform", sorted(set(r["platform"] for r in reviews)))
    with col2:
        theme_filter = st.multiselect("Theme", [(k, v["name"]) for k, v in OPPORTUNITY_AREAS.items()], format_func=lambda x: x[1])
    with col3:
        sent_filter = st.multiselect("Sentiment", sorted(set(r["sentiment"] for r in reviews)))
    with col4:
        seg_filter = st.multiselect("User Segment", sorted(set(r["user_segment"] for r in reviews)))

    filtered = reviews
    if plat_filter:
        filtered = [r for r in filtered if r["platform"] in plat_filter]
    if theme_filter:
        tkeys = [t[0] for t in theme_filter]
        filtered = [r for r in filtered if any(t in tkeys for t in r["themes"])]
    if sent_filter:
        filtered = [r for r in filtered if r["sentiment"] in sent_filter]
    if seg_filter:
        filtered = [r for r in filtered if r.get("user_segment") in seg_filter]

    st.markdown(f"**Showing {len(filtered)} of {len(reviews)} reviews**")
    st.divider()
    for r in filtered[:50]:
        themes_str = " · ".join(OPPORTUNITY_AREAS[t]["name"] for t in r["themes"] if t in OPPORTUNITY_AREAS)
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
        st.info(f"Showing first 50 of {len(filtered)}. Use filters to narrow down.")


# ═══════════════════════════════════════
#  RQ MAPPING
# ═══════════════════════════════════════
elif st.session_state["page"] == "rq_mapping":
    st.markdown("<h1 style='color:#282C3F;'>Research Question ↔ Evidence Mapping</h1>", unsafe_allow_html=True)
    for rq_id, rq_text in RESEARCH_QUESTIONS:
        with st.expander(f"**{rq_id}**: {rq_text}"):
            themes = RQ_TO_THEMES.get(rq_id, [])
            st.markdown("**Related Opportunity Areas:**")
            for t in themes:
                opp = OPPORTUNITY_AREAS[t]
                count = sum(1 for r in reviews if t in r["themes"])
                st.markdown(f"- {opp['name']} — Impact: `{opp['impact']}/10` · {count} reviews")
            relevant = [r for r in reviews if any(t in r["themes"] for t in themes)]
            st.markdown(f"**Sample Evidence** ({len(relevant)} total)")
            for r in relevant[:5]:
                st.markdown(f"> *\"{r['text']}\"*\n>\n> — {r['platform']}, {r.get('user_segment','')}")


# ═══════════════════════════════════════
#  COMPARISON MATRIX
# ═══════════════════════════════════════
elif st.session_state["page"] == "comparison":
    st.markdown("<h1 style='color:#282C3F;'>Opportunity Comparison Matrix</h1>", unsafe_allow_html=True)

    rows = []
    for key, opp in OPPORTUNITY_AREAS.items():
        count = sum(1 for r in reviews if key in r["themes"])
        sents = Counter(r["sentiment"] for r in reviews if key in r["themes"])
        rows.append({
            "Opportunity Area": opp["name"],
            "Impact (0-10)": opp["impact"],
            "Evidence": count,
            "Platforms": len(set(r["platform"] for r in reviews if key in r["themes"])),
            "RQs": len([rid for rid, th in RQ_TO_THEMES.items() if key in th]),
            "% Negative": round(sents.get("negative", 0) / max(count, 1) * 100),
        })
    rows.sort(key=lambda x: x["Impact (0-10)"], reverse=True)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Impact vs Evidence Volume")
    try:
        import altair as alt
        import pandas as pd
        chart_data = []
        for key, opp in OPPORTUNITY_AREAS.items():
            chart_data.append({"name": opp["name"], "impact": opp["impact"],
                               "evidence": sum(1 for r in reviews if key in r["themes"])})
        df = pd.DataFrame(chart_data)
        chart = alt.Chart(df).mark_circle(size=200, color="#FF3F6C").encode(
            x=alt.X("evidence:Q", title="Evidence Count"),
            y=alt.Y("impact:Q", title="Impact Score", scale=alt.Scale(domain=[5, 10])),
            tooltip=["name", "impact", "evidence"],
        ).properties(height=400).interactive()
        text = alt.Chart(df).mark_text(dy=-15, fontSize=11).encode(x="evidence:Q", y="impact:Q", text="name:N")
        st.altair_chart(chart + text, use_container_width=True)
    except ImportError:
        pass

    st.subheader("Synthesis for Part 2")
    st.success(
        "**Top 3 non-monetary intervention opportunities:**\n\n"
        "1. **Size & Fit Uncertainty** (9.2) — Largest barrier across reviews AND survey. "
        "Inconsistent sizing, no virtual try-on, vague fit descriptions.\n\n"
        "2. **Wishlist Clutter & Decision Fatigue** (8.1) — No filters, categories, or comparison tools. "
        "Wishlists become unmanageable dumping grounds.\n\n"
        "3. **Price Watching & Sale Waiting** (7.8) — Users use wishlists as price trackers. "
        "Solving the information gap is valid without offering discounts.\n\n"
        "**The discovered user problem**: The platform fails to resolve uncertainties between "
        "'I like this' and 'I'm buying this' — and the wishlist compounds it by becoming an "
        "unmanageable backlog that erodes purchase intent over time."
    )
