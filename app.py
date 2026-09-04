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

    if not api_key or len(api_key) < 10:
        return "__ERROR__: No valid API key. Go to Streamlit Cloud → Settings → Secrets and add: GROQ_API_KEY = \"gsk_your-key-here\""

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
                "User-Agent": "MyntraDiscoveryEngine/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass
            last_error = f"Model {model}: HTTP {e.code} — {error_body[:200]}"
            continue
        except Exception as e:
            last_error = f"Model {model}: {e}"
            continue

    return f"__ERROR__: {last_error}"


# ─── Custom CSS (Myntra themed) ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    [data-testid="stToolbar"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    button[data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebarCollapse"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .stSidebar button[kind="header"] { display: none !important; }
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
    ("dashboard", "DASHBOARD"),
    ("public_reviews", "ASK INSIGHTS FROM PUBLIC REVIEWS"),
    ("user_survey", "ASK INSIGHTS FROM USER SURVEY"),
    ("explorer", "REVIEW EXPLORER"),
    ("comparison", "COMPARISON MATRIX"),
]

if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"

with st.sidebar:
    st.markdown(
        "<h2 style='margin:0 0 -8px 0;font-size:20px;'>🔍 Discovery Engine</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='margin:0;padding:0 0 0 32px;font-size:11px;opacity:0.7;line-height:1.3;max-width:190px;'>AI-powered fashion shopping behavior analysis</p>",
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
    st.divider()

    # Load API key from secrets
    api_key = ""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

    if api_key:
        st.markdown(f"🟢 **AI: Connected** (`{api_key[:8]}...`)")
    else:
        st.markdown("🔴 **AI: No key found**")
        st.caption("Add GROQ_API_KEY in Settings → Secrets")

# Make api_key available outside sidebar
api_key = ""
try:
    api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    api_key = ""


# ─── System prompts ───
REVIEW_SYSTEM = (
    "You are a senior user researcher analyzing Myntra (India's top fashion e-commerce) "
    "wishlist-to-purchase conversion. You're given retrieved user reviews as evidence. "
    "Answer the user's question grounded ONLY in the provided reviews. "
    "Structure your answer as follows. Do NOT number the sections.\n\n"
    "Start with a direct 2-3 sentence answer to the question — no heading, no label, just the answer.\n\n"
    "Then a section headed **Key Themes** listing 3-5 prominent patterns with brief evidence from the reviews.\n\n"
    "Then a section headed **User Segments Affected** listing which types of users are most impacted.\n\n"
    "Quote specific reviews as evidence. Be specific, not generic."
)

SURVEY_SYSTEM = (
    "You are a senior user researcher analyzing a primary user survey about "
    "Myntra wishlist behavior. Survey has quantitative (Likert 1-5) and qualitative responses. "
    "Answer grounded ONLY in the provided survey data. "
    "Structure your answer as follows. Do NOT number the sections.\n\n"
    "Start with a direct 2-3 sentence answer — no heading, no label, just the finding.\n\n"
    "Then a section headed **Supporting Data** referencing specific responses, quoting verbatim where useful.\n\n"
    "Then a section headed **Demographic Patterns** noting any differences by gender, age, city.\n\n"
    "Be specific. Reference respondent demographics when quoting."
)


# ═══════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════
if st.session_state["page"] == "dashboard":
    st.markdown("<h1 style='color:#282C3F;'>Discovery Dashboard</h1>", unsafe_allow_html=True)
    st.caption("AI-powered analysis across 7 platforms + primary user survey.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Public Reviews", len(reviews))
    col2.metric("Survey Responses", len(survey))
    col3.metric("Opportunity Areas", len(OPPORTUNITY_AREAS))

    st.subheader("Reviews by Platform")
    cols = st.columns(len(set(r["platform"] for r in reviews)))
    for i, (plat, cnt) in enumerate(Counter(r["platform"] for r in reviews).most_common()):
        with cols[i]:
            st.metric(plat, cnt)

    # ── Affinity Mapping Section ──
    st.subheader("Affinity Mapping")
    with st.expander("View affinity mapping — 140 reviews → 179 codes → 7 themes"):
        st.markdown(
            "Bottom-up inductive thematic coding of 140 user reviews from 7 platforms. "
            "Each review was read as a qualitative observation, assigned 1–3 inductive codes "
            "(179 unique codes total), then codes were clustered into **7 emergent themes**."
        )
        st.markdown("**140** reviews coded · **179** unique codes · **7** themes emerged · 74 negative / 30 mixed / 30 neutral / 6 positive")

        AFFINITY_THEMES = [
            {
                "name": "The Trust Deficit",
                "subtitle": "Nothing on this app is what it claims to be",
                "color": "#DC2626", "obs": 28,
                "platforms": "Reddit: 14, YouTube: 6, Trustpilot: 5, Play Store: 3",
                "description": "Photos lie. Descriptions lie. Star ratings lie. Discounts lie. Users have learned to trust nothing the platform shows them and have built a parallel information ecosystem — YouTube hauls, Reddit threads, third-party price trackers — to do what product pages should do: tell the truth.",
                "codes": ["Photos vs reality (3)", "Color mismatch (4)", "Reviews feel fake (2)", "Fake discounting proven (1)", "YouTube as trust proxy (2)", "Trust destroyed by single experience (3)"],
                "quotes": [
                    ("Trustpilot", "Customer photos look NOTHING like listing. Dress was completely different shade of blue."),
                    ("Reddit", "Product has 4.5 stars but text reviews all complain about quality and sizing. Stars are lies."),
                    ("Reddit", "Tracked jeans for 4 months. 'Discounted' at 1799 was always 1799. Fake discounting."),
                ],
                "wishlist_link": "Trust deficit keeps items stuck in wishlist indefinitely. Users like items but don't trust what they'll receive.",
                "workarounds": "YouTube haul videos, PriceTracker apps, brand websites for size charts, reading only 1-2 star reviews",
            },
            {
                "name": "The Fit Lottery",
                "subtitle": "Size M means 5 different things",
                "color": "#EA580C", "obs": 25,
                "platforms": "Reddit: 12, Play Store: 10, YouTube: 2",
                "description": "Size is not a specification — it's a gamble. The same label means different things across brands, across batches, and even versus the platform's own size recommendations.",
                "codes": ["Cross-brand inconsistency (4)", "Measurement mismatch (4)", "Model unrepresentative (3)", "Recommendation contradicts reviews (2)", "Vague fit descriptions (3)", "User workaround math (1)"],
                "quotes": [
                    ("YouTube", "Measured 5 'M size' tees from different brands. Chest ranged 38 to 42 inches."),
                    ("Reddit", "Model is 5'9\". I'm 5'2\". Dress will look completely different on me."),
                    ("Play Store", "Size recommendation says M, reviews say order L. Who do I trust? Ended up not buying."),
                ],
                "wishlist_link": "8 reviews explicitly say items 'stay in wishlist' because of fit uncertainty. #1 reason wishlisted items don't convert.",
                "workarounds": "Only buying known brands, consulting tailors, ordering 2 sizes, measuring own garments",
            },
            {
                "name": "The Wishlist Graveyard",
                "subtitle": "200 items, no tools, no hope",
                "color": "#7C3AED", "obs": 28,
                "platforms": "Reddit: 12, App Store: 9, Play Store: 5",
                "description": "The wishlist has no tools — no filters, no sorting, no categories, no comparison, no smart notifications, no auto-cleanup. Users who try to use it as a shopping tool are forced into external workarounds.",
                "codes": ["No filters/sorting (3)", "No side-by-side comparison (2)", "Out-of-stock clogging (2)", "Notification waste (4)", "Manipulative notifications (2)", "Wishlist overwhelm (2)", "Wants decision help not reminders (1)"],
                "quotes": [
                    ("Play Store", "200+ items. Scroll endlessly. Half out of stock. No filters. Basic feature missing."),
                    ("Reddit", "I maintain a spreadsheet of wishlist items with sizes, prices, and review notes. Because Myntra gives zero tools."),
                    ("Reddit", "Wishlist to purchase ratio probably 50:1. Great for discovery, terrible for decision-making."),
                ],
                "wishlist_link": "This IS the wishlist problem. Every other theme feeds into this one — the graveyard accumulates items that other themes prevent from converting.",
                "workarounds": "External spreadsheets, cart used as 'real wishlist', periodic mass-deletion, turning off all notifications",
            },
            {
                "name": "The Return Trap",
                "subtitle": "Buying online = accepting you might lose your money",
                "color": "#B91C1C", "obs": 17,
                "platforms": "Reddit: 7, PissedConsumer: 4, Play Store: 4",
                "description": "Returns are designed to be punitive rather than supportive. Seal-tag catch-22s, shrinking return windows, and failed quality checks create a system where buying = accepting risk.",
                "codes": ["Seal tag catch-22 (3)", "False quality check rejection (2)", "No return during sale (2)", "Return pickup delayed (2)", "Trust destroyed (2)", "Offline migration (1)"],
                "quotes": [
                    ("Trustpilot", "Returned shoes, never opened inner box. Failed quality check. Lost 3200rs. Never buying expensive items again."),
                    ("Reddit", "How am I supposed to check if a shirt fits without removing the seal tag? Catch-22."),
                    ("Reddit", "Recently removing return option on more products. How to buy clothes online without returns?"),
                ],
                "wishlist_link": "Return anxiety compounds fit uncertainty. Users keep items in wishlist because the cost of a wrong purchase is too high.",
                "workarounds": "Avoiding non-returnable items, only buying during non-sale for return eligibility, sticking to low-cost items",
            },
            {
                "name": "The Social Decision Gap",
                "subtitle": "No one to ask, nowhere to check, no way to be sure",
                "color": "#2563EB", "obs": 18,
                "platforms": "Reddit: 11, YouTube: 5, Play Store: 1",
                "description": "Fashion purchases are inherently social decisions but Myntra treats them as solo transactions. Users need styling context, peer validation, and real-life product views that the platform doesn't provide.",
                "codes": ["External validation seeking (3)", "Styling uncertainty (3)", "Real-body representation needed (2)", "Fabric/material unknown (3)", "Community as product research (2)"],
                "quotes": [
                    ("Reddit", "Send wishlist screenshots to friends on WhatsApp. Takes 2-3 days. A share/poll feature inside Myntra would be amazing."),
                    ("Reddit", "Always check YouTube reviews. Product photos lie. Need to see how fabric falls on a real person."),
                    ("Play Store", "Spent 3 weeks with item in wishlist before buying because couldn't figure out what to pair it with."),
                ],
                "wishlist_link": "Users can't resolve styling/validation questions within the app, so items stay in wishlist while they consult external sources.",
                "workarounds": "WhatsApp screenshots to friends, YouTube haul watching, Instagram outfit checks",
            },
            {
                "name": "The Price Game",
                "subtitle": "Everyone's waiting for a sale that may never come",
                "color": "#059669", "obs": 14,
                "platforms": "Reddit: 8, Play Store: 5, YouTube: 1",
                "description": "Users have learned that Myntra's pricing is cyclical. The rational behavior is to wishlist at full price and wait for EORS/Big Fashion Festival. Third-party price trackers validate this strategy.",
                "codes": ["Sale-cycle shopping (3)", "Price tracking via third-party (2)", "Wishlist as price watch (3)", "Price manipulation suspicion (2)", "Budget gating (2)"],
                "quotes": [
                    ("Reddit", "Added jacket at 2400. During EORS dropped to 1100. Bought immediately. Wishlist = price watch list."),
                    ("Reddit", "Only buying during Big Fashion Festival. Rest of year = browsing and wishlisting."),
                    ("Reddit", "Using PriceHistory app for Myntra. Why doesn't Myntra just show price history?"),
                ],
                "wishlist_link": "Large segment explicitly uses wishlists as price-watch lists. Items don't convert because users are rationally waiting for predictable sale cycles.",
                "workarounds": "PriceHistory/PriceTracker apps, EORS calendar tracking, cross-platform price comparison",
            },
            {
                "name": "The Idle Accumulator",
                "subtitle": "I saved it because saving felt good, not because I want it",
                "color": "#6B7280", "obs": 10,
                "platforms": "Reddit: 10",
                "description": "A significant portion of wishlist additions have zero purchase intent. Users save items for dopamine, inspiration, gift hints, or as bookmarks for offline shopping. The act of saving IS the experience.",
                "codes": ["Boredom browsing (2)", "Aspirational saving (1)", "Gift registry usage (2)", "Offline reference usage (1)", "Mood board behavior (2)", "Night browsing regret (1)"],
                "quotes": [
                    ("Reddit", "I add things when bored browsing at night. No real intention to buy. Maybe 1 in 20 I actually purchase."),
                    ("Reddit", "College student. Wishlist expensive stuff as aspirational shopping. Can't afford now but saving for when I start working."),
                    ("Reddit", "Use Myntra like Instagram. Scroll, heart things, never buy. App encourages browsing not buying."),
                ],
                "wishlist_link": "These additions were never purchase intent. They're noise in the wishlist that buries the items that might have converted.",
                "workarounds": "Periodic wishlist purges, separate mental categories for 'real' vs 'aspirational' saves",
            },
        ]

        # Theme bar
        total_obs = sum(t["obs"] for t in AFFINITY_THEMES)
        bar_html = '<div style="display:flex;height:24px;border-radius:5px;overflow:hidden;gap:2px;margin:12px 0 8px;">'
        for t in AFFINITY_THEMES:
            pct = (t["obs"] / total_obs) * 100
            bar_html += f'<div style="width:{pct}%;background:{t["color"]};display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff;min-width:24px;">{t["obs"]}</div>'
        bar_html += '</div>'
        legend_html = '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">'
        for t in AFFINITY_THEMES:
            legend_html += f'<span style="font-size:11px;color:#666;display:flex;align-items:center;gap:4px;"><span style="width:8px;height:8px;border-radius:2px;background:{t["color"]};display:inline-block;"></span>{t["name"]}</span>'
        legend_html += '</div>'
        st.markdown(bar_html + legend_html, unsafe_allow_html=True)

        # Each theme as a sub-expander
        for t in AFFINITY_THEMES:
            with st.expander(f'{t["name"]} — "{t["subtitle"]}" ({t["obs"]} observations)'):
                st.markdown(f'**Platforms:** {t["platforms"]}')
                st.markdown(t["description"])

                st.markdown("**Codes identified:**")
                for c in t["codes"]:
                    st.markdown(f"- {c}")

                st.markdown("**Key quotes:**")
                for plat, quote in t["quotes"]:
                    st.markdown(f'> *"{quote}"*\n>\n> — {plat}')

                st.markdown(f'**Link to wishlist conversion:** {t["wishlist_link"]}')
                st.markdown(f'**User workarounds:** {t["workarounds"]}')

        # Reinforcing loop
        st.markdown("---")
        st.markdown("**The Reinforcing Loop**")
        st.markdown(
            "These seven themes aren't independent — they form a reinforcing system that keeps items trapped in the wishlist. "
            "The Fit Lottery creates uncertainty. The Trust Deficit means users can't resolve it from product pages. "
            "The Return Trap means they can't resolve it by trying the product either. "
            "The Social Decision Gap means they can't get help within the app. "
            "The Wishlist Graveyard means the platform provides no tools to manage the accumulation."
        )
        st.markdown(
            "**The core insight:** the wishlist doesn't have a conversion problem — it has a confidence problem "
            "wrapped in a tools problem wrapped in a trust problem. Users want to buy. They add items because something "
            "genuinely attracted them. But the platform gives them no way to move from attraction to confidence."
        )

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
    st.markdown("<h1 style='color:#282C3F;'>Review Explorer</h1>", unsafe_allow_html=True)
    st.caption("Browse and filter the review corpus by platform, theme, sentiment, and user segment.")

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

# ═══════════════════════════════════════
#  COMPARISON MATRIX
# ═══════════════════════════════════════
elif st.session_state["page"] == "comparison":
    st.markdown("<h1 style='color:#282C3F;'>Impact vs Evidence Volume</h1>", unsafe_allow_html=True)

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
        st.info("Chart requires altair library.")
