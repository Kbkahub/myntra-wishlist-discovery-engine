# 🔍 Myntra Wishlist → Purchase Discovery Engine

An AI-powered user research tool that analyzes public feedback at scale to discover **why Myntra users add products to their wishlist but don't purchase them**.

Built for the PM Growth Case Study — Part 1: Discovery Engine.

## Live Demo

[▶ Open on Streamlit Cloud](https://your-app-url.streamlit.app)

---

## What It Does

| Feature | Description |
|---|---|
| **📊 Dashboard** | Overview of 8 opportunity areas ranked by impact, with evidence counts, platform breakdown, sentiment distribution, and user segments |
| **💬 Ask the Data** | RAG-style search — type any question, the engine retrieves relevant reviews using TF-IDF, then uses Claude API to synthesize an evidence-grounded answer |
| **📑 Evidence Explorer** | Filter and browse 120+ user reviews by platform, theme, sentiment, and user segment |
| **🗺️ RQ Mapping** | Maps each of the 10 research questions to opportunity areas with sample evidence |
| **⚖️ Comparison Matrix** | Side-by-side comparison of all opportunity areas with an interactive impact vs evidence scatter plot |

## Data Sources

The review corpus includes feedback collected from:
- **Play Store** — Myntra app reviews
- **App Store** — iOS app reviews  
- **Reddit** — r/india, r/IndianFashionAddicts, and related subreddits
- **YouTube** — Myntra haul and review video comments
- **Trustpilot** — Myntra company reviews
- **PissedConsumer** — Consumer complaint data
- **Twitter/X** — Fashion shopping discussions

## 8 Opportunity Areas Discovered

1. **Size & Fit Uncertainty** (Impact: 9.2/10) — Inconsistent sizing across brands, no virtual try-on, vague fit descriptions
2. **Wishlist Clutter & Decision Fatigue** (Impact: 8.1/10) — No filters, sorting, or organization tools in wishlist
3. **Price Watching & Sale Waiting** (Impact: 7.8/10) — Users use wishlist as price tracker, waiting for sales
4. **Return Process Anxiety** (Impact: 7.5/10) — Failed quality checks, refund delays, restrictive return policies
5. **Social & Styling Validation Gap** (Impact: 7.2/10) — Need to see real-life looks and get peer opinions
6. **Occasion-Driven Postponement** (Impact: 6.8/10) — Wishlisting for future events with no revisit triggers
7. **No Comparison Infrastructure** (Impact: 6.5/10) — Can't compare similar wishlisted items side by side
8. **Product Authenticity & Quality Doubts** (Impact: 6.3/10) — Mismatch between photos and actual products

---

## Setup & Deployment

### Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/myntra-wishlist-discovery.git
cd myntra-wishlist-discovery

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repo → Branch: `main` → Main file: `app.py`
5. (Optional) Add your Anthropic API key in **Settings → Secrets**:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
6. Click **Deploy**

The app works without an API key (dashboard, explorer, mapping all function). The Claude-powered "Ask the Data" synthesis requires the API key.

### Project Structure

```
myntra-wishlist-discovery/
├── app.py                  # Main Streamlit application
├── data/
│   └── reviews.json        # Pre-collected review corpus (120+ reviews)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
└── README.md               # This file
```

---

## Tech Stack

- **Streamlit** — UI framework
- **TF-IDF Search** — Lightweight retrieval (no sklearn dependency, custom implementation)
- **Claude API (Sonnet 4.6)** — AI synthesis for question-answering
- **Altair** — Interactive data visualization

## Methodology

1. **Data Collection** — Scraped and manually collected user reviews from 7+ platforms
2. **Theme Tagging** — Each review tagged with relevant opportunity areas using a taxonomy of 8 friction themes
3. **Sentiment & Segment Classification** — Reviews classified by sentiment and inferred user segment
4. **Impact Scoring** — Each opportunity area scored 0-10 based on frequency, severity, and breadth of evidence
5. **RAG Search** — TF-IDF retrieval + Claude synthesis for interactive question-answering

---

*Built as part of PM Growth Course — Myntra Wishlist Conversion Case Study*
