# 🔍 Myntra Wishlist → Purchase Discovery Engine

An AI-powered user research tool that analyzes public feedback and primary survey data to discover **what stops Myntra users from buying wishlisted items**.

Discovery Engine

---

## What It Does

| Feature | Description |
|---|---|
| **Dashboard** | Overview of 8 opportunity areas ranked by impact, platform breakdown, sentiment distribution, and user segments |
| **Ask Insights from Public Reviews** | Type any question — the engine retrieves relevant reviews using TF-IDF search, then uses AI to synthesize an evidence-grounded answer |
| **Ask Insights from User Survey** | Same RAG-style search over 25 primary survey responses with AI-synthesized insights referencing demographics and barrier ratings |
| **Review Explorer** | Filter and browse 140+ user reviews by platform, theme, sentiment, and user segment |
| **Comparison Matrix** | Interactive impact vs evidence scatter plot comparing all opportunity areas |

## Data Sources

**Public Reviews (140+)** collected from:
- Play Store — Myntra app reviews
- Reddit — r/india, r/IndianFashionAddicts, and related subreddits
- YouTube — Myntra haul and review video comments
- App Store — iOS app reviews
- Trustpilot — Myntra company reviews
- PissedConsumer — Consumer complaint data
- Twitter/X — Fashion shopping discussions

**Primary User Survey (25 responses)** covering:
- Shopping behavior and wishlist usage patterns
- Barrier ratings across 12 friction categories (Likert 1-5)
- Open-ended responses on purchase decisions, wishlist meaning, and unmet needs
- Demographics: gender, age group, city, employment status

## 8 Opportunity Areas Discovered

1. **Size & Fit Uncertainty** (Impact: 9.2/10)
2. **Wishlist Clutter & Decision Fatigue** (Impact: 8.1/10)
3. **Price Watching & Sale Waiting** (Impact: 7.8/10)
4. **Return Process Anxiety** (Impact: 7.5/10)
5. **Social & Styling Validation Gap** (Impact: 7.2/10)
6. **Occasion-Driven Postponement** (Impact: 6.8/10)
7. **No Comparison Infrastructure** (Impact: 6.5/10)
8. **Product Authenticity & Quality Doubts** (Impact: 6.3/10)

---

## Setup & Deployment

### Local Development

```bash
git clone https://github.com/YOUR_USERNAME/myntra-wishlist-discovery-engine.git
cd myntra-wishlist-discovery-engine
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → Branch: `main` → Main file: `app.py` → **Deploy**
4. Add your Groq API key in **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_your-key-here"
   ```

Get a free Groq API key at [console.groq.com](https://console.groq.com) (no payment required).

### Project Structure

```
myntra-wishlist-discovery-engine/
├── app.py                  # Main Streamlit application
├── data/
│   ├── reviews.json        # Public review corpus (140+ reviews)
│   └── survey.json         # Primary user survey (25 responses)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
└── README.md
```

---

## Tech Stack

- **Streamlit** — UI framework
- **TF-IDF Search** — Lightweight retrieval engine (custom implementation, no external dependencies)
- **Groq API** — AI synthesis for RAG-style question answering
- **Altair** — Interactive data visualization

## Methodology

1. **Data Collection** — Public reviews scraped and collected from 7 platforms; primary survey conducted with 25 users
2. **Theme Tagging** — Each review tagged with relevant opportunity areas across 8 friction themes
3. **Sentiment & Segment Classification** — Reviews classified by sentiment and inferred user segment
4. **Impact Scoring** — Each opportunity area scored 0–10 based on frequency, severity, and breadth of evidence
5. **RAG Search** — TF-IDF retrieval + Groq AI synthesis for interactive question-answering across both data sources

---

*Built as part of PM Growth Course — Myntra Wishlist Conversion Case Study*
