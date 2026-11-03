"""
Nigerian News Classifier AI
---------------------------
A Streamlit web app that classifies a news article as
Business, Politics or Sports using a saved TF-IDF vectorizer + SVM model.

Run with:  streamlit run app.py
"""

import base64
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
CSS_PATH = BASE_DIR / "style.css"
HERO_IMAGE = BASE_DIR / "assets" / "hero.png"      # banner at the top
FOOTER_IMAGE = BASE_DIR / "assets" / "footer.png"  # image card in the sidebar

st.set_page_config(
    page_title="Nigerian News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Look of each category ("icon" is a name from ICONS, "css" matches style.css)
CATEGORIES = {
    "Business": {"icon": "chart", "css": "business"},
    "Politics": {"icon": "landmark", "css": "politics"},
    "Sports": {"icon": "soccer", "css": "sports"},
}

# Sample articles used by the "Load ... Example" buttons
SAMPLES = {
    "Business": (
        "The Central Bank of Nigeria has announced new monetary policies aimed at "
        "stabilizing the naira. The CBN governor stated that the new reforms will "
        "attract foreign investment and boost the economy. Analysts say inflation "
        "and interest rates remain a concern for banks and companies."
    ),
    "Politics": (
        "The Senate has passed the electoral amendment bill after a lengthy debate. "
        "The President is expected to sign the bill into law, while opposition "
        "lawmakers say INEC needs more independence ahead of the next general "
        "election. Governors and party leaders have also reacted to the new policy."
    ),
    "Sports": (
        "The Super Eagles defeated their opponents 2-1 in a thrilling World Cup "
        "qualifier on Sunday. The striker scored twice in the second half, and the "
        "coach praised the players for their performance. Fans celebrated in the "
        "stadium as the team moved to the top of the group."
    ),
}

# Small built-in icons (no internet needed). Each one is the inside of a 24x24 SVG.
ICONS = {
    "house": '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h5v-6h4v6h5V9.5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.5v.01"/>',
    "newspaper": '<rect x="3" y="4" width="15" height="16" rx="2"/><path d="M18 9h3v9a2 2 0 0 1-2 2h-1"/><path d="M7 8h7M7 12h7M7 16h4"/>',
    "mail": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/>',
    "pen": '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h6"/>',
    "chart": '<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 5-6"/>',
    "landmark": '<path d="M3 10 12 4l9 6Z"/><path d="M5 10v9M9.5 10v9M14.5 10v9M19 10v9M3 21h18"/>',
    "soccer": '<circle cx="12" cy="12" r="9"/><path d="m12 8 3.5 2.5-1.3 4h-4.4l-1.3-4Z"/><path d="M12 3v5M15.5 10.5 20 9M14.2 14.5 17 18.5M9.8 14.5 7 18.5M8.5 10.5 4 9"/>',
    "check": '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    "sliders": '<path d="M4 6h8M18 6h2M4 12h2M12 12h8M4 18h10M20 18h0"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>',
    "award": '<circle cx="12" cy="9" r="6"/><path d="m8.5 14-1.5 7 5-3 5 3-1.5-7"/>',
    "bulb": '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-3.5 10.9c.7.6 1 1.3 1 2.1h5c0-.8.3-1.5 1-2.1A6 6 0 0 0 12 3Z"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
}


# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the saved TF-IDF vectorizer and SVM model (only once)."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


@st.cache_data
def image_to_data_uri(path: Path) -> str:
    """Turn an image file into text so CSS can use it as a background."""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def html(text: str) -> str:
    """Clean up HTML for Streamlit and swap [[icon_name]] for the real SVG icon."""
    text = "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())
    for name, shape in ICONS.items():
        svg = (
            '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{shape}</svg>'
        )
        text = text.replace(f"[[{name}]]", svg)
    return text


def load_css():
    """Inject style.css and the two background images into the page."""
    css = CSS_PATH.read_text(encoding="utf-8")
    images = (
        f":root{{--hero-img:url('{image_to_data_uri(HERO_IMAGE)}');"
        f"--footer-img:url('{image_to_data_uri(FOOTER_IMAGE)}');}}"
    )
    st.markdown(f"<style>{css}\n{images}</style>", unsafe_allow_html=True)


def classify(text: str):
    """Return (category, confidence, error_message).

    confidence is a number from 0 to 100, or None if there was an error.
    The SVM gives one score per category (decision_function). We turn those
    scores into percentages with softmax so that the three values add up to 100%.
    """
    try:
        vectorizer, model = load_model()
        features = vectorizer.transform([text])  # same input format used in training
        if features.nnz == 0:
            return None, None, (
                "The model does not recognise any words in this text. "
                "Please paste a real news headline or article."
            )
        category = str(model.predict(features)[0])
        scores = model.decision_function(features)[0]
        exp_scores = np.exp(scores - scores.max())
        confidence = float(exp_scores.max() / exp_scores.sum() * 100)
        return category, confidence, None
    except Exception as error:  # show a friendly message instead of crashing
        return None, None, f"Something went wrong while classifying: {error}"


# ---------------------------------------------------------------------------
# 3. BUTTON ACTIONS (callbacks)
# ---------------------------------------------------------------------------
def load_sample(category: str):
    st.session_state.article = SAMPLES[category]
    st.session_state.result = None
    st.session_state.confidence = None
    st.session_state.message = None


def clear_text():
    st.session_state.article = ""
    st.session_state.result = None
    st.session_state.confidence = None
    st.session_state.message = None


def forget_old_result():
    st.session_state.result = None
    st.session_state.confidence = None


# ---------------------------------------------------------------------------
# 4. PAGE PARTS
# ---------------------------------------------------------------------------
def render_navbar_and_hero():
    st.markdown(
        html(
            """
            <div id="top"></div>
            <div class="navbar">
              <div class="brand">
                <div class="brand-logo">[[newspaper]]</div>
                <div>
                  <div class="brand-title">NIGERIAN NEWS CLASSIFIER</div>
                  <div class="brand-sub">Automatic Classification of News Articles (Politics, Sports, Business)</div>
                </div>
              </div>
              <nav class="nav-links">
                <a class="active" href="#top">[[house]]<span>Home</span></a>
                <a href="#about">[[info]]<span>About</span></a>
                <a href="#news">[[newspaper]]<span>News</span></a>
                <a href="#contact">[[mail]]<span>Contact</span></a>
              </nav>
            </div>
            <div class="hero">
              <div class="hero-title">Stay Informed,<br><span class="mint">Nigeria’s News,</span> Classified</div>
              <p>Our AI reads the news for you and tells you what it’s about: Politics, Sports or Business.</p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_input_card():
    with st.container(key="card_input"):
        st.markdown(
            html(
                """
                <div class="card-title">[[pen]] TEXT INPUT AREA</div>
                <p class="card-sub">Paste your news headline or article content in the box below:</p>
                """
            ),
            unsafe_allow_html=True,
        )
        text = st.text_area(
            "News article",
            key="article",
            height=170,
            placeholder="Type or paste a news headline or article here...",
            label_visibility="collapsed",
            on_change=forget_old_result,
        )

        classify_col, clear_col = st.columns([4, 1])
        with classify_col:
            clicked = st.button(
                "Classify Article",
                key="classify",
                type="primary",
                icon=":material/search:",
            )
        with clear_col:
            st.button("Clear", key="clear", icon=":material/close:", on_click=clear_text)

        if clicked:
            if not text.strip():
                st.session_state.result = None
                st.session_state.confidence = None
                st.session_state.message = "Please enter a news headline or article first."
            else:
                category, confidence, error = classify(text)
                st.session_state.result = category
                st.session_state.confidence = confidence
                st.session_state.message = error

        if st.session_state.message:
            st.warning(st.session_state.message)


def render_examples_card():
    with st.container(key="card_examples"):
        st.markdown(
            html(
                '<div class="card-title" id="news">[[file]] '
                "OR DON’T HAVE AN ARTICLE? TRY A LIVE EXAMPLE</div>"
            ),
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        buttons = [
            (col1, "Business", ":material/monitoring:"),
            (col2, "Politics", ":material/account_balance:"),
            (col3, "Sports", ":material/sports_soccer:"),
        ]
        for column, name, icon in buttons:
            with column:
                st.button(
                    f"Load {name} Example",
                    key=f"ex_{name.lower()}",
                    icon=icon,
                    on_click=load_sample,
                    args=(name,),
                )


def render_result_card():
    category = st.session_state.result
    confidence = st.session_state.confidence
    if not category:
        return  # the result only appears after classification

    # If the model ever returns an unexpected label, fall back to a plain style
    style = CATEGORIES.get(category, {"icon": "newspaper", "css": "business"})
    with st.container(key="card_result"):
        st.markdown(
            html(
                f"""
                <div class="result cat-{style['css']}">
                  <div class="result-head">[[target]] PREDICTION RESULT</div>
                  <div class="result-banner">
                    <div class="rb-check">[[check]]</div>
                    <div class="rb-text">
                      <div class="rb-label">Predicted Category:</div>
                      <div class="rb-value">{category}</div>
                    </div>
                    <div class="rb-cat-icon">[[{style['icon']}]]</div>
                  </div>
                  <div class="result-confidence">
                    [[chart]]
                    <span class="rc-label">Confidence Score:</span>
                    <span class="rc-value">{confidence:.1f}%</span>
                  </div>
                  <div class="result-model">
                    [[sliders]]
                    Model used: Support Vector Machine (SVM) with TF-IDF features
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )


def render_sidebar_cards():
    st.markdown(
        html(
            """
            <div class="card side-card">
              <div class="card-title side-title">[[award]] Quick Info</div>
              <div class="qi-row">
                <div class="qi-icon business">[[chart]]</div>
                <div><div class="qi-name business">Business</div><div class="qi-desc">Economy, Finance, Companies</div></div>
              </div>
              <div class="qi-row">
                <div class="qi-icon politics">[[landmark]]</div>
                <div><div class="qi-name politics">Politics</div><div class="qi-desc">Government, Elections, Policy</div></div>
              </div>
              <div class="qi-row">
                <div class="qi-icon sports">[[soccer]]</div>
                <div><div class="qi-name sports">Sports</div><div class="qi-desc">Football, Athletics, Competitions</div></div>
              </div>
            </div>

            <div class="card side-card about-card" id="about">
              <div class="card-title side-title">[[bulb]] About This App</div>
              <p class="about-text">Nigerian News Classifier uses machine learning to automatically classify
              news articles into Politics, Sports, and Business. Powered by TF-IDF and Support Vector Machine (SVM),
              it makes news categorisation simple, fast, and efficient.</p>
              <hr class="about-divider">
              <div class="about-quote">“Smarter News. Better Decisions.”</div>
            </div>

            <div class="photo-card">
              <div class="photo-text">
                <span class="flag-chip"></span>
                <span>Nigeria’s Growth<br>Through Informed Decisions.</span>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<div class="footer" id="contact">Nigerian News Classifier'
        '<span class="sep">|</span>Built with Streamlit ❤️</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 5. MAIN PROGRAM
# ---------------------------------------------------------------------------
def main():
    # Starting values for the page
    st.session_state.setdefault("article", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("confidence", None)
    st.session_state.setdefault("message", None)

    load_css()

    # Stop with a clear message if the model files are missing or broken
    try:
        load_model()
    except Exception as error:
        st.error(
            "The model files could not be loaded. Make sure `models/svm_model.pkl` and "
            f"`models/tfidf_vectorizer.pkl` exist.\n\nDetails: {error}"
        )
        st.stop()

    render_navbar_and_hero()

    with st.container(key="main_area"):
        left, right = st.columns([7, 3], gap="large")
        with left:
            render_input_card()
            render_examples_card()
            render_result_card()
        with right:
            render_sidebar_cards()

    render_footer()


main()
