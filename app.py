# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered",
)


def inject_app_styles():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        :root {
            --bg: #fafafa;
            --surface: #ffffff;
            --border: #e4e4e7;
            --text: #18181b;
            --muted: #71717a;
            --accent: #18181b;
            --accent-soft: #f4f4f5;
            --radius: 14px;
            --shadow: 0 1px 2px rgba(24, 24, 27, 0.06), 0 8px 24px rgba(24, 24, 27, 0.06);
        }

        html, body, [class*="css"] {
            font-family: "Plus Jakarta Sans", system-ui, sans-serif;
        }

        .stApp {
            background: var(--bg);
            background-image:
                radial-gradient(circle at 1px 1px, rgba(24, 24, 27, 0.04) 1px, transparent 0);
            background-size: 24px 24px;
        }

        .block-container {
            max-width: 760px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }

        .app-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.85rem 1rem;
            margin: -1rem -1rem 1.5rem -1rem;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            min-width: 0;
        }

        .brand-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--accent);
            color: #fff;
            font-size: 1rem;
            flex-shrink: 0;
        }

        .brand-title {
            margin: 0;
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text);
            line-height: 1.2;
        }

        .brand-sub {
            margin: 0;
            font-size: 0.78rem;
            color: var(--muted);
            line-height: 1.2;
        }

        .env-badge {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text);
            background: var(--accent-soft);
            border: 1px solid var(--border);
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            white-space: nowrap;
        }

        .page-head h1 {
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--text);
        }

        .page-head p {
            margin: 0.35rem 0 0 0;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0 0 1.25rem 0;
        }

        .pill {
            font-size: 0.75rem;
            font-weight: 600;
            color: #3f3f46;
            background: #fff;
            border: 1px solid var(--border);
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
        }

        .panel-title {
            margin: 0 0 0.15rem 0;
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text);
        }

        .panel-desc {
            margin: 0 0 1rem 0;
            font-size: 0.82rem;
            color: var(--muted);
        }

        .field-label {
            margin: 0.25rem 0 0.5rem 0;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #52525b;
        }

        .result-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin: 1.5rem 0 0.75rem 0;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid var(--border);
        }

        .result-meta-left {
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }

        .live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.15);
        }

        .result-title {
            margin: 0;
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text);
        }

        .result-tag {
            font-size: 0.72rem;
            font-weight: 600;
            color: #3f3f46;
            background: #fff;
            border: 1px solid var(--border);
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
            background: var(--surface) !important;
            box-shadow: var(--shadow);
            padding-top: 0.75rem;
            padding-bottom: 0.75rem;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] textarea {
            border-radius: 10px !important;
            border-color: var(--border) !important;
        }

        div[data-baseweb="radio"] label {
            background: #fff !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 0.45rem 0.75rem !important;
        }

        div.stButton > button[kind="primary"] {
            background: var(--accent) !important;
            border-radius: 12px !important;
            min-height: 2.75rem;
            font-weight: 700 !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: #fff;
            box-shadow: var(--shadow);
        }

        hr {
            margin: 1.25rem 0 !important;
            border-color: var(--border) !important;
        }

        footer, #MainMenu { visibility: hidden; height: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_app_styles()
