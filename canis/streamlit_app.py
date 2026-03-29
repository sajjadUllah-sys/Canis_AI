"""
streamlit_app.py

CANIS Testing Interface
Run with: streamlit run streamlit_app.py

Lets you test the full AI pipeline:
- Dog profile injection
- Plan tier switching
- Full conversation with history
- Session metadata tracking
- RAG context visibility (debug mode)
- Risk tier display
"""

import os
import sys
import json
import zipfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path so canis package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canis.chat import get_canis_response
from canis.session_tracker import SessionMetadata
from canis.rag.retriever import Retriever

# ── Database download constants ───────────────────────────────────
DB_DIR = "chroma_db"
ZIP_PATH = "chroma_db.zip"
FILE_ID = "1TrvepoUnEJEtfcRv6b1UTVp78_vh7YS7"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"


@st.cache_resource(show_spinner=False)
def setup_database():
    """
    Ensures the ChromaDB folder exists.
    On first run (e.g. Streamlit Cloud), downloads the prebuilt
    chroma_db.zip from Google Drive and extracts it.
    """
    if not os.path.exists(DB_DIR):
        import gdown

        with st.spinner("📦 Downloading knowledge base — this only happens once…"):
            gdown.download(DOWNLOAD_URL, ZIP_PATH, quiet=False)

        with st.spinner("📂 Extracting knowledge base…"):
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(".")

        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)

        st.success("✅ Knowledge base ready!")
    return True


@st.cache_resource(show_spinner=False)
def get_retriever():
    """Create and cache a Retriever instance."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return Retriever(db_dir=DB_DIR, api_key=api_key)


# ── Ensure DB is ready before anything else ───────────────────────
setup_database()

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CANIS — Testing Interface",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Risk tier colors ──────────────────────────────────────────────
TIER_CONFIG = {
    "green":  {"color": "#28a745", "label": "🟢 Educational",         "bg": "#d4edda"},
    "yellow": {"color": "#ffc107", "label": "🟡 Structured Guidance",  "bg": "#fff3cd"},
    "orange": {"color": "#fd7e14", "label": "🟠 Professional Referral","bg": "#fde5cc"},
    "red":    {"color": "#dc3545", "label": "🔴 Veterinary Emergency", "bg": "#f8d7da"},
}


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════

def init_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "session_metadata" not in st.session_state:
        st.session_state.session_metadata = SessionMetadata.new().to_dict()
    if "last_risk_tier" not in st.session_state:
        st.session_state.last_risk_tier = "green"
    if "debug_system_prompt" not in st.session_state:
        st.session_state.debug_system_prompt = ""

init_state()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — Profile & Settings
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🐕 CANIS Testing")
    st.caption("Dog Psychology AI — Test Interface")



    # ── Plan Tier ─────────────────────────────────────────────────
    st.subheader("📋 Plan Tier")
    plan_tier = st.radio(
        "Select user plan",
        options=["basic", "premium"],
        index=1,
        horizontal=True,
        help="Basic: concepts only. Premium: full protocols."
    )

    st.divider()

    # ── Dog Profile ───────────────────────────────────────────────
    st.subheader("🐶 Dog Profile")

    with st.expander("Basic Information", expanded=True):
        dog_name   = st.text_input("Name", value="Jenny")
        dog_breed  = st.text_input("Breed", value="Golden Retriever")
        dog_age    = st.text_input("Age", value="6 years")
        dog_gender = st.selectbox("Gender", ["Female", "Male"])
        dog_weight = st.text_input("Weight", value="34kg")
        neutered   = st.checkbox("Neutered/Spayed", value=True)

    with st.expander("Background & Living"):
        origin     = st.text_input("Origin", value="Shelter dog")
        home_since = st.text_input("At home since", value="May 3rd, 2021")
        home_type  = st.selectbox("Home type", ["Apartment", "House", "Farm", "Other"])
        other_pets = st.text_input("Other pets (comma-separated)", value="Cat, Parrot")
        family     = st.text_input("Family members", value="2 Adults")

    with st.expander("Health & Sensitivities"):
        medical_raw  = st.text_input("Medical conditions (comma-separated)", value="Anxiety, Hip Dysplasia")
        triggers_raw = st.text_input("Triggers/Sensitivities (comma-separated)", value="Thunder, Vacuum")

    with st.expander("Behavior & Personality"):
        behavioral_raw = st.text_input("Behavioral conditions (comma-separated)", value="Reactivity, Barking Issues")
        traits_raw     = st.text_input("Behavioral traits (comma-separated)", value="Focused, Gentle")
        extra_notes    = st.text_area("Additional notes", value="", height=80)

    # Build profile dict
    dog_profile = {
        "name":                        dog_name,
        "breed":                       dog_breed,
        "age":                         dog_age,
        "gender":                      dog_gender,
        "weight":                      dog_weight,
        "neutered_spayed":             neutered,
        "origin":                      origin,
        "at_current_residence_since":  home_since,
        "home_type":                   home_type,
        "other_pets":                  [p.strip() for p in other_pets.split(",") if p.strip()],
        "family_members":              family,
        "medical_conditions":          [m.strip() for m in medical_raw.split(",") if m.strip()],
        "triggers":                    [t.strip() for t in triggers_raw.split(",") if t.strip()],
        "behavioral_conditions":       [b.strip() for b in behavioral_raw.split(",") if b.strip()],
        "behavioral_traits":           [t.strip() for t in traits_raw.split(",") if t.strip()],
        "notes":                       extra_notes,
    }

    st.divider()

    # ── Debug Mode ────────────────────────────────────────────────
    st.subheader("🔧 Debug Options")
    debug_mode   = st.checkbox("Show system prompt", value=False)
    show_metadata = st.checkbox("Show session metadata", value=True)

    st.divider()

    # ── Reset ─────────────────────────────────────────────────────
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.session_metadata     = SessionMetadata.new().to_dict()
        st.session_state.last_risk_tier        = "green"
        st.session_state.debug_system_prompt   = ""
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────
col_title, col_tier = st.columns([3, 1])
with col_title:
    st.title("CANIS Behavioral Intelligence Engine")
    st.caption(f"Testing profile: **{dog_profile['name']}** ({dog_profile['breed']}) — Plan: **{plan_tier.upper()}**")
with col_tier:
    tier_info = TIER_CONFIG.get(st.session_state.last_risk_tier, TIER_CONFIG["green"])
    st.markdown(
        f"""<div style='background:{tier_info["bg"]};padding:10px;border-radius:8px;text-align:center;margin-top:20px'>
        <strong>Last Response Tier</strong><br>{tier_info["label"]}
        </div>""",
        unsafe_allow_html=True
    )

st.divider()

# ── Session Metadata Panel ────────────────────────────────────────
if show_metadata:
    meta = st.session_state.session_metadata
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        color = "red" if meta.get("turns_without_progress", 0) >= 3 else "normal"
        st.metric("Turns w/o Progress", meta.get("turns_without_progress", 0))
    with mcol2:
        distress = meta.get("user_distress_level", "low")
        st.metric("User Distress", distress.upper())
    with mcol3:
        st.metric("Regression Reported", "YES ⚠️" if meta.get("regression_reported") else "No")
    with mcol4:
        st.metric("Pro Referral Given", "YES" if meta.get("professional_referral_given") else "No")

    st.divider()

# ── Conversation Display ──────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.conversation_history:
        st.info(f"👋 Start a conversation about **{dog_profile['name']}**'s behavior. "
                f"Try asking: *'Why does {dog_profile['name']} bark at other dogs on walks?'*")
    else:
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🐕"):
                    st.markdown(msg["content"])

# ── Chat Input ────────────────────────────────────────────────────
user_input = st.chat_input(f"Ask about {dog_profile['name']}'s behavior...")

if user_input:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call CANIS
    with st.chat_message("assistant", avatar="🐕"):
        with st.spinner("CANIS is thinking..."):
            result = get_canis_response(
                user_message=user_input,
                dog_profile=dog_profile,
                plan_tier=plan_tier,
                conversation_history=st.session_state.conversation_history,
                session_metadata=st.session_state.session_metadata,
                api_key=api_key,
                retriever=get_retriever(),
                debug=debug_mode,
            )

        # Display response
        st.markdown(result["response"])

        # Show risk tier badge
        tier = result["risk_tier"]
        tier_info = TIER_CONFIG.get(tier, TIER_CONFIG["green"])
        st.markdown(
            f"<small style='color:{tier_info['color']}'>{tier_info['label']}</small>",
            unsafe_allow_html=True
        )

    # Update state
    st.session_state.conversation_history = result["updated_history"]
    st.session_state.session_metadata     = result["updated_metadata"]
    st.session_state.last_risk_tier        = result["risk_tier"]

    if debug_mode and "system_prompt" in result:
        st.session_state.debug_system_prompt = result["system_prompt"]

    st.rerun()

# ── Debug: System Prompt ──────────────────────────────────────────
if debug_mode and st.session_state.debug_system_prompt:
    with st.expander("🔍 Assembled System Prompt (last request)", expanded=False):
        st.code(st.session_state.debug_system_prompt, language="text")

# ── Debug: Raw Session Metadata ───────────────────────────────────
if debug_mode:
    with st.expander("🔍 Raw Session Metadata", expanded=False):
        st.json(st.session_state.session_metadata)

    with st.expander("🔍 Raw Conversation History", expanded=False):
        st.json(st.session_state.conversation_history)
