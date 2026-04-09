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
- Consultation usage tracking (mock backend)
- Dynamic system prompt generation with UX rules
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

# ── Consultation limits ───────────────────────────────────────────
BASIC_LIMIT   = 5
PREMIUM_LIMIT = 50


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

    # ── Mock backend: consultation tracking ───────────────────────
    if "basic_consultations_used" not in st.session_state:
        st.session_state.basic_consultations_used = 0
    if "premium_consultations_used" not in st.session_state:
        st.session_state.premium_consultations_used = 0
    if "premium_addons" not in st.session_state:
        st.session_state.premium_addons = 0
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0

init_state()


# ══════════════════════════════════════════════════════════════════
# DYNAMIC SYSTEM PROMPT GENERATION
# ══════════════════════════════════════════════════════════════════

def build_dynamic_system_prompt(plan_tier: str, dog_profile: dict) -> str:
    """
    Constructs a highly specific system prompt based on:
    - Universal UX/tone rules (both tiers)
    - Tier-specific behavioral instructions
    - Interaction count (for upgrade injection)
    - Dog profile context
    """

    # ── Universal Rules (applied to both tiers) ───────────────────
    universal_rules = """
═══════════════════════════════════════════
CONVERSATIONAL UX RULES (ALWAYS APPLY)
═══════════════════════════════════════════

TONE & FORMAT:
- Never write like an article or blog post. You are having a real conversation.
- Avoid generic numbered lists (1, 2, 3...) unless you are explicitly outlining a
  justified Premium protocol with phases and durations.
- Reason with the user through guided thinking rather than presenting a menu of options.

PACING:
- Shift from "Here are several things you can do" to "Here's what might be happening,
  here's where to start, and let me understand more so we can adapt."
- Lead with interpretation and a focused starting point, then deepen as you learn more.

CONVERSATION FLOW:
- The conversation must NEVER end after an answer. ALWAYS conclude your response with
  a relevant, contextual follow-up question to keep the interaction alive and gather
  better context about the dog's specific situation.
- Your follow-up question should be specific to what was just discussed, not generic.
"""

    # ── Tier-Specific Rules ───────────────────────────────────────
    if plan_tier.lower() == "basic":
        tier_rules = """
═══════════════════════════════════════════
BASIC PLAN — BEHAVIORAL INSTRUCTIONS
═══════════════════════════════════════════

You are CANIS. Provide general guidance to understand the dog's behavior.
Explain possible underlying causes with reassurance and normalization.
Give 1 or 2 high-level suggestions.

DO NOT use lists.
DO NOT provide step-by-step plans, structured routines, timelines, or adaptive guidance.
Answer "Why is this happening?" but not fully "What exactly should I do?"

Your goal is to help the user understand the emotional and behavioral landscape,
not to hand them a protocol.
"""
        # Soft upgrade injection after first interaction
        if st.session_state.interaction_count >= 1:
            tier_rules += """
UPGRADE PROMPT INJECTION:
Before your final follow-up question, include a soft, conversational upgrade prompt.
Weave it naturally into your response — for example:
"Want a step-by-step plan tailored specifically for this?" or
"I can guide you through this with a structured plan if you'd like — that's available on Premium."
Do NOT make the upgrade prompt feel forced or salesy. It should feel like a natural offer to help more.
"""

    else:  # premium
        tier_rules = """
═══════════════════════════════════════════
PREMIUM PLAN — BEHAVIORAL INSTRUCTIONS
═══════════════════════════════════════════

You are CANIS. Your goal is deep behavioral interpretation, ongoing conversational
coaching, and delivering real results.

Do not default to rigid instructions immediately; first, ask targeted questions to
build a clear picture of the dog, environment, and situation.

Once context is gathered, provide structured, step-by-step protocols.
When presenting protocols, you MAY use clear, well-justified numbered lists
outlining phases, durations, and progress indicators.

Provide adaptive guidance — what to do if it works, and what to do if it doesn't.

Answer "What should I do step by step, and how do I adjust over time?"
"""

    # ── Dog Profile Context ───────────────────────────────────────
    profile_context = _format_profile_for_prompt(dog_profile)

    # ── Assemble ──────────────────────────────────────────────────
    return universal_rules + "\n" + tier_rules + "\n" + profile_context


def _format_profile_for_prompt(dog_profile: dict) -> str:
    """Format dog profile data into a prompt-injectable context block."""
    parts = ["\n--- KNOWN DOG CONTEXT (use this to personalize your response) ---"]

    name = dog_profile.get("name", "Unknown")
    parts.append(f"Dog Name: {name}")
    parts.append(f"Breed: {dog_profile.get('breed', 'Unknown')}")
    parts.append(f"Age: {dog_profile.get('age', 'Unknown')}")
    parts.append(f"Gender: {dog_profile.get('gender', 'Unknown')}")
    parts.append(f"Weight: {dog_profile.get('weight', 'Unknown')}")
    parts.append(f"Neutered/Spayed: {'Yes' if dog_profile.get('neutered_spayed') else 'No'}")
    parts.append(f"Origin: {dog_profile.get('origin', 'Unknown')}")
    parts.append(f"Home type: {dog_profile.get('home_type', 'Unknown')}")

    other_pets = dog_profile.get("other_pets", [])
    parts.append(f"Other pets: {', '.join(other_pets) if other_pets else 'None'}")
    parts.append(f"Family: {dog_profile.get('family_members', 'Unknown')}")

    medical = dog_profile.get("medical_conditions", [])
    parts.append(f"Medical conditions: {', '.join(medical) if medical else 'None'}")

    triggers = dog_profile.get("triggers", [])
    parts.append(f"Known triggers: {', '.join(triggers) if triggers else 'None'}")

    behavioral = dog_profile.get("behavioral_conditions", [])
    parts.append(f"Behavioral conditions: {', '.join(behavioral) if behavioral else 'None'}")

    traits = dog_profile.get("behavioral_traits", [])
    parts.append(f"Behavioral traits: {', '.join(traits) if traits else 'None'}")

    notes = dog_profile.get("notes", "")
    if notes:
        parts.append(f"Additional notes: {notes}")

    parts.append(f"Always refer to the dog as {name}.")
    parts.append("--- END KNOWN DOG CONTEXT ---")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# CONSULTATION LIMIT HELPERS
# ══════════════════════════════════════════════════════════════════

def is_chat_disabled(plan_tier: str) -> bool:
    """Returns True if the user has exhausted their consultation quota."""
    if plan_tier == "basic":
        return st.session_state.basic_consultations_used >= BASIC_LIMIT
    else:  # premium
        return (
            st.session_state.premium_consultations_used >= PREMIUM_LIMIT
            and st.session_state.premium_addons <= 0
        )


def consume_consultation(plan_tier: str):
    """Deduct one consultation from the user's quota."""
    if plan_tier == "basic":
        st.session_state.basic_consultations_used += 1
    else:  # premium
        if st.session_state.premium_consultations_used < PREMIUM_LIMIT:
            st.session_state.premium_consultations_used += 1
        elif st.session_state.premium_addons > 0:
            st.session_state.premium_addons -= 1


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

    # ── Consultation Usage Tracker ────────────────────────────────
    if plan_tier == "basic":
        used = st.session_state.basic_consultations_used
        st.markdown(f"**Consultations:** `{used}/{BASIC_LIMIT}` used")
        st.progress(min(used / BASIC_LIMIT, 1.0))

        if used >= BASIC_LIMIT:
            st.error(
                "🚫 Consultation limit reached. "
                "Please upgrade to Premium to continue."
            )

    else:  # premium
        used = st.session_state.premium_consultations_used
        addons = st.session_state.premium_addons
        total_available = PREMIUM_LIMIT + addons
        total_used = used + max(0, addons - st.session_state.premium_addons)  # simplify display

        st.markdown(f"**Consultations:** `{used}/{PREMIUM_LIMIT}` base used")
        st.progress(min(used / PREMIUM_LIMIT, 1.0))

        if addons > 0:
            st.info(f"➕ **Add-on credits remaining:** {addons}")
        elif used >= PREMIUM_LIMIT:
            st.warning(
                "⚠️ Base consultations exhausted. "
                "Purchase add-ons below to continue."
            )

        # ── Add-on Purchase (Simulated) ───────────────────────────
        st.markdown("**Add-ons (Simulate Purchase)**")
        addon_cols = st.columns(3)
        with addon_cols[0]:
            if st.button("+10", key="addon_10", use_container_width=True):
                st.session_state.premium_addons += 10
                st.rerun()
        with addon_cols[1]:
            if st.button("+25", key="addon_25", use_container_width=True):
                st.session_state.premium_addons += 25
                st.rerun()
        with addon_cols[2]:
            if st.button("+50", key="addon_50", use_container_width=True):
                st.session_state.premium_addons += 50
                st.rerun()

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
        st.session_state.interaction_count     = 0
        st.rerun()

    if st.button("🗑️ Reset All Counters", use_container_width=True):
        st.session_state.basic_consultations_used   = 0
        st.session_state.premium_consultations_used = 0
        st.session_state.premium_addons             = 0
        st.session_state.interaction_count          = 0
        st.session_state.conversation_history       = []
        st.session_state.session_metadata           = SessionMetadata.new().to_dict()
        st.session_state.last_risk_tier             = "green"
        st.session_state.debug_system_prompt        = ""
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
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
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
    with mcol5:
        st.metric("Interactions", st.session_state.interaction_count)

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
chat_disabled = is_chat_disabled(plan_tier)

if chat_disabled:
    if plan_tier == "basic":
        st.warning(
            "🔒 You've reached your Basic plan limit (5 consultations). "
            "Upgrade to Premium to unlock 50 consultations and structured coaching."
        )
    else:
        st.warning(
            "🔒 All Premium consultations and add-ons exhausted. "
            "Purchase add-ons from the sidebar to continue."
        )

user_input = st.chat_input(
    f"Ask about {dog_profile['name']}'s behavior...",
    disabled=chat_disabled,
)

if user_input:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()

    # ── Increment counters ────────────────────────────────────────
    st.session_state.interaction_count += 1
    consume_consultation(plan_tier)

    # ── Build the dynamic system prompt ───────────────────────────
    dynamic_prompt = build_dynamic_system_prompt(plan_tier, dog_profile)

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
                dynamic_system_prompt=dynamic_prompt,
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

    with st.expander("🔍 Mock Backend State", expanded=False):
        st.json({
            "basic_consultations_used":   st.session_state.basic_consultations_used,
            "premium_consultations_used":  st.session_state.premium_consultations_used,
            "premium_addons":             st.session_state.premium_addons,
            "interaction_count":          st.session_state.interaction_count,
        })
