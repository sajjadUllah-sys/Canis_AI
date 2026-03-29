"""
chat.py

Core CANIS chat function.
This is the main entry point that the Django backend calls.

The function assembles the layered system prompt, calls OpenAI,
updates session metadata, and returns the response.
"""

import os
import re
import logging
from typing import Optional

from openai import OpenAI

from .master_prompt import CANIS_MASTER_PROMPT
from .profile_formatter import format_profile
from .session_tracker import SessionMetadata, SessionTracker, TIER_GREEN

log = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────
CHAT_MODEL  = "gpt-4o"
MAX_TOKENS  = 1500
TEMPERATURE = 0.4   # Low temp for consistent, safe behavioral guidance

# ── Max conversation history turns to keep (sliding window) ───────
MAX_HISTORY_TURNS = 10


def get_canis_response(
    user_message: str,
    dog_profile: dict,
    plan_tier: str,
    conversation_history: list[dict],
    session_metadata: dict,
    api_key: str,
    retriever=None,               # Optional: Retriever instance (rag/retriever.py)
    debug: bool = False,          # If True, return assembled system prompt too
) -> dict:
    """
    Main function called by the Django backend for every chat turn.

    Args:
        user_message:          The user's latest message
        dog_profile:           Dog profile dict from Django
        plan_tier:             "basic" or "premium"
        conversation_history:  List of {"role": ..., "content": ...} dicts
                               (full history, Django manages storage)
        session_metadata:      Dict from SessionMetadata.to_dict()
                               (Django stores and passes this each turn)
        api_key:               OpenAI API key
        retriever:             Optional Retriever instance for RAG
        debug:                 If True, include assembled system prompt in return

    Returns:
        {
            "response":          str  — CANIS's response to display
            "updated_history":   list — conversation history with new turn appended
            "updated_metadata":  dict — updated session metadata for Django to store
            "risk_tier":         str  — detected risk tier (green/yellow/orange/red)
            "system_prompt":     str  — (only if debug=True)
        }
    """

    client = OpenAI(api_key=api_key)

    # ── 1. Deserialize session metadata ───────────────────────────
    metadata = SessionMetadata.from_dict(session_metadata) if session_metadata else SessionMetadata.new()

    # ── 2. RAG: retrieve relevant knowledge chunks ─────────────────
    context_block = ""
    if retriever:
        try:
            chunks = retriever.retrieve(
                query=user_message,
                top_k=5,
                dog_profile=dog_profile
            )
            context_block = retriever.format_context_block(chunks)
        except Exception as e:
            log.error(f"RAG retrieval failed: {e}")
            context_block = ""

    # ── 3. Check for forced escalation ────────────────────────────
    escalation_note = SessionTracker.get_escalation_note(metadata)

    # ── 4. Assemble system prompt ──────────────────────────────────
    system_prompt = _build_system_prompt(
        dog_profile=dog_profile,
        plan_tier=plan_tier,
        context_block=context_block,
        escalation_note=escalation_note,
    )

    # ── 5. Build messages array ────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (sliding window)
    trimmed_history = _trim_history(conversation_history, MAX_HISTORY_TURNS)
    messages.extend(trimmed_history)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # ── 6. Call OpenAI ─────────────────────────────────────────────
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        assistant_response = completion.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"OpenAI API call failed: {e}")
        assistant_response = (
            "I'm having trouble processing your request right now. "
            "Please try again in a moment. If this is an emergency, "
            "please contact your veterinarian directly."
        )

    # ── 7. Detect risk tier from response ──────────────────────────
    detected_risk_tier = _detect_risk_tier(assistant_response)

    # ── 8. Update session metadata ─────────────────────────────────
    updated_metadata = SessionTracker.update(
        metadata=metadata,
        user_message=user_message,
        assistant_response=assistant_response,
        detected_risk_tier=detected_risk_tier,
    )

    # ── 9. Update conversation history ────────────────────────────
    updated_history = list(trimmed_history) + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": assistant_response},
    ]

    # ── 10. Build return payload ───────────────────────────────────
    result = {
        "response":         assistant_response,
        "updated_history":  updated_history,
        "updated_metadata": updated_metadata.to_dict(),
        "risk_tier":        detected_risk_tier,
    }

    if debug:
        result["system_prompt"] = system_prompt

    return result


# ══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════

def _build_system_prompt(
    dog_profile: dict,
    plan_tier: str,
    context_block: str,
    escalation_note: Optional[str],
) -> str:
    """Assemble the layered system prompt in correct order."""

    parts = [
        # Layer 0: CANIS master prompt (always first)
        CANIS_MASTER_PROMPT,

        # Layer 1: Dog profile
        "\n\n" + format_profile(dog_profile),

        # Layer 2: Plan tier declaration
        f"\n\nCURRENT USER PLAN: {plan_tier.upper()}",
        (
            "Since this is a BASIC plan user, provide concepts and general direction only. "
            "Do NOT provide step-by-step protocols, specific exercises, or detailed timelines."
            if plan_tier.lower() == "basic"
            else
            "Since this is a PREMIUM plan user, you may provide full phased protocols, "
            "specific exercises, timelines, and progress tracking."
        ),
    ]

    # Layer 3: RAG context (only if retrieved)
    if context_block:
        parts.append(f"\n\n{context_block}")

    # Layer 4: Escalation flag (only if triggered)
    if escalation_note:
        parts.append(f"\n\n{escalation_note}")

    return "\n".join(parts)


def _trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """
    Keep only the last N turns of conversation history.
    Each turn = 1 user message + 1 assistant message = 2 entries.
    """
    max_messages = max_turns * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def _detect_risk_tier(response: str) -> str:
    """
    Heuristically detect which risk tier CANIS responded with
    by scanning the response text.
    Used to update session metadata.
    """
    response_lower = response.lower()

    # Red tier signals
    red_signals = [
        "immediate veterinary", "emergency clinic", "contact your vet",
        "veterinarian now", "do not delay", "urgent", "emergency"
    ]
    if any(s in response_lower for s in red_signals):
        return "red"

    # Orange tier signals
    orange_signals = [
        "professional evaluation", "certified", "behaviorist",
        "in-person", "caab", "dacvb", "behavior consultant",
        "safety is the priority"
    ]
    if any(s in response_lower for s in orange_signals):
        return "orange"

    # Yellow tier signals
    yellow_signals = [
        "phase 1", "phase 2", "protocol", "desensitization",
        "counter-conditioning", "threshold", "management strategy"
    ]
    if any(s in response_lower for s in yellow_signals):
        return "yellow"

    return "green"


# ── Standalone test ───────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_profile = {
        "name": "Jenny",
        "breed": "Golden Retriever",
        "age": "6 years",
        "gender": "Female",
        "weight": "34kg",
        "neutered_spayed": True,
        "origin": "Shelter dog",
        "at_current_residence_since": "May 3rd, 2021",
        "home_type": "Apartment",
        "other_pets": ["Cat", "Parrot"],
        "family_members": "2 Adults",
        "medical_conditions": ["Anxiety", "Hip Dysplasia"],
        "triggers": ["Thunder", "Vacuum"],
        "behavioral_conditions": ["Reactivity", "Barking Issues"],
        "behavioral_traits": ["Focused", "Gentle"],
    }

    result = get_canis_response(
        user_message="Why does Jenny bark so much at other dogs when we're on walks?",
        dog_profile=test_profile,
        plan_tier="premium",
        conversation_history=[],
        session_metadata={},
        api_key=os.getenv("OPENAI_API_KEY"),
        retriever=None,
        debug=True,
    )

    print("RISK TIER:", result["risk_tier"])
    print("\nRESPONSE:\n", result["response"])
