"""
session_tracker.py

Manages per-conversation metadata that CANIS's Layer 5 requires.
Tracks escalation signals across turns so the AI can auto-trigger
professional referrals when thresholds are met.

Django backend is responsible for storing and passing this dict
forward on each API call.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import re


# ── Risk tier constants ───────────────────────────────────────────
TIER_GREEN  = "green"   # Educational
TIER_YELLOW = "yellow"  # Structured guidance
TIER_ORANGE = "orange"  # Professional referral
TIER_RED    = "red"     # Veterinary emergency

# ── Distress level constants ──────────────────────────────────────
DISTRESS_LOW    = "low"
DISTRESS_MEDIUM = "medium"
DISTRESS_HIGH   = "high"


@dataclass
class SessionMetadata:
    """
    Tracks escalation signals across conversation turns.
    Serializable to dict for Django storage.
    """
    # Progress tracking
    turns_without_progress: int   = 0
    total_turns: int              = 0

    # Risk signals
    regression_reported: bool     = False
    user_distress_level: str      = DISTRESS_LOW
    implementation_consistency: str = "unknown"  # high / partial / low / unknown

    # Referral tracking
    professional_referral_given: bool = False
    vet_referral_given: bool          = False
    last_risk_tier: str               = TIER_GREEN

    # Complexity tracking
    active_priority_levels: list  = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for Django storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMetadata":
        """Deserialize from Django-stored dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def new(cls) -> "SessionMetadata":
        """Create a fresh session (new conversation)."""
        return cls()


class SessionTracker:
    """
    Stateless utility class.
    Takes a SessionMetadata, analyzes the latest assistant response,
    and returns an updated SessionMetadata.

    Django calls this after each AI response and stores the result.
    """

    # Keywords indicating user distress
    HIGH_DISTRESS_SIGNALS = [
        "give up", "rehome", "re-home", "surrender", "terrible owner",
        "can't do this", "cannot do this", "hopeless", "nothing works",
        "failed", "failure", "i'm done", "i am done", "at my limit",
        "end of my rope", "exhausted", "desperate"
    ]
    MEDIUM_DISTRESS_SIGNALS = [
        "frustrated", "stressed", "worried", "concerned", "struggling",
        "difficult", "hard time", "not working", "tried everything"
    ]

    # Keywords indicating regression
    REGRESSION_SIGNALS = [
        "getting worse", "more aggressive", "worsening", "escalating",
        "more reactive", "increased barking", "new behavior", "never did this before",
        "worse than before", "regression", "regressed"
    ]

    # Keywords indicating progress
    PROGRESS_SIGNALS = [
        "better", "improving", "improved", "progress", "working",
        "calmer", "less reactive", "less barking", "doing well",
        "positive change", "responding well"
    ]

    @classmethod
    def update(
        cls,
        metadata: SessionMetadata,
        user_message: str,
        assistant_response: str,
        detected_risk_tier: str,
    ) -> SessionMetadata:
        """
        Analyze the latest turn and return updated metadata.

        Args:
            metadata:           Current session metadata
            user_message:       What the user just said
            assistant_response: What CANIS just responded
            detected_risk_tier: Risk tier detected for this turn

        Returns:
            Updated SessionMetadata
        """
        import copy
        updated = copy.deepcopy(metadata)
        updated.total_turns += 1
        updated.last_risk_tier = detected_risk_tier

        user_lower = user_message.lower()

        # ── Distress detection ────────────────────────────────────
        if any(signal in user_lower for signal in cls.HIGH_DISTRESS_SIGNALS):
            updated.user_distress_level = DISTRESS_HIGH
        elif any(signal in user_lower for signal in cls.MEDIUM_DISTRESS_SIGNALS):
            if updated.user_distress_level != DISTRESS_HIGH:
                updated.user_distress_level = DISTRESS_MEDIUM

        # ── Regression detection ──────────────────────────────────
        if any(signal in user_lower for signal in cls.REGRESSION_SIGNALS):
            updated.regression_reported = True

        # ── Progress tracking ─────────────────────────────────────
        if any(signal in user_lower for signal in cls.PROGRESS_SIGNALS):
            updated.turns_without_progress = 0
        else:
            # Only increment if we're in an active guidance situation
            if detected_risk_tier in [TIER_YELLOW, TIER_ORANGE]:
                updated.turns_without_progress += 1

        # ── Referral tracking ─────────────────────────────────────
        response_lower = assistant_response.lower()
        if any(phrase in response_lower for phrase in [
            "veterinarian", "emergency clinic", "vet immediately"
        ]):
            updated.vet_referral_given = True

        if any(phrase in response_lower for phrase in [
            "certified", "behaviorist", "professional evaluation",
            "behavior consultant", "caab", "dacvb"
        ]):
            updated.professional_referral_given = True

        return updated

    @classmethod
    def should_force_referral(cls, metadata: SessionMetadata) -> bool:
        """
        Returns True if escalation thresholds are met and CANIS
        should automatically recommend professional support.
        """
        if metadata.turns_without_progress >= 4:
            return True
        if metadata.user_distress_level == DISTRESS_HIGH:
            return True
        if metadata.regression_reported and metadata.turns_without_progress >= 2:
            return True
        return False

    @classmethod
    def get_escalation_note(cls, metadata: SessionMetadata) -> Optional[str]:
        """
        Returns an escalation instruction to append to the system prompt
        if thresholds are met. Returns None if no escalation needed.
        """
        if not cls.should_force_referral(metadata):
            return None

        reasons = []
        if metadata.turns_without_progress >= 4:
            reasons.append(f"{metadata.turns_without_progress} turns without reported progress")
        if metadata.user_distress_level == DISTRESS_HIGH:
            reasons.append("user is expressing high distress or hopelessness")
        if metadata.regression_reported:
            reasons.append("user has reported regression in behavior")

        return (
            f"\n--- ESCALATION FLAG ---\n"
            f"Based on conversation history: {'; '.join(reasons)}.\n"
            f"You MUST recommend professional in-person support in this response.\n"
            f"Follow the Layer 5.3 escalation protocol.\n"
            f"--- END ESCALATION FLAG ---"
        )


# ── Standalone test ───────────────────────────────────────────────
if __name__ == "__main__":
    meta = SessionMetadata.new()
    print("Initial:", meta.to_dict())

    meta = SessionTracker.update(
        meta,
        user_message="It's been 5 weeks and nothing is working. I'm starting to think I'm a terrible owner.",
        assistant_response="I understand how exhausted you are.",
        detected_risk_tier=TIER_YELLOW,
    )

    print("\nAfter turn:", meta.to_dict())
    print("Force referral:", SessionTracker.should_force_referral(meta))
    print("Escalation note:", SessionTracker.get_escalation_note(meta))
