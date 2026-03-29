"""
profile_formatter.py
Converts a dog profile dictionary from Django into a formatted
string block for injection into the CANIS system prompt.
"""

from typing import Optional


def format_profile(profile: dict) -> str:
    """
    Converts a dog profile dict (passed from Django backend) into
    a formatted string block for the system prompt.

    Args:
        profile: Dict containing dog profile fields.

    Returns:
        Formatted string to inject into system prompt.
    """

    # ── Basic Information ──────────────────────────────────────────
    name        = profile.get("name", "Unknown")
    breed       = profile.get("breed", "Unknown")
    age         = profile.get("age", "Unknown")
    gender      = profile.get("gender", "Unknown")
    weight      = profile.get("weight", "Unknown")
    neutered    = profile.get("neutered_spayed", False)
    neutered_str = "Yes" if neutered else "No"

    # ── Background & History ───────────────────────────────────────
    origin      = profile.get("origin", "Unknown")
    at_home_since = profile.get("at_current_residence_since", "Unknown")

    # ── Living Situation ───────────────────────────────────────────
    home_type   = profile.get("home_type", "Unknown")
    other_pets  = profile.get("other_pets", [])
    other_pets_str = ", ".join(other_pets) if other_pets else "None"
    family_members = profile.get("family_members", "Unknown")

    # ── Health & Sensitivities ─────────────────────────────────────
    medical_conditions = profile.get("medical_conditions", [])
    medical_str = ", ".join(medical_conditions) if medical_conditions else "None"
    triggers = profile.get("triggers", [])
    triggers_str = ", ".join(triggers) if triggers else "None"

    # ── Behavior & Personality ─────────────────────────────────────
    behavioral_conditions = profile.get("behavioral_conditions", [])
    behavioral_str = ", ".join(behavioral_conditions) if behavioral_conditions else "None"
    behavioral_traits = profile.get("behavioral_traits", [])
    traits_str = ", ".join(behavioral_traits) if behavioral_traits else "None"

    # ── Additional notes (optional) ────────────────────────────────
    notes = profile.get("notes", "")

    # ── Build the block ────────────────────────────────────────────
    block = f"""
--- ACTIVE DOG PROFILE ---
Name: {name}
Breed: {breed}
Age: {age} | Gender: {gender} | Weight: {weight} | Neutered/Spayed: {neutered_str}

Background:
  Origin: {origin}
  At current residence since: {at_home_since}

Living Situation:
  Home type: {home_type}
  Other pets: {other_pets_str}
  Family members: {family_members}

Health & Sensitivities:
  Medical conditions: {medical_str}
  Known triggers/sensitivities: {triggers_str}

Behavior & Personality:
  Behavioral conditions: {behavioral_str}
  Behavioral traits: {traits_str}
"""

    if notes:
        block += f"\nAdditional notes: {notes}\n"

    block += """
IMPORTANT INSTRUCTIONS FOR THIS PROFILE:
- Always address the dog by name in your responses.
- Factor in ALL profile fields when interpreting behavior and giving guidance.
- Medical conditions and triggers must always be considered before any recommendation.
- Behavioral conditions listed are owner-reported, not diagnoses — treat them as context.
- Tailor life-stage guidance to the dog's current age.
--- END OF PROFILE ---
"""
    return block.strip()


def get_life_stage(age_years) -> str:
    """
    Returns the life stage label based on age.
    Used internally by other modules for context.
    """
    try:
        age = float(str(age_years).replace("years", "").strip())
    except (ValueError, TypeError):
        return "unknown"

    if age < 0.5:
        return "puppy"
    elif age < 1.5:
        return "adolescent"
    elif age < 7:
        return "adult"
    else:
        return "senior"


# ── Example usage (for testing this module directly) ──────────────
if __name__ == "__main__":
    sample_profile = {
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

    print(format_profile(sample_profile))
    print("\nLife stage:", get_life_stage("6 years"))
