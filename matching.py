import math
import re
from difflib import SequenceMatcher
from functools import lru_cache
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_model():
    """Load the Sentence Transformer once and reuse it."""
    return SentenceTransformer(MODEL_NAME)

# text helpers
def clean_text(value):
    text = "" if value is None else str(value)
    if text.lower() in ("nan", "none", "nat"):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def normalise(text):
    text = re.sub(r"[^a-z0-9+#&./ -]", " ", clean_text(text).lower())
    return re.sub(r"\s+", " ", text).strip()

def split_items(value):
    """Split skills/interests into normalised items (commas, semicolons, pipes, line breaks)."""
    return [p for p in (normalise(x) for x in re.split(r"[,;|\n]+", clean_text(value))) if p]

def contains_phrase(text, phrase):
    text, phrase = normalise(text), normalise(phrase)
    return bool(text and phrase and phrase in text)

def _join(*values):
    return " ".join(t for t in map(clean_text, values) if t)

#skill lexicon
SKILL_ALIASES = {
    "teach": "teaching", "teaches": "teaching", "taught": "teaching",
    "mentored": "mentoring", "mentor": "mentoring",
    "coded": "coding", "programmed": "programming",
    "designing": "graphic design", "designer": "graphic design",
    "photograph": "photography", "photographed": "photography",
    "analyze": "data analysis", "analysing": "data analysis", "analyzing": "data analysis",
    # Practical equivalences used by the opportunity data.
    "website maintenance": "web development",
    "basic programming": "programming",
    "troubleshooting": "it support",
    "digital tools": "digital literacy",
    "excel/google sheets": "spreadsheets",
    "google sheets": "spreadsheets",
    "basic data analysis": "data analysis",
    "data analytics": "data analysis",
    "visual communication": "graphic design",
    "visual storytelling": "storytelling",
    "photo editing": "photo editing",
    "patient support": "healthcare",
    "health awareness": "healthcare",
    "mental health awareness": "mental health",
    "community engagement": "community outreach",
    "environmental awareness": "environment",
    "sustainability": "sustainability",
    "event planning": "event management",
    "event coordination": "event management",
    "volunteer coordination": "event management",
    "record keeping": "documentation",
    "legal documentation": "legal documentation",
    "financial administration": "finance",
    "bookkeeping": "accounting",
    "social media management": "social media",
}

DEFAULT_SKILLS = [
    "python", "java", "c", "c++", "sql", "machine learning", "artificial intelligence","data analysis", "data analytics", "data science", "excel", "google sheets","spreadsheet", "power bi", "tableau",
    "data entry", "web development","website maintenance", "programming", "coding", "it support", "digital tools","digital literacy", "graphic design", "canva", "adobe photoshop", "adobe illustrator",
    "branding", "photography", "photo editing", "video editing", "content writing","content creation", "social media", "communication", "marketing", "fundraising", "event management", "event coordination",
    "teaching", "mentoring", "tutoring", "research", "documentation", "legal documentation", "community outreach", "public speaking", "counselling", "counseling", "psychology", "healthcare", "first aid",
    "finance", "accounting", "project management", "operations", "environment", "sustainability", "waste management", "recycling",
]

def skill_lexicon(volunteers, opportunities):
    """Skill vocabulary from the CSV data plus a small general list."""
    terms = set(DEFAULT_SKILLS)
    for df, column in ((volunteers, "Skills"), (opportunities, "Skills_Required")):
        if df is not None and not df.empty:
            terms.update(i for v in df.get(column, []) for i in split_items(v) if len(i) >= 2)
    return sorted(terms, key=lambda x: (-len(x), x))

def extract_skills(text, lexicon=None):
    """Return the lexicon skills found in free text (longest phrase wins)."""
    text = normalise(text)
    if not text:
        return []
    lexicon = DEFAULT_SKILLS if lexicon is None else lexicon
    found = {s for s in lexicon if contains_phrase(text, s)}
    found |= {c for p, c in SKILL_ALIASES.items() if contains_phrase(text, p) and c in lexicon}
    final = []
    for skill in sorted(found, key=lambda x: (-len(x), x)):
        if not any(skill != other and skill in other for other in final):
            final.append(skill)
    return sorted(final)


#text used for semantic matching 
def volunteer_text(volunteer):
    keys = ("Skills", "Interests", "Experience", "Qualification", "Bio",
            "Availability", "Preferred_Mode", "Location")
    return _join(*(volunteer.get(k, "") for k in keys))


def opportunity_text(opportunity):
    keys = ("Role_Title", "Description", "Skills_Required")
    qualification = opportunity.get("Qualification (optional but useful)",
                                    opportunity.get("Qualification", ""))
    rest = ("Experience_Required", "Area", "Location", "Mode (Online/Offline/Hybrid)", "Time_Commitment")
    return _join(*(opportunity.get(k, "") for k in keys), qualification,
                 *(opportunity.get(k, "") for k in rest))


def _embedding_similarity(text_a, text_b):
    a, b = clean_text(text_a), clean_text(text_b)
    if not a or not b:
        return 0.0
    emb = get_model().encode([a, b], normalize_embeddings=True, show_progress_bar=False)
    return max(0.0, min(1.0, float(cosine_similarity([emb[0]], [emb[1]])[0][0])))


def _calibrate_semantic(score):
    """Sigmoid that separates weak, moderate and strong semantic matches."""
    score = max(0.0, min(1.0, float(score)))
    return 1.0 / (1.0 + math.exp(-8.0 * (score - 0.45)))


def _text_tokens(value):
    return {t for t in re.findall(r"[a-z0-9+#]+", normalise(value)) if len(t) > 2}


def _token_overlap(text_a, text_b):
    a, b = _text_tokens(text_a), _text_tokens(text_b)
    return len(a & b) / len(a | b) if a and b else 0.0


def interest_semantic_score(volunteer, opportunity):
    """Semantic + word-overlap compatibility of interests with the opportunity."""
    interest = clean_text(volunteer.get("Interests", ""))
    role = _join(opportunity.get("Area", ""), opportunity.get("Role_Title", ""),
                 opportunity.get("Description", ""))
    if not interest or not role:
        return 0.5, []
    score = _embedding_similarity(interest, role) * 0.75 + _token_overlap(interest, role) * 0.25
    reasons = []
    if score >= 0.72:
        reasons.append("Your interests strongly align with this cause")
    elif score >= 0.55:
        reasons.append("Your interests align with this opportunity")
    return max(0.0, min(1.0, score)), reasons


def profile_completeness(volunteer):
    fields = ("Skills", "Interests", "Experience", "Qualification", "Bio",
              "Availability", "Preferred_Mode", "Location")
    return sum(1 for f in fields if clean_text(volunteer.get(f, ""))) / len(fields)


#score components
def semantic_similarity(volunteer, opportunity):
    return _calibrate_semantic(_embedding_similarity(volunteer_text(volunteer), opportunity_text(opportunity)))

def location_mode_score(volunteer, opportunity):
    """Average of a location score and a mode score (each 0, 0.5 or 1), plus reasons."""
    v_loc = normalise(volunteer.get("Location", ""))
    o_loc = normalise(opportunity.get("Location", ""))
    v_mode = normalise(volunteer.get("Preferred_Mode", ""))
    o_mode = normalise(opportunity.get("Mode (Online/Offline/Hybrid)", ""))
    location = 0.0
    if v_loc and o_loc:
        if v_loc == o_loc or v_loc in o_loc or o_loc in v_loc:
            location = 1.0
        elif "mumbai" in v_loc and "mumbai" in o_loc:
            location = 0.5
    mode = 0.0
    if v_mode and o_mode:
        if v_mode == o_mode:
            mode = 1.0
        elif "hybrid" in (v_mode, o_mode):
            mode = 0.5
    reasons = []
    if location == 1.0:
        reasons.append("Location matches your preference")
    elif location == 0.5:
        reasons.append("Both are within Mumbai")
    if mode == 1.0:
        reasons.append("Mode matches your preference")
    elif mode == 0.5:
        reasons.append("Preferred mode is partly compatible")
    return (location + mode) / 2, reasons

def availability_score(volunteer, opportunity):
    """Compatibility of free-form availability with the time commitment (days + hours)."""
    v_avail = normalise(volunteer.get("Availability", ""))
    o_time = normalise(opportunity.get("Time_Commitment", ""))
    if not v_avail or not o_time:
        return 0.5
    weekend = {"weekend", "weekends", "saturday", "sunday"}
    weekdays = {"weekday", "weekdays", "monday", "tuesday", "wednesday", "thursday", "friday"}
    v_we, v_wd = (any(w in v_avail for w in s) for s in (weekend, weekdays))
    o_we, o_wd = (any(w in o_time for w in s) for s in (weekend, weekdays))
    day = 0.5
    if (o_we and v_we) or (o_wd and v_wd):
        day = 1.0
    elif (o_we and v_wd) or (o_wd and v_we):
        day = 0.0
    v_hours = re.findall(r"(?:up to|maximum|max|for)?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", v_avail)
    o_hours = re.findall(r"(?:about|around|approximately|up to|for)?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", o_time)
    hours = 0.5
    if v_hours and o_hours:
        vh, oh = max(map(float, v_hours)), max(map(float, o_hours))
        hours = 1.0 if vh >= oh else 0.6 if vh >= oh * 0.75 else 0.0
    return day * 0.6 + hours * 0.4

def _canonical_skill(skill):
    """Map equivalent skill wording to one comparable form."""
    s = normalise(skill)
    if not s:
        return ""
    # Longest phrases first so "website maintenance" is handled before "maintenance".
    for phrase, canonical in sorted(SKILL_ALIASES.items(), key=lambda x: -len(x[0])):
        if s == phrase:
            return canonical
    return s


def _required_skills(value):
    """Read the opportunity's comma-separated skill requirements as individual skills."""
    return [clean_text(x) for x in re.split(r"[,;|\n]+", clean_text(value)) if clean_text(x)]


def _skill_is_match(required, detected):
    """Match equivalent skill phrases without requiring identical wording."""
    r = _canonical_skill(required)
    d = _canonical_skill(detected)
    if not r or not d:
        return False
    if r == d or r in d or d in r:
        return True
    rt, dt = set(_text_tokens(r)), set(_text_tokens(d))
    if rt and dt and len(rt & dt) / len(rt | dt) >= 0.5:
        return True
    return SequenceMatcher(None, r, d).ratio() >= 0.78


def skill_gap_analysis(volunteer, opportunity, lexicon=None):
    """Required vs detected skills using exact, equivalent, and close wording matches."""
    required = _required_skills(opportunity.get("Skills_Required", ""))
    detected_text = _join(volunteer.get("Skills", ""), volunteer.get("Experience", ""),
                          volunteer.get("Bio", ""))
    detected = split_items(detected_text)

    matched = []
    matched_detected = set()
    for req in required:
        hit = next((det for det in detected
                    if det not in matched_detected and _skill_is_match(req, det)), None)
        if hit:
            matched.append(req)
            matched_detected.add(hit)

    return {
        "required": required,
        "detected": detected,
        "matched": matched,
        "missing": [s for s in required if s not in matched],
        "coverage": round((len(matched) / len(required) if required else 1.0) * 100, 1),
    }


def experience_score(volunteer, opportunity):
    required = normalise(opportunity.get("Experience_Required", ""))
    actual = normalise(volunteer.get("Experience", ""))
    if not required or required in {"not specified", "none", "na", "n/a"}:
        return 0.5
    if not actual:
        return 0.0
    if required in actual or actual in required:
        return 1.0
    for words in (("advanced", "expert", "senior"),
                  ("intermediate", "1-3 years", "2-3 years", "3 years"),
                  ("beginner", "basic", "fresher", "student", "entry")):
        if any(w in required for w in words) and any(w in actual for w in words):
            return 1.0
    return 0.5


def qualification_score(volunteer, opportunity):
    required = normalise(opportunity.get("Qualification (optional but useful)",
                                         opportunity.get("Qualification", "")))
    actual = normalise(volunteer.get("Qualification", ""))
    if not required or required in {"not specified", "optional", "none", "na", "n/a"}:
        return 0.5
    if not actual:
        return 0.0
    if required in actual or actual in required:
        return 1.0
    required_tokens = set(required.split())
    if not required_tokens:
        return 0.5
    overlap = len(required_tokens & set(actual.split())) / len(required_tokens)
    return 1.0 if overlap >= 0.5 else 0.5 if overlap > 0 else 0.0


# ---- overall match ----
WEIGHTS = {  # volunteer side: skills and interests drive the recommendation
    "semantic": 0.25, "skill": 0.35, "interest": 0.15, "experience": 0.08,
    "qualification": 0.05, "location_mode": 0.07, "availability": 0.05,
}
APPLICANT_WEIGHTS = {  # NGO side: still prioritise skills, experience and qualification
    "semantic": 0.30, "skill": 0.30, "interest": 0.10, "experience": 0.12,
    "qualification": 0.08, "location_mode": 0.05, "availability": 0.05,
}

def _weighted(values, weights):
    total = 0.0  # plain loop (not sum) so scores stay exactly the same on every Python version
    for key, weight in weights.items():
        total += values[key] * weight
    return total

def match_result(volunteer, opportunity, lexicon=None):
    """AI-assisted volunteer <-> opportunity match with score breakdown and reasons."""
    lexicon = DEFAULT_SKILLS if lexicon is None else lexicon
    gap = skill_gap_analysis(volunteer, opportunity, lexicon)
    required, matched = gap["required"], gap["matched"]
    interest, interest_reasons = interest_semantic_score(volunteer, opportunity)
    location_mode, location_reasons = location_mode_score(volunteer, opportunity)
    parts = {
        "semantic": semantic_similarity(volunteer, opportunity),
        "skill": len(matched) / len(required) if required else 0.5,
        "interest": interest,
        "experience": experience_score(volunteer, opportunity),
        "qualification": qualification_score(volunteer, opportunity),
        "location_mode": location_mode,
        "availability": availability_score(volunteer, opportunity),
    }
    overall = _weighted(parts, WEIGHTS)
    reasons = []
    if parts["semantic"] >= 0.78:
        reasons.append("Your profile strongly matches the role's overall requirements")
    elif parts["semantic"] >= 0.60:
        reasons.append("Your profile is semantically aligned with this role")
    if matched:
        reasons.append("Matching skills: " + ", ".join(s.title() for s in matched[:4]))
    reasons += interest_reasons + location_reasons
    if parts["experience"] >= 0.9:
        reasons.append("Your experience fits the stated experience requirement")
    if parts["qualification"] >= 0.9:
        reasons.append("Your qualification aligns with the opportunity")
    if parts["availability"] >= 0.9:
        reasons.append("Your stated availability fits the time commitment")
    scores = {k: round(v * 100, 1) for k, v in parts.items()}
    return {
        "overall": int(round(overall * 100)),
        **scores,
        "reasons": reasons[:6],
        "skill_gap": gap,
        "matched_skills": matched,
        "required_skills": required,
        "profile_completeness": round(profile_completeness(volunteer) * 100, 1),
        "rows": dict(scores),
    }

def applicant_match_result(volunteer, opportunity, lexicon=None):
    """NGO-side score: same breakdown, with the applicant weights."""
    base = match_result(volunteer, opportunity, lexicon=lexicon)
    values = {k: base[k] / 100 for k in APPLICANT_WEIGHTS}
    values["experience"] = experience_score(volunteer, opportunity)
    values["qualification"] = qualification_score(volunteer, opportunity)
    overall = _weighted(values, APPLICANT_WEIGHTS)
    return {**base, "overall": int(round(overall * 100))}

def rank_opportunities(volunteer, opportunities, lexicon=None):
    """(opportunity, result) pairs, best match first."""
    if opportunities is None or opportunities.empty:
        return []
    results = [(o, match_result(volunteer, o, lexicon=lexicon)) for _, o in opportunities.iterrows()]
    results.sort(key=lambda item: item[1]["overall"], reverse=True)
    return results

def rank_applicants(applicants, opportunity, lexicon=None):
    """(volunteer, application, result) tuples, best first. Items may be rows or (row, application) pairs."""
    if applicants is None or len(applicants) == 0:
        return []
    ranked = []
    for item in applicants:
        volunteer, application = item if isinstance(item, tuple) and len(item) == 2 else (item, None)
        ranked.append((volunteer, application, applicant_match_result(volunteer, opportunity, lexicon=lexicon)))
    ranked.sort(key=lambda item: item[2]["overall"], reverse=True)
    return ranked

#explainable AI display
def render_why_this_match(match, title=None):
    """Show the score breakdown and human-readable reasons in Streamlit."""
    import streamlit as st
    if title:
        st.subheader(title)
    st.progress(max(0.0, min(1.0, match["overall"] / 100)))
    labels = (("semantic", "Semantic similarity"), ("skill", "Skill"), ("interest", "Interest"),("experience", "Experience"), ("qualification", "Qualification"),("location_mode", "Location/mode"), ("availability", "Availability"))
    st.caption(" · ".join(f"{label} {match[key]}%" for key, label in labels if key in match))
    if match.get("reasons"):
        st.markdown("**Why this match?**")
        for reason in match["reasons"][:4]:
            st.write("✓ " + reason)
