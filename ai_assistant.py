"""SkillConnect AI: Volunteer Decision & Guidance Assistant + NGO Recruitment Advisor.

The language model does the conversation and reasoning. The application's own CSV data and matching engine remain the source of truth. This module does NOT duplicate dashboard pages (recommendations, My Applications, Saved);
it adds decision support on top of them:
Volunteer: suitability check, plain-language explanations, skill-term interpretation, application coaching, realistic-fit finder, better use of existing skills.
NGO: applicant pool patterns, missing skills, recruitment problems, opportunity description improvements, who needs attention.
"""
import json
import os
import re
import pandas as pd

# Small helpers

def _clean(value):
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def _safe_records(df, columns=None, limit=80):
    if df is None or df.empty:
        return []
    frame = df.copy()
    if columns:
        keep = [c for c in columns if c in frame.columns]
        frame = frame[keep]
    frame = frame.head(limit)
    return frame.fillna("").to_dict(orient="records")

def _split_skills(value):
    """Split a 'a, b; c | d' style skills string into a clean list."""
    text = _clean(value)
    if not text:
        return []
    parts = re.split(r"[,;|/\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def _skills_overlap(volunteer_skills, required_skills):
    """Loose case-insensitive overlap. Returns (matched, not_found)."""
    have = [s.lower() for s in volunteer_skills]
    matched, not_found = [], []
    for req in required_skills:
        r = req.lower()
        if any(r == h or r in h or h in r for h in have):
            matched.append(req)
        else:
            not_found.append(req)
    return matched, not_found

def _to_float(value):
    m = re.search(r"-?\d+(\.\d+)?", _clean(value))
    return float(m.group()) if m else None

def _truncate(text, n):
    text = _clean(text)
    return text if len(text) <= n else text[: n - 1] + "…"

def _profile_gaps(volunteer):
    """Fields that would weaken an application if left empty."""
    important = {
        "Bio": "bio",
        "Skills": "skills",
        "Experience": "experience",
        "Availability": "availability",
        "Preferred_Mode": "preferred mode (online/offline/hybrid)",
        "Location": "location",
        "Qualification": "qualification",
    }
    return [label for col, label in important.items() if not _clean(volunteer.get(col, ""))]

def _norm_id(value):
    """'14', 14 and '14.0' all become '14'."""
    t = _clean(value)
    return t[:-2] if t.endswith(".0") else t

def _ngo_name_map(ngos):
    """SrNo -> NGO name, so IDs can always be shown as real names."""
    names = {}
    if ngos is not None and not ngos.empty and {"SrNo", "Name"} <= set(ngos.columns):
        for _, r in ngos.iterrows():
            names[_norm_id(r.get("SrNo", ""))] = _clean(r.get("Name", ""))
    return names

def _opp_lookup(opportunities, ngo_names):
    """Opportunity_ID -> {Role_Title, NGO_Name}."""
    lookup = {}
    if opportunities is None or opportunities.empty or "Opportunity_ID" not in opportunities.columns:
        return lookup
    for _, r in opportunities.iterrows():
        lookup[_norm_id(r.get("Opportunity_ID", ""))] = {
            "Role_Title": _clean(r.get("Role_Title", "")) or "Opportunity",
            "NGO_Name": ngo_names.get(_norm_id(r.get("NGO_ID", "")), "Unknown NGO"),
        }
    return lookup

def _decorate(records, lookup):
    """Attach Role_Title / NGO_Name to any record that has an Opportunity_ID."""
    for rec in records:
        info = lookup.get(_norm_id(rec.get("Opportunity_ID", "")))
        if info:
            rec.update(info)
    return records

OPP_COLUMNS = [
    "Opportunity_ID", "NGO_ID", "Role_Title", "Description", "Skills_Required",
    "Area", "Location", "Mode (Online/Offline/Hybrid)", "Time_Commitment",
    "Status (Open/Closed)",
]


# Context builders

def _volunteer_context(context, volunteer, opportunities, applications, saved,
                       rank_opportunities, skill_lexicon, ngo_names):
    fields = [
        "Volunteer_ID", "Name", "Skills", "Interests", "Experience",
        "Qualification", "Bio", "Availability", "Preferred_Mode", "Location",
    ]
    context["volunteer"] = {
        k: _clean(volunteer.get(k, "")) for k in fields if _clean(volunteer.get(k, ""))
    }
    context["profile_gaps"] = _profile_gaps(volunteer)
    lookup = _opp_lookup(opportunities, ngo_names)

    open_ops = opportunities.copy()
    if "Status (Open/Closed)" in open_ops.columns:
        open_ops = open_ops[
            open_ops["Status (Open/Closed)"].astype(str).str.lower().isin(["open", ""])
        ]

    ranked = rank_opportunities(
        volunteer, open_ops, lexicon=skill_lexicon(None, opportunities),
    )

    # Match results for every open opportunity, so any opportunity the user asks about carries the engine's score/reasons/gaps.
    match_lookup = {}
    for row, result in ranked[:80]:
        match_lookup[str(row.get("Opportunity_ID", ""))] = {
            "score": result.get("overall"),
            "reasons": result.get("reasons", [])[:4],
            "skill_gaps": result.get("skill_gap", {}).get("missing", [])[:5],
        }
    context["match_lookup"] = match_lookup

    context["top_matches"] = [
        {
            "opportunity_id": str(row.get("Opportunity_ID", "")),
            "role_title": _clean(row.get("Role_Title", "Opportunity")),
            "ngo_name": ngo_names.get(_norm_id(row.get("NGO_ID", "")), "Unknown NGO"),
            "score": result.get("overall"),
            "reasons": result.get("reasons", [])[:4],
            "skill_gaps": result.get("skill_gap", {}).get("missing", [])[:5],
        }
        for row, result in ranked[:5]
    ]

    my_id = str(volunteer.get("Volunteer_ID", ""))
    if "Volunteer_ID" in applications.columns:
        context["applications"] = _decorate(_safe_records(
            applications[applications["Volunteer_ID"].astype(str) == my_id],
            ["Application_ID", "Opportunity_ID", "Application_Date", "Status", "Match_Score"],
        ), lookup)
    if "Volunteer_ID" in saved.columns:
        context["saved"] = _decorate(_safe_records(
            saved[saved["Volunteer_ID"].astype(str) == my_id], ["Opportunity_ID"],
        ), lookup)

    # Compact list of all opportunities (descriptions shortened); the one the user is focused on is expanded in full later by _focus_block().
    ops = _safe_records(opportunities, OPP_COLUMNS, limit=80)
    for op in ops:
        op["Description"] = _truncate(op.get("Description", ""), 220)
        # Show the NGO by name, never by its numeric ID.
        op["NGO_Name"] = ngo_names.get(_norm_id(op.pop("NGO_ID", "")), "Unknown NGO")
        mid = match_lookup.get(str(op.get("Opportunity_ID", "")))
        if mid:
            op["match_score"] = mid["score"]
    context["opportunities"] = ops

def _ngo_context(context, ngo, opportunities, applications, volunteers, ngo_names):
    fields = ["SrNo", "Name", "Area_of_Work", "Address", "Working_Since"]
    context["ngo"] = {k: _clean(ngo.get(k, "")) for k in fields if _clean(ngo.get(k, ""))}
    ngo_id = str(ngo.get("SrNo", ""))

    if "NGO_ID" in opportunities.columns:
        my_ops = opportunities[opportunities["NGO_ID"].astype(str) == ngo_id]
    else:
        my_ops = opportunities.iloc[0:0]

    context["opportunities"] = _safe_records(my_ops, OPP_COLUMNS)

    if not my_ops.empty and "Opportunity_ID" in applications.columns:
        my_apps = applications[
            applications["Opportunity_ID"].astype(str).isin(my_ops["Opportunity_ID"].astype(str))
        ]
    else:
        my_apps = applications.iloc[0:0]
    context["applications"] = _safe_records(
        my_apps,
        ["Application_ID", "Volunteer_ID", "Opportunity_ID", "Application_Date", "Status", "Match_Score"],
    )

    #Show applicants and roles by name, not by ID.
    vol_names = {}
    if volunteers is not None and not volunteers.empty and {"Volunteer_ID", "Name"} <= set(volunteers.columns):
        vol_names = {_norm_id(r["Volunteer_ID"]): _clean(r["Name"]) for _, r in volunteers.iterrows()}
    for rec in context["applications"]:
        rec["Applicant_Name"] = vol_names.get(_norm_id(rec.get("Volunteer_ID", "")), "Unknown applicant")
    _decorate(context["applications"], _opp_lookup(opportunities, ngo_names))

    context["recruitment_insights"] = _ngo_insights(my_ops, my_apps, volunteers)

def _ngo_insights(my_ops, my_apps, volunteers):
    """Pre-computed facts so the model interprets patterns instead of guessing."""
    insights = {"per_opportunity": [], "zero_applicant_opportunities": [],
                "description_flags": [], "common_applicant_skills": [],
                "required_skills_no_applicant_has": [], "applicants_needing_attention": []}
    if my_ops is None or my_ops.empty:
        return insights
    titles = {str(r.get("Opportunity_ID", "")): _clean(r.get("Role_Title", "")) for _, r in my_ops.iterrows()}

    #Per-opportunity applicant stats
    for _, op in my_ops.iterrows():
        oid = str(op.get("Opportunity_ID", ""))
        apps = my_apps[my_apps["Opportunity_ID"].astype(str) == oid] if not my_apps.empty else my_apps
        scores = [s for s in (_to_float(v) for v in apps.get("Match_Score", [])) if s is not None]
        status_counts = apps["Status"].astype(str).value_counts().to_dict() if "Status" in apps.columns and not apps.empty else {}
        entry = {
            "opportunity_id": oid,
            "role_title": _clean(op.get("Role_Title", "")),
            "applicants": int(len(apps)),
            "status_counts": status_counts,
            "avg_match_score": round(sum(scores) / len(scores), 1) if scores else None,
            "mode": _clean(op.get("Mode (Online/Offline/Hybrid)", "")),
            "location": _clean(op.get("Location", "")),
            "time_commitment": _clean(op.get("Time_Commitment", "")),
        }
        insights["per_opportunity"].append(entry)
        if len(apps) == 0 and _clean(op.get("Status (Open/Closed)", "")).lower() in {"open", ""}:
            insights["zero_applicant_opportunities"].append(entry["role_title"] or oid)

        # Description quality flags
        desc = _clean(op.get("Description", ""))
        flags = []
        if len(desc.split()) < 25:
            flags.append("description is very short")
        if not _clean(op.get("Skills_Required", "")):
            flags.append("no required skills listed")
        if not _clean(op.get("Time_Commitment", "")):
            flags.append("no time commitment stated")
        if flags:
            insights["description_flags"].append({"opportunity_id": oid, "role_title": entry["role_title"], "issues": flags})

    # Applicant skill patterns (needs the volunteers table)
    if volunteers is not None and not volunteers.empty and not my_apps.empty and "Volunteer_ID" in volunteers.columns:
        vol = volunteers.copy()
        vol["Volunteer_ID"] = vol["Volunteer_ID"].astype(str)
        vol_by_id = vol.set_index("Volunteer_ID")
        applicant_ids = my_apps["Volunteer_ID"].astype(str).unique()

        skill_counts = {}
        applicant_skill_sets = {}
        for vid in applicant_ids:
            if vid not in vol_by_id.index:
                continue
            skills = _split_skills(vol_by_id.loc[vid].get("Skills", ""))
            applicant_skill_sets[vid] = skills
            for s in skills:
                skill_counts[s.title()] = skill_counts.get(s.title(), 0) + 1
        insights["common_applicant_skills"] = sorted(skill_counts.items(), key=lambda kv: -kv[1])[:10]

        all_applicant_skills = [s for lst in applicant_skill_sets.values() for s in lst]
        missing = set()
        for _, op in my_ops.iterrows():
            _, not_found = _skills_overlap(all_applicant_skills, _split_skills(op.get("Skills_Required", "")))
            missing.update(s.title() for s in not_found)
        insights["required_skills_no_applicant_has"] = sorted(missing)[:10]

        # Strong applicants with incomplete profiles
        for _, app in my_apps.iterrows():
            vid = str(app.get("Volunteer_ID", ""))
            score = _to_float(app.get("Match_Score", ""))
            if vid not in vol_by_id.index or score is None or score < 60:
                continue
            row = vol_by_id.loc[vid]
            gaps = [lbl for col, lbl in (("Availability", "availability"), ("Bio", "bio"), ("Experience", "experience"))
                    if not _clean(row.get(col, ""))]
            if gaps:
                insights["applicants_needing_attention"].append({
                    "volunteer_id": vid,
                    "name": _clean(row.get("Name", "")),
                    "opportunity_id": str(app.get("Opportunity_ID", "")),
                    "role_title": titles.get(str(app.get("Opportunity_ID", "")), ""),
                    "match_score": score,
                    "missing_info": gaps,
                    "status": _clean(app.get("Status", "")),
                })
        insights["applicants_needing_attention"] = insights["applicants_needing_attention"][:10]

    return insights

def build_context(role, volunteer, ngo, opportunities, applications, saved, ngos,
                  rank_opportunities, skill_lexicon, volunteers=None):
    """Build a compact, factual context packet for the model.

    `volunteers` (the volunteers DataFrame) is optional but recommended: it
    lets the NGO assistant analyse applicant skills and profile completeness.
    """
    context = {
        "role": role,
        "volunteer": {},
        "ngo": {},
        "opportunities": [],
        "applications": [],
        "saved": [],
        "ngos": [],
        "top_matches": [],
    }

    ngo_names = _ngo_name_map(ngos)

    if role == "Volunteer" and volunteer is not None:
        _volunteer_context(context, volunteer, opportunities, applications, saved,
                           rank_opportunities, skill_lexicon, ngo_names)
    elif ngo is not None:
        _ngo_context(context, ngo, opportunities, applications, volunteers, ngo_names)

    # Directory context (no passwords or private fields)
    context["ngos"] = _safe_records(ngos, ["SrNo", "Name", "Area_of_Work", "Address"], limit=60)
    return context


# Focus opportunity (what is the user asking about?)
_FOCUS_INTENT = re.compile(
    r"\b(this|it|apply|applying|ready|eligible|qualified|qualify|explain|simple|simply|"
    r"requirement|requirements|gap|gaps|coach|help me)\b", re.I)

def _match_opportunity_in_text(text, ops):
    low = text.lower()
    for op in ops:
        oid = str(op.get("Opportunity_ID", "")).lower()
        title = _clean(op.get("Role_Title", "")).lower()
        if oid and re.search(r"\b" + re.escape(oid) + r"\b", low):
            return op
        if title and len(title) > 3 and title in low:
            return op
    return None


def _focus_block(user_text, history, context, opportunities_df):
    """Return a detailed block for the opportunity the user is talking about."""
    if context.get("role") != "Volunteer":
        return None
    ops = context.get("opportunities", [])
    if not ops:
        return None

    chosen, assumed = None, False
    sel = str(context.get("selected_opportunity_id") or "")
    if sel:
        chosen = next((o for o in ops if str(o.get("Opportunity_ID", "")) == sel), None)
    if chosen is None:
        chosen = _match_opportunity_in_text(user_text, ops)
    if chosen is None:
        for item in reversed(history or []):
            if item.get("role") == "user":
                chosen = _match_opportunity_in_text(item.get("content", ""), ops)
                if chosen:
                    break
    if chosen is None and _FOCUS_INTENT.search(user_text) and context.get("top_matches"):
        top_id = str(context["top_matches"][0]["opportunity_id"])
        chosen = next((o for o in ops if str(o.get("Opportunity_ID", "")) == top_id), None)
        assumed = chosen is not None
    if chosen is None:
        return None

    oid = str(chosen.get("Opportunity_ID", ""))
    # Full (untruncated) record
    full = dict(chosen)
    if opportunities_df is not None and "Opportunity_ID" in opportunities_df.columns:
        match_rows = opportunities_df[opportunities_df["Opportunity_ID"].astype(str) == oid]
        if not match_rows.empty:
            full = _safe_records(match_rows, OPP_COLUMNS, limit=1)[0]
            full["NGO_Name"] = chosen.get("NGO_Name", "Unknown NGO")
            full.pop("NGO_ID", None)

    vol = context.get("volunteer", {})
    matched, not_found = _skills_overlap(_split_skills(vol.get("Skills", "")),
                                         _split_skills(full.get("Skills_Required", "")))
    vol_mode = _clean(vol.get("Preferred_Mode", "")).lower()
    opp_mode = _clean(full.get("Mode (Online/Offline/Hybrid)", "")).lower()
    vol_loc = _clean(vol.get("Location", "")).lower()
    opp_loc = _clean(full.get("Location", "")).lower()

    logistics = []
    if vol_mode and opp_mode and opp_mode != "hybrid" and vol_mode != "hybrid" and vol_mode != opp_mode:
        logistics.append(f"mode mismatch: volunteer prefers {vol_mode}, role is {opp_mode}")
    if opp_mode in {"offline", "hybrid"} and vol_loc and opp_loc and vol_loc not in opp_loc and opp_loc not in vol_loc:
        logistics.append(f"location differs: volunteer in '{vol.get('Location')}', role in '{full.get('Location')}'")
    if not vol.get("Availability"):
        logistics.append("volunteer availability is not filled in")

    return {
        "assumed_from_top_match": assumed,
        "opportunity": full,
        "engine_match": context.get("match_lookup", {}).get(oid, {}),
        "skills_volunteer_has_for_this_role": matched,
        "required_skills_not_found_in_profile": not_found,
        "logistics_flags": logistics,
        "profile_gaps": context.get("profile_gaps", []),
    }



# Cheap local answers (only for short, direct lookups)
def _local_answer(user_text, context):
    """Answer pure status lookups without an API call.

    Deliberately strict: only short messages, so guidance questions such as
    'Help me apply, I already have 2 applications' still reach the model.
    """
    text = user_text.lower().strip()
    if len(text.split()) > 6:
        return None

    if context.get("role") == "Volunteer":
        if re.fullmatch(r"(show |list )?(my )?applications?( status)?\??", text):
            apps = context.get("applications", [])
            if not apps:
                return "You don't have any applications recorded yet."
            lines = ["Here are your current applications:"]
            for app in apps[:8]:
                title = app.get("Role_Title") or "an opportunity"
                org = f" at {app['NGO_Name']}" if app.get("NGO_Name") else ""
                lines.append(f"• {title}{org}: {app.get('Status', 'Unknown')}")
            lines.append("\nFor details, open **My Applications**. I can also help you prepare a stronger application.")
            return "\n".join(lines)

        if re.fullmatch(r"(show |list )?(my )?saved( opportunities)?\??", text):
            saved = context.get("saved", [])
            if not saved:
                return "You don't have any saved opportunities yet."
            items = [
                f"{x.get('Role_Title', 'Opportunity')}" + (f" ({x['NGO_Name']})" if x.get("NGO_Name") else "")
                for x in saved
            ]
            return ("You've saved: " + ", ".join(items)
                    + ".\nWant me to compare them or check which one you can realistically do?")
    return None


# System instructions
BASE_RULES = """
You are SkillConnect AI, a decision-support and guidance assistant inside a real
NGO/volunteer platform. You sit on top of the platform's matching engine.

GROUND RULES
- The supplied JSON is the source of truth. Never invent NGOs, opportunities, requirements, statuses, scores or user details. If something is missing, say it is missing.

- Match scores come from the platform's matching engine. Do not expose the internal score breakdown unless the user asks how matching works.

- Do NOT act as a search page or dashboard. Do not just list top matches, applications, or saved items; the website already shows those. Add value by interpreting, advising and explaining trade-offs.

- NEVER show internal IDs to the user (no "NGO #14", "Opportunity 29", "Volunteer_ID 44").Always refer to NGOs by their name (`NGO_Name`, or look the ID up in the `ngos` directory: SrNo -> Name),
to opportunities by their `Role_Title`, and to applicants by `Applicant_Name`/`name`.
If a name truly cannot be found, say "an NGO" / "this role" instead of printing an ID.

- Use plain, simple language. Many users are first-time volunteers, students or not familiar with NGO jargon. Explain any jargon you must use.

- Be inclusive without labelling the user. If someone mentions living far away, being online-only, having no degree, no NGO experience, or non-traditional experience (e.g. farming, household management, community roles), treat it
  as a normal constraint and work with it. Never be condescending.
  
- Be warm, concise and practical. Prefer short sections and a clear next step. Give a straight answer first (e.g. "Yes, you can apply"), then the reasoning.

- If a focus opportunity is provided and `assumed_from_top_match` is true, say which opportunity you assumed and invite the user to correct you.
"""

VOLUNTEER_RULES = """
YOU ARE TALKING TO A VOLUNTEER. Handle these five jobs well:

1. "CAN I ACTUALLY DO THIS?" (suitability).
    Compare profile vs the opportunity: skills, experience, qualification, location, preferred mode, availability, time commitment.
    Use `skills_volunteer_has_for_this_role`,`required_skills_not_found_in_profile` and `logistics_flags`.
    Say clearly whether they can apply, what they already match, what is missing, whether the gap is actually mandatory, and the one main thing to check.
    Encourage first-timers honestly; do not oversell.

2. EXPLAIN SIMPLY. When asked to explain an opportunity, give: "In simple terms", "What you'll actually do" (3-4 bullets), and "What you don't need".
    Translate NGO language into everyday words, using only what the opportunity data supports.

3. SKILL TERMS. When asked what a skill/requirement means (e.g. community mobilization), explain in one or two sentences with a concrete example.
   When the user says "I have X, does that count?", map their real experience to the requirement, even if the vocabulary differs.
   Be honest about whether it is a full, partial or weak match.

4. APPLICATION COACH. When asked for help applying: list the 1-3 profile fixes to make first (use `profile_gaps` and the bio/experience),
    their strengths for this role, potential gaps and how to address them honestly,
    then offer a short draft application message in the volunteer's own voice
    using only facts from their profile. Never fabricate experience.

5. REALISTIC FIT / USING SKILLS BETTER. For "what can I realistically do?"
   weigh location, mode, availability, time commitment, experience and skills across the opportunities, and explain the trade-off (e.g. strong skill match but weekday-only).
   For experienced volunteers, name their existing skills, the kinds of NGO roles that would use several of them together,
   and any under-used skill, so their skills are used well rather than wasted.
   If nothing fits, say so and suggest what would change that(e.g. completing availability, considering online roles).
"""

NGO_RULES = """
YOU ARE TALKING TO AN NGO. Act as a recruitment advisor, not a dashboard.
Use `recruitment_insights` (pre-computed facts) plus their opportunities and
applications.

Handle:
- "Why are we getting poor applicants?": compare required skills, description clarity, qualification, experience and time/mode against applicant skills and scores.
    Point to concrete causes (vague description, missing duties, unrealistic requirements).
    
- "What skills are we missing?": use common_applicant_skills and required_skills_no_applicant_has.

- "Why aren't people applying?": only use data-supported reasons (zero applicants, weekday/offline/location constraints, thin description, no time commitment).
    Compare with their other opportunities.
    
- "Who needs attention?": use applicants_needing_attention; suggest contacting strong-match applicants whose availability/bio is missing.
  Do not just restate rankings.
  
- "Improve my description": give specific rewrites (what the volunteer actually does, time, mode, what is NOT required) based on description_flags.
    Be honest when the data is too thin to support a conclusion. Never invent applicant details. Keep advice actionable: finish with 1-3 concrete next steps.
"""

def _build_prompt(user_text, history, context, opportunities_df=None):
    role_rules = VOLUNTEER_RULES if context.get("role") == "Volunteer" else NGO_RULES
    prompt_context = {k: v for k, v in context.items() if k not in {"match_lookup", "selected_opportunity_id"}}

    prompt = BASE_RULES + role_rules + "\n\nCURRENT APP DATA (JSON):\n"
    prompt += json.dumps(prompt_context, ensure_ascii=False, default=str)

    focus = _focus_block(user_text, history, context, opportunities_df)
    if focus:
        prompt += "\n\nFOCUS OPPORTUNITY (what the user is asking about):\n"
        prompt += json.dumps(focus, ensure_ascii=False, default=str)

    if history:
        prompt += "\n\nRECENT CONVERSATION:\n"
        for item in history[-8:]:
            if item.get("role") in {"user", "assistant"}:
                prompt += f"{item['role'].upper()}: {item.get('content', '')}\n"

    prompt += "\n\nUSER MESSAGE:\n" + user_text
    return prompt


# Model call

def ask_ai(user_text, history, context, opportunities_df=None):
    """Ask the Gemini model while grounding it in current app data."""
    local = _local_answer(user_text, context)
    if local:
        return local

    # Streamlit Cloud Secrets first, then environment variables.
    api_key = None
    secret_source = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            secret_source = "Streamlit Secrets"
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            secret_source = "environment variable"

    if isinstance(api_key, str):
        api_key = api_key.strip()

    if not api_key:
        return (
            "Gemini configuration error: `GEMINI_API_KEY` was not found.\n\n"
            "Add it to Streamlit Cloud → Settings → Secrets as:\n"
            '`GEMINI_API_KEY = "YOUR_KEY"`\n\n'
            "Then reboot/redeploy the app."
        )

    try:
        from google import genai
    except ImportError:
        return (
            "The Gemini AI package is not installed. "
            "Add `google-genai` to requirements.txt and redeploy."
        )

    prompt = _build_prompt(user_text, history, context, opportunities_df)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("SKILLCONNECT_AI_MODEL", "gemini-3.5-flash-lite"),
            contents=prompt,
            config={
                "max_output_tokens": 900,
                "temperature": 0.5,
            },
        )

        answer = getattr(response, "text", None)
        if answer:
            return answer.strip()

        return "I couldn't generate a response right now. Please try again."

    except Exception as e:
        error_text = str(e).lower()

        if "quota" in error_text or "429" in error_text or "resource_exhausted" in error_text:
            return (
                "The AI Assistant has temporarily reached its free Gemini "
                "usage limit. Please try again later."
            )

        if "api key" in error_text or "permission" in error_text or "unauthenticated" in error_text:
            return (
                "The Gemini API key could not be authenticated. "
                "Please check `GEMINI_API_KEY` in Streamlit Cloud → Settings → Secrets."
            )

        return (
            "The AI Assistant could not connect to the AI service right now. "
            "Please try again."
        )

# Streamlit page: multi-chat history with persistent CSV storage

# Persistent storage for the AI assistant's "My Chats".
# Session State is still used for the active UI, but every chat is also
# written to this CSV so it survives logout/re-login and browser sessions.
AI_CHATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_chats.csv")
AI_CHAT_COLUMNS = [
    "Chat_ID", "Owner_ID", "Role", "Title", "Created", "Updated", "Messages"
]

def _ensure_ai_chats_file():
    """Create the persistent chat CSV if it does not exist."""
    if not os.path.exists(AI_CHATS_FILE):
        pd.DataFrame(columns=AI_CHAT_COLUMNS).to_csv(AI_CHATS_FILE, index=False)

def _load_persistent_chats(owner):
    """Load all saved chats belonging to one logged-in owner."""
    _ensure_ai_chats_file()
    try:
        df = pd.read_csv(AI_CHATS_FILE, dtype=str).fillna("")
    except Exception:
        return {}

    chats = {}
    for _, row in df[df["Owner_ID"] == owner].iterrows():
        chat_id = _clean(row.get("Chat_ID", ""))
        if not chat_id:
            continue
        try:
            messages = json.loads(row.get("Messages", "[]") or "[]")
            if not isinstance(messages, list):
                messages = []
        except (TypeError, json.JSONDecodeError):
            messages = []

        created_text = _clean(row.get("Created", ""))
        updated_text = _clean(row.get("Updated", ""))
        try:
            created = pd.to_datetime(created_text).to_pydatetime()
        except Exception:
            from datetime import datetime
            created = datetime.now()
        try:
            updated = pd.to_datetime(updated_text).to_pydatetime()
        except Exception:
            updated = created

        chats[chat_id] = {
            "title": _clean(row.get("Title", "New chat")) or "New chat",
            "created": created,
            "updated": updated,
            "messages": messages,
        }
    return chats

def _save_persistent_chat(owner, role, chat_id, chat):
    """Insert or update one chat in the persistent CSV."""
    _ensure_ai_chats_file()
    try:
        df = pd.read_csv(AI_CHATS_FILE, dtype=str).fillna("")
    except Exception:
        df = pd.DataFrame(columns=AI_CHAT_COLUMNS)

    # Make sure an older/empty CSV still has the expected columns.
    for column in AI_CHAT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[AI_CHAT_COLUMNS]

    from datetime import datetime
    now = datetime.now()
    created = chat.get("created", now)
    if hasattr(created, "isoformat"):
        created_text = created.isoformat(timespec="seconds")
    else:
        created_text = str(created)

    row = {
        "Chat_ID": chat_id,
        "Owner_ID": owner,
        "Role": role,
        "Title": chat.get("title", "New chat"),
        "Created": created_text,
        "Updated": now.isoformat(timespec="seconds"),
        "Messages": json.dumps(chat.get("messages", []), ensure_ascii=False),
    }

    matches = df["Chat_ID"].astype(str).eq(str(chat_id)) & df["Owner_ID"].astype(str).eq(str(owner))
    if matches.any():
        df.loc[matches, AI_CHAT_COLUMNS] = [row[column] for column in AI_CHAT_COLUMNS]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(AI_CHATS_FILE, index=False)

def _delete_persistent_chat(owner, chat_id):
    """Delete one owner's chat from persistent storage."""
    _ensure_ai_chats_file()
    try:
        df = pd.read_csv(AI_CHATS_FILE, dtype=str).fillna("")
    except Exception:
        return
    if not {"Chat_ID", "Owner_ID"}.issubset(df.columns):
        return
    keep = ~((df["Chat_ID"].astype(str) == str(chat_id)) &
            (df["Owner_ID"].astype(str) == str(owner)))
    df.loc[keep].to_csv(AI_CHATS_FILE, index=False)


VOLUNTEER_SUGGESTIONS = [
    "Can I apply for this? I don't have much experience.",
    "Explain this opportunity in simple words.",
    "Help me apply: what should I fix first?",
    "I can only volunteer on weekends. What can I realistically do?",
    "How can I use my existing skills better?",
]

NGO_SUGGESTIONS = [
    "Why are we getting poor applicants?",
    "What skills are we missing in our applicant pool?",
    "Why aren't people applying to some of my roles?",
    "Which applicants need my attention?",
    "How can I improve my opportunity descriptions?",
]

AI_STATE_KEYS = [
    "ai_chats", "ai_current_chat_id", "ai_pending_prompt",
    "ai_focus_opportunity", "ai_owner", "ai_chat_messages",
]

def clear_ai_state(keep_page_tracker=False):
    """Wipe ALL AI chat state. Call this on logout."""
    import streamlit as st
    for key in AI_STATE_KEYS:
        st.session_state.pop(key, None)
    if not keep_page_tracker:
        st.session_state.pop("_ai_last_page", None)


def reset_ai_on_leave(current_page, ai_page_name="AI Assistant"):
    """Call once per run in app.py, BEFORE routing to pages.

    When the user leaves the AI page, the active conversation is closed, so
    coming back opens a fresh empty chat. Saved chats stay in the sidebar.
    """
    import streamlit as st
    if st.session_state.get("_ai_last_page") == ai_page_name and current_page != ai_page_name:
        st.session_state.pop("ai_current_chat_id", None)
        st.session_state.pop("ai_pending_prompt", None)
        st.session_state.pop("ai_focus_opportunity", None)
    st.session_state["_ai_last_page"] = current_page


def _make_title(text, limit=38):
    title = re.sub(r"\s+", " ", text).strip().rstrip("?.! ")
    if not title:
        return "New chat"
    title = title[0].upper() + title[1:]
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


def _owner_key(role, volunteer, ngo):
    if role == "Volunteer" and volunteer is not None:
        return f"Volunteer:{_clean(volunteer.get('Volunteer_ID', ''))}"
    if ngo is not None:
        return f"NGO:{_clean(ngo.get('SrNo', ''))}"
    return f"{role}:unknown"


def _render_chat_sidebar(st, chat_panel="page"):
    """Chat list with New chat / switch / delete.

    chat_panel="page"    -> shown on the AI page itself (works even if the app hides its sidebar)
    chat_panel="sidebar" -> shown in st.sidebar
    """
    from datetime import date
    chats = st.session_state.ai_chats
    current = st.session_state.get("ai_current_chat_id")

    if chat_panel == "sidebar":
        container = st.sidebar
    else:
        container = st.expander(f"💬 My chats ({len(chats)})", expanded=not current)

    with container:
        if chat_panel == "sidebar":
            st.markdown("### SkillConnect AI")
        if st.button("➕ New chat", key="ai_new_chat", use_container_width=True):
            st.session_state.pop("ai_current_chat_id", None)
            st.session_state.pop("ai_pending_prompt", None)
            st.rerun()

        if not chats:
            st.caption("Your chats will appear here.")
            return

        ordered = sorted(chats.items(), key=lambda kv: kv[1]["created"], reverse=True)
        today = date.today()
        groups = [
            ("Today", [kv for kv in ordered if kv[1]["created"].date() == today]),
            ("Previous", [kv for kv in ordered if kv[1]["created"].date() != today]),
        ]
        for label, items in groups:
            if not items:
                continue
            st.caption(label)
            for cid, chat in items:
                c1, c2 = st.columns([5, 1])
                if c1.button(chat["title"], key=f"ai_open_{cid}", use_container_width=True,
                             type="primary" if cid == current else "secondary"):
                    # Restore the selected conversation immediately into session state.
                    # This makes the active chat deterministic even after a fresh login.
                    st.session_state.ai_current_chat_id = cid
                    st.session_state.ai_chat_messages = list(chat.get("messages", []))
                    st.session_state.pop("ai_pending_prompt", None)
                    st.rerun()
                if c2.button("🗑", key=f"ai_del_{cid}", help="Delete this chat"):
                    del chats[cid]
                    _delete_persistent_chat(st.session_state.get("ai_owner", ""), cid)
                    if current == cid:
                        st.session_state.pop("ai_current_chat_id", None)
                    st.rerun()


def render_ai_assistant(role, volunteer, ngo, opportunities, applications, saved, ngos,
                        rank_opportunities, skill_lexicon, volunteers=None, chat_panel="page"):
    """Render the Streamlit chat page.

    Pass `volunteers=<volunteers DataFrame>` from app.py so NGO insights can
    analyse applicant skills and show applicant names.
    """
    import streamlit as st
    from datetime import datetime
    import uuid

    # Safety net: if a different person is now logged in, drop the old chats.
    owner = _owner_key(role, volunteer, ngo)
    if st.session_state.get("ai_owner") != owner:
        clear_ai_state(keep_page_tracker=True)
        st.session_state["ai_owner"] = owner
        # Restore this user's saved chats from persistent storage.
        st.session_state["ai_chats"] = _load_persistent_chats(owner)
    else:
        st.session_state.setdefault("ai_chats", _load_persistent_chats(owner))

    # Drop a stale pointer (e.g. chat was deleted).
    if st.session_state.get("ai_current_chat_id") not in st.session_state.ai_chats:
        st.session_state.pop("ai_current_chat_id", None)

    st.header("SkillConnect AI")
    if role == "Volunteer":
        st.caption("Your volunteering guide: check if you can realistically do a role, understand "
                   "requirements in simple words, and get help preparing your application.")
    else:
        st.caption("Your recruitment advisor: understand your applicant pool, spot skill gaps, "
                   "and improve your opportunities.")

    _render_chat_sidebar(st, chat_panel)

    context = build_context(
        role, volunteer, ngo, opportunities, applications, saved, ngos,
        rank_opportunities, skill_lexicon, volunteers=volunteers,
    )

    # Volunteers can pin the opportunity they're asking about.
    if role == "Volunteer" and context.get("opportunities"):
        options = {"(None, I'll mention it in my question)": ""}
        for op in context["opportunities"]:
            if _clean(op.get("Status (Open/Closed)", "")).lower() in {"open", ""}:
                label = f"{op.get('Role_Title', 'Opportunity')} — {op.get('NGO_Name', 'NGO')}"
                options[label] = str(op.get("Opportunity_ID", ""))
        choice = st.selectbox("Ask about a specific opportunity (optional)", list(options.keys()),
                              key="ai_focus_opportunity")
        context["selected_opportunity_id"] = options.get(choice, "")

    current_id = st.session_state.get("ai_current_chat_id")

    # Always derive the displayed conversation from the selected saved chat.
    # This prevents a stale/empty message list from showing the welcome screen
    # after the user opens a previous conversation.
    if current_id and current_id in st.session_state.ai_chats:
        messages = st.session_state.ai_chats[current_id].get("messages", [])
        st.session_state["ai_chat_messages"] = messages
    else:
        messages = []
        st.session_state["ai_chat_messages"] = []

    if not messages:
        suggestions = VOLUNTEER_SUGGESTIONS if role == "Volunteer" else NGO_SUGGESTIONS
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, key=f"ai_suggestion_{i}", use_container_width=True):
                st.session_state.ai_pending_prompt = suggestion

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state.pop("ai_pending_prompt", None)
    user_text = st.chat_input("Ask SkillConnect AI anything about volunteering...")
    if pending and not user_text:
        user_text = pending

    if user_text:
        # Start a new saved chat on the first message; title comes from it.
        if not current_id:
            current_id = uuid.uuid4().hex[:8]
            st.session_state.ai_chats[current_id] = {
                "title": _make_title(user_text),
                "created": datetime.now(),
                "updated": datetime.now(),
                "messages": [],
            }
            st.session_state.ai_current_chat_id = current_id
            # IMPORTANT: continue using the exact list stored inside the chat.
            # Otherwise the local `messages` list can diverge from the saved chat.
            messages = st.session_state.ai_chats[current_id]["messages"]
            st.session_state["ai_chat_messages"] = messages

        messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.spinner("SkillConnect AI is thinking..."):
                answer = ask_ai(user_text, messages[:-1], context, opportunities_df=opportunities)
            st.markdown(answer)
        messages.append({"role": "assistant", "content": answer})
        st.session_state.ai_chats[current_id]["updated"] = datetime.now()
        st.session_state["ai_chat_messages"] = messages

        # Persist the complete conversation so it survives logout/re-login.
        _save_persistent_chat(
            st.session_state.get("ai_owner", owner), role, current_id,
            st.session_state.ai_chats[current_id],
        )
        st.rerun()  # refresh sidebar so the new chat/title appears
