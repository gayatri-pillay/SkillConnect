import pandas as pd
import streamlit as st
import os
import re
from ai_assistant import render_ai_assistant
from matching import (match_result, applicant_match_result, rank_opportunities, render_why_this_match, skill_lexicon, skill_gap_analysis)

# DATA LOCATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
DATA = DATA_FOLDER if os.path.isdir(DATA_FOLDER) else BASE_DIR
REQUIRED_CSVS = ("volunteer.csv", "ngo.csv", "opportunities.csv", "application.csv")

# PAGE SETTINGS
st.set_page_config(page_title="Skill Connect for Social Impact", page_icon="🤝", layout="wide")
with open(os.path.join(BASE_DIR, "style.css"), encoding="utf-8") as css_file:
    st.markdown("<style>" + css_file.read() + "</style>", unsafe_allow_html=True)

# HELPER FUNCTIONS
def save(data, filename):
    filepath = os.path.join(DATA, filename)
    try:
        data.to_csv(filepath, index=False)
        return True
    except PermissionError:
        st.error("Cannot save " + filename
            + ". Please close the CSV file if it is open in Excel or another program, then try again.")
        return False
    except Exception as e:
        st.error("Could not save " + filename + ": " + str(e))
        return False

def new_id(data, column):
    if len(data) == 0:
        return 1
    if column not in data.columns:
        return 1
    numbers = pd.to_numeric(data[column], errors="coerce")
    if numbers.dropna().empty:
        return 1
    return int(numbers.max()) + 1

def valid_phone(phone):
    """Accept only exactly 10 numeric digits."""
    return bool(re.fullmatch(r"\d{10}", str(phone).strip()))

def active_application_count(volunteer_id):
    if applications.empty:
        return 0
    my_apps = applications[applications["Volunteer_ID"].astype(str) == str(volunteer_id)]
    return len(my_apps[my_apps["Status"].astype(str).str.lower() != "withdrawn"])

# DEADLINE / STATUS / CAPACITY HELPERS
def today_date():
    """Today as a normalised timestamp, used for all deadline maths."""
    return pd.Timestamp.now().normalize()

def parse_date(value):
    """Return a normalised Timestamp, or None when the value is unusable."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ["nan", "nat", "none", "not specified"]:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()

def days_left(opportunity):
    """Days remaining until the deadline. None when there is no valid deadline."""
    deadline = parse_date(opportunity.get("Application_Deadline", ""))
    if deadline is None:
        return None
    return int((deadline - today_date()).days)

def deadline_passed(opportunity):
    remaining = days_left(opportunity)
    if remaining is None:
        return False
    return remaining < 0

def stored_status(opportunity):
    """The status as written in the CSV, normalised to Open/Closed/Cancelled."""
    raw = str(opportunity.get("Status (Open/Closed)", "Open")).strip().lower()
    if raw == "cancelled" or raw == "canceled":
        return "Cancelled"
    if raw == "closed":
        return "Closed"
    return "Open"

def accepted_count(opportunity_id):
    """How many volunteers have already been accepted for this opportunity."""
    if applications.empty:
        return 0
    rows = applications[(applications["Opportunity_ID"].astype(str) == str(opportunity_id)) & (
            applications["Status"].astype(str).str.lower() == "accepted")]
    return len(rows)

def volunteers_required(opportunity):
    try:
        required = int(float(str(opportunity.get("Volunteers_Required", 1))))
    except (ValueError, TypeError):
        required = 1
    return max(1, required)

def slots_left(opportunity):
    """Remaining volunteer slots for this opportunity (never negative)."""
    return max(0, volunteers_required(opportunity) - accepted_count(opportunity["Opportunity_ID"]))

def display_status(opportunity):
    """The status a user should actually see. Cancelled and Closed are stored states. Expired and Full are derived, because an opportunity can be stored as Open while no longer accepting applications."""
    status = stored_status(opportunity)
    if status != "Open":
        return status
    if deadline_passed(opportunity):
        return "Expired"
    if slots_left(opportunity) == 0:
        return "Full"
    return "Open"

def status_badge(status):
    icons = {"Open": "🟢 Open", "Closed": "⚪ Closed", "Cancelled": "🔴 Cancelled", "Expired": "🟠 Expired",
        "Full": "🔵 Full"}
    return icons.get(status, status)

def accepting_applications(opportunity):
    """Applications are only accepted while open, in date and not yet full."""
    return display_status(opportunity) == "Open"

def deadline_note(opportunity):
    """A short human message about the deadline, or an empty string."""
    remaining = days_left(opportunity)
    if remaining is None:
        return ""
    if remaining < 0:
        return "⛔ Deadline passed " + str(abs(remaining)) + " day(s) ago"
    if remaining == 0:
        return "⚠️ Deadline is today"
    if remaining <= 7:
        return "⏳ Deadline approaching — " + str(remaining) + " day(s) left"
    return "📅 " + str(remaining) + " day(s) left"

# DISPLAY-ONLY TEXT FORMATTING
# These helpers only affect how a value is drawn on screen. The stored CSV value is never modified, so the NGO directory data stays authoritative and the raw text remains available for matching.
# Optional short labels for entries that are too long to read as a tag. Keys are compared in lowercase.
SHORT_LABELS = {"alternative and augmentative communication": "AAC",
    "augmentative and alternative communication": "AAC", "online teaching program": "Online Teaching",
    "information and communication technology": "ICT", "water sanitation and hygiene": "WASH",
    "monitoring and evaluation": "M&E", "corporate social responsibility": "CSR",
    "non governmental organisation": "NGO", "persons with disabilities": "Disability Support"}

def pretty_label(item):
    """Tidy a single entry for display. Never changes the stored value."""
    text = str(item or "").strip()

    # Collapse runs of whitespace.
    text = re.sub(r"\s+", " ", text)

    # A hyphen used as a separator ("Autism- Mental Health") reads better as an ampersand. Hyphens inside a word ("Follow-up") are left alone.
    text = re.sub(r"\s*-\s+", " & ", text)

    # Trim stray punctuation left over from the source formatting.
    text = text.strip(" -–—·•:;,.")
    if not text:
        return ""

    # Substitute a short label when one is defined for this entry.
    shortened = SHORT_LABELS.get(text.lower())
    if shortened:
        return shortened

    # "Language And Communication" reads better as "Language & Communication".
    text = re.sub(r"\band\b", "&", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Re-case only when the source casing carries no information. A single all-caps word is treated as an acronym (AAC, WASH, ICT) and kept as is; multi-word all-caps text is shouting and gets title cased.
    if text.islower():
        text = text.title()
    elif text.isupper() and " " in text:
        text = text.title()
    return text

# Entries are separated by a comma, a semicolon, or a hyphen that has a space on BOTH sides ("Mental Health - Online Teaching").
# A hyphen attached to the preceding word ("Autism- Mental Health") joins one entry, and a hyphen inside a word ("Follow-up") is left alone.
ENTRY_SEPARATOR = re.compile(r"\s*[,;]\s*|\s+[-\u2013\u2014]\s+")

def split_list_field(value):
    """Split a directory field into clean display labels."""
    text = str(value or "").strip()
    if not text or text.lower() in ["nan", "none", "not specified", ""]:
        return []
    labels = []
    seen = set()
    for part in ENTRY_SEPARATOR.split(text):
        label = pretty_label(part)
        if not label:
            continue

        # Remove duplicates without being case-sensitive about it.
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels

def render_tags(value, heading=None, style="inline", empty_text="Not specified"):
    '''Draw a comma-separated field as readable tags. style="inline" gives a single wrapped row of tags. style="list" gives one labelled tag per line, for cards.'''
    labels = split_list_field(value)
    if heading:
        st.markdown("**" + heading + "**")
    if not labels:
        st.caption(empty_text)
        return labels
    if style == "list":
        for label in labels:
            st.markdown("🏷️ " + label)
    else:
        st.markdown(" · ".join("`" + label + "`" for label in labels))
    return labels

def tags_caption(value, limit=4):
    """A compact one-line summary of a list field, for dense card layouts."""
    labels = split_list_field(value)
    if not labels:
        return ""
    shown = labels[:limit]
    text = " · ".join(shown)
    if len(labels) > limit:
        text += " · +" + str(len(labels) - limit) + " more"
    return text

# SHARED CONSTANTS
VOLUNTEER_PAGES = ["Dashboard", "Find opportunities", "My applications", "Saved opportunities", "My profile",
    "AI Assistant"]

OPPORTUNITY_AREAS = ["Education & Mentoring", "Community Outreach", "Environment & Sustainability",
    "Health & Wellbeing", "Technology & Digital Support", "Graphic Design & Content",
    "Social Media & Communications", "Events & Volunteering", "Administration & Records",
    "Fundraising & Partnerships", "Photography & Storytelling", "Legal & Documentation"]

PROFILE_COMPLETION_FIELDS = ["Name", "Email", "Phone", "Location", "Skills", "Interests", "Availability",
    "Preferred_Mode", "Experience", "Qualification", "Bio"]

APPLICATION_STATUS_LABELS = {"pending": "🟡 Pending", "shortlisted": "🔵 Shortlisted", "accepted": "🟢 Accepted",
    "rejected": "🔴 Rejected", "withdrawn": "⚪ Withdrawn"}

def profile_completion(volunteer):
    """Percentage of profile fields the volunteer has filled in."""
    filled = 0
    for field in PROFILE_COMPLETION_FIELDS:
        value = volunteer.get(field, "")
        if pd.notna(value) and str(value).strip() != "":
            filled += 1
    return int((filled / len(PROFILE_COMPLETION_FIELDS)) * 100)

# APPLICANT MATCHING AI matching lives in matching.py (Sentence Transformer + cosine similarity + multi-factor ranking + "why this match" reasons).
def logout():
    st.session_state.logged_in = False
    st.session_state.role = ""
    st.session_state.id = ""

    # Reset navigation state so the next login always starts at the Dashboard.
    st.session_state.volunteer_page = "Dashboard"
    st.session_state.ngo_page = "Dashboard"
    st.session_state.pop("selected_opportunity", None)

    st.rerun()

# LOAD DATA
try:
    volunteers = pd.read_csv(os.path.join(DATA, "volunteer.csv"))
    ngos = pd.read_csv(os.path.join(DATA, "ngo.csv"))
    opportunities = pd.read_csv(os.path.join(DATA, "opportunities.csv"))
    applications = pd.read_csv(os.path.join(DATA, "application.csv"))
except FileNotFoundError as e:
    st.error("A required CSV file could not be found.")
    st.code(str(e))
    missing = [f for f in REQUIRED_CSVS if not os.path.isfile(os.path.join(DATA, f))]
    st.warning("Missing file(s): " + ", ".join(missing) + "\n\n"
        "Put these CSV files in the `data` folder beside `code.py` in your repository, "
        "then redeploy. The app also supports older copies with CSVs beside `code.py`. "
        "The app is currently looking in: " + DATA)
    st.stop()

# FIX APPLICATION CSV COLUMNS
if "Application_ID" not in applications.columns:
    applications["Application_ID"] = pd.Series(dtype="int64")

if "Volunteer_ID" not in applications.columns:
    applications["Volunteer_ID"] = pd.Series(dtype="object")

if "Opportunity_ID" not in applications.columns:
    applications["Opportunity_ID"] = pd.Series(dtype="object")

if "Application_Date" not in applications.columns:
    applications["Application_Date"] = ""

if "Status" not in applications.columns:
    applications["Status"] = "Pending"

if "Match_Score" not in applications.columns:
    applications["Match_Score"] = ""

# OPPORTUNITY SCHEMA : Every field the dashboard relies on is guaranteed to exist, so older opportunities.csv files keep working after this upgrade.
opportunity_columns = {"Role_Title": "", "Description": "", "Skills_Required": "", "Area": "", "Location": "",
    "Mode (Online/Offline/Hybrid)": "", "Time_Commitment": "", "Duration": "",
    "Qualification (optional but useful)": "", "Experience_Required": "", "Volunteers_Required": "1",
    "Application_Deadline": "", "Status (Open/Closed)": "Open"}

for column, default_value in opportunity_columns.items():
    if column not in opportunities.columns:
        opportunities[column] = default_value

# Older files may spell cancellation differently or leave status blank.
status_map = {"canceled": "Cancelled", "cancelled": "Cancelled", "closed": "Closed", "open": "Open", "": "Open",
    "nan": "Open"}

opportunities["Status (Open/Closed)"] = (opportunities["Status (Open/Closed)"].astype(str).str.strip().str.lower()
    .map(status_map).fillna("Open"))


# VOLUNTEER PROFILE SCHEMA
# Phase 2 fields are added safely so older volunteer.csv files still work.
profile_columns = {"Phone": "", "Preferred_Mode": "", "Bio": "", "Qualification": ""}
for column, default_value in profile_columns.items():
    if column not in volunteers.columns:
        volunteers[column] = default_value

# NGO SCHEMA
# Location is read when browsing and creating opportunities, so older ngo.csv files get an empty default.
ngo_columns = {"Location": ""}
for column, default_value in ngo_columns.items():
    if column not in ngos.columns:
        ngos[column] = default_value

# SAVED FILE
saved_file = os.path.join(DATA, "saved_opportunities.csv")
if os.path.exists(saved_file):
    try:
        saved = pd.read_csv(saved_file)
    except Exception:
        saved = pd.DataFrame(columns=["Volunteer_ID", "Opportunity_ID"])
else:
    saved = pd.DataFrame(columns=["Volunteer_ID", "Opportunity_ID"])


# SESSION
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = ""

if "id" not in st.session_state:
    st.session_state.id = ""

# LOGIN PAGE
def login_page():
    st.title("🤝 Skill Connect for Social Impact")
    st.write("### Find a cause. Share your skills. Make time count.")
    login, volunteer_register, ngo_register = st.tabs(["Sign in", "Join as a volunteer", "Register an NGO"])

    # LOGIN
    with login:
        role = st.radio("Sign in as", ["Volunteer", "NGO"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Sign in", use_container_width=True):
            if not email.strip() or not password.strip():
                st.error("Please enter both email and password.")
            elif role == "Volunteer":
                user = volunteers[(volunteers["Email"].astype(str).str.lower() == email.lower()) & (volunteers[
                            "Password"].astype(str) == password)]
                if len(user) > 0:
                    st.session_state.logged_in = True
                    st.session_state.role = "Volunteer"
                    st.session_state.id = (user.iloc[0]["Volunteer_ID"])

                    # Every new volunteer login starts at the Dashboard.
                    # Reloads while already logged in still preserve the current page.
                    st.session_state.volunteer_page = "Dashboard"
                    st.session_state.pop("selected_opportunity", None)

                    st.rerun()
                else:
                    st.error("Invalid email or password.")
            else:
                user = ngos[(ngos["Email_ID"].astype(str).str.lower() == email.lower()) & (ngos["Password"]
                        .astype(str) == password)]
                if len(user) > 0:
                    st.session_state.logged_in = True
                    st.session_state.role = "NGO"
                    st.session_state.id = (user.iloc[0]["SrNo"])

                    # Every new NGO login starts at the Dashboard.
                    # Reloads while already logged in preserve the current page.
                    st.session_state.ngo_page = "Dashboard"

                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    # VOLUNTEER REGISTRATION
    with volunteer_register:
        name = st.text_input("Full name", key="vname")
        email = st.text_input("Email", key="vemail")
        password = st.text_input("Password", type="password", key="vpass")
        skills = st.text_input("Skills", key="vskills")
        interests = st.text_input("Causes you care about", key="vinterests")
        availability = st.text_input("Availability", key="vavailability")
        experience = st.text_input("Experience", key="vexperience")
        qualification = st.text_input("Highest qualification", key="vqualification", placeholder="e.g. B.Ed, B.Sc, MA"
        )
        phone = st.text_input("Phone number", key="vphone", max_chars=10, placeholder="10 digits only")
        preferred_mode = st.selectbox("Preferred mode", ["Offline", "Hybrid", "Online"], key="vpreferred_mode")
        bio = st.text_area("Bio", key="vbio", placeholder="Tell NGOs a little about yourself...")
        location = st.text_input("Location", key="vlocation")
        if st.button("Create volunteer profile"):
            if not name or not email or not password:
                st.error("Name, email and password are required.")
            elif not skills.strip() or not location.strip():
                st.error("Skills and location are required so we can recommend opportunities.")
            elif not valid_phone(phone):
                st.error("Phone number must contain exactly 10 digits (numbers only).")
            elif not bio.strip():
                st.error("Please add a short bio.")
            elif email.lower() in volunteers["Email"].astype(str).str.lower().values:
                st.error("An account with this email already exists.")
            else:
                new_volunteer = {"Volunteer_ID": new_id(volunteers, "Volunteer_ID"), "Name": name, "Email": email,
                    "Password": password, "Skills": skills, "Interests": interests, "Availability": availability,
                    "Experience": experience, "Qualification": qualification, "Phone": phone.strip(),
                    "Preferred_Mode": preferred_mode, "Bio": bio.strip(), "Location": location}
                volunteers.loc[len(volunteers)] = new_volunteer
                if save(volunteers, "volunteer.csv"):
                    st.success("Profile created. Please sign in.")

    # NGO REGISTRATION
    with ngo_register:
        name = st.text_input("NGO name", key="nname")
        email = st.text_input("Organisation email", key="nemail")
        password = st.text_input("Password", type="password", key="npass")
        area = st.text_input("Area of work", key="narea")
        address = st.text_area("Address", key="naddress")
        since = st.text_input("Working since", key="nsince")
        if st.button("Create NGO account"):
            if not name or not email or not password:
                st.error("NGO name, email and password are required.")
            elif not area.strip() or not address.strip():
                st.error("Area of work and address are required.")
            elif email.lower() in ngos["Email_ID"].astype(str).str.lower().values:
                st.error("An account with this email already exists.")
            else:
                new_ngo = {"SrNo": new_id(ngos, "SrNo"), "Name": name, "Email_ID": email, "Password": password,
                    "Address": address, "Area_of_Work": area, "Working_Since": since}
                ngos.loc[len(ngos)] = new_ngo
                if save(ngos, "ngo.csv"):
                    st.success("NGO account created. Please sign in.")

# VOLUNTEER PAGES (one function per sidebar page)
def volunteer_home(volunteer):
    st.header("Welcome, " + str(volunteer["Name"]) + "!")
    open_opportunities = opportunities[opportunities.apply(accepting_applications, axis=1)
    ] if not opportunities.empty else opportunities
    my_saved = saved[saved["Volunteer_ID"].astype(str) == str(st.session_state.id)]
    col1, col2, col3 = st.columns(3)
    col1.metric("Open opportunities", len(open_opportunities))
    col2.metric("Active applications", active_application_count(st.session_state.id))
    col3.metric("Saved", len(my_saved))
    st.subheader("Your profile")
    completion = profile_completion(volunteer)
    st.progress(completion)
    st.caption("Profile completion: " + str(completion) + "%")
    st.write("**Skills:**", volunteer.get("Skills", ""))
    st.write("**Causes:**", volunteer.get("Interests", ""))
    st.write("**Availability:**", volunteer.get("Availability", ""))
    st.write("**Preferred mode:**", volunteer.get("Preferred_Mode", ""))
    st.write("**Location:**", volunteer.get("Location", ""))
    st.divider()
    st.subheader("Recommended for you")

    if open_opportunities.empty:
        st.info("There are no open opportunities to recommend right now. Check back soon.")
    else:
        with st.spinner("Finding the best matches for your profile..."):
            lexicon = skill_lexicon(volunteers, opportunities)
            ranked = rank_opportunities(volunteer, open_opportunities, lexicon=lexicon)
        top = ranked[:3]
        if not top:
            st.info("We could not build recommendations yet. Complete your profile and try again.")
        else:
            for opportunity, result in top:
                oid = opportunity["Opportunity_ID"]
                score = result["overall"]
                with st.container(border=True):
                    rec_left, rec_right = st.columns([4, 1])
                    with rec_left:
                        st.markdown("⭐ **" + str(opportunity.get("Role_Title", "Opportunity")) + "** — "
                            + str(score) + "% match")
                        ngo_rows = ngos[ngos["SrNo"].astype(str) == str(opportunity.get("NGO_ID", ""))]
                        ngo_name = (str(ngo_rows.iloc[0]["Name"]) if not ngo_rows.empty else "NGO")
                        st.caption(ngo_name)
                        for reason in (result.get("reasons") or [])[:3]:
                            st.write("✓ " + reason)
                        gap = result.get("skill_gap", {})
                        missing = gap.get("missing", [])
                        if missing:
                            st.caption("Potential skill gaps: "
                                + ", ".join(skill.title() for skill in missing[:4]))
                    with rec_right:
                        st.metric("Match", str(score) + "%")
                        if st.button("View role", key="rec_view_" + str(oid), use_container_width=True):
                            st.session_state.selected_opportunity = str(oid)
                            st.session_state.volunteer_page = "Find opportunities"
                            st.rerun()

    if st.button("Find opportunities →", use_container_width=True):
        st.session_state.volunteer_page = "Find opportunities"
        st.rerun()

def volunteer_opportunity_details(volunteer, selected_id):
    global saved
    detail_rows = opportunities[opportunities["Opportunity_ID"].astype(str) == str(selected_id)]
    if detail_rows.empty:
        st.error("Opportunity not found.")
        if st.button("← Back to opportunities"):
            del st.session_state.selected_opportunity
            st.rerun()
        return
    opportunity = detail_rows.iloc[0]
    lexicon = skill_lexicon(volunteers, opportunities)
    ngo_rows = ngos[ngos["SrNo"].astype(str) == str(opportunity["NGO_ID"])]
    ngo_name = (str(ngo_rows.iloc[0]["Name"]) if not ngo_rows.empty else "NGO")
    oid = opportunity["Opportunity_ID"]

    # A withdrawn application is inactive, so the volunteer can apply again.
    active_application = applications[(applications["Volunteer_ID"].astype(str) == str(st.session_state.id))
        & (applications["Opportunity_ID"].astype(str) == str(oid))
        & (applications["Status"].astype(str).str.lower() != "withdrawn")]
    already_applied = not active_application.empty
    already_saved = ((saved["Volunteer_ID"].astype(str) == str(st.session_state.id)) & (
            saved["Opportunity_ID"].astype(str) == str(oid))).any()
    if st.button("← Back to opportunities"):
        del st.session_state.selected_opportunity
        st.rerun()
    st.header(str(opportunity["Role_Title"]))
    st.caption("Posted by " + ngo_name)
    left, right = st.columns([2, 1])
    with left:
        st.subheader("About this opportunity")
        st.write(str(opportunity.get("Description", "")))
        st.subheader("Skills required")
        render_tags(opportunity.get("Skills_Required", ""), style="inline",
            empty_text="No specific skills listed.")

        # The posting NGO's directory areas, shown as tags.
        ngo_areas = (str(ngo_rows.iloc[0].get("Area_of_Work", "")) if not ngo_rows.empty else "")
        if split_list_field(ngo_areas):
            st.subheader("About " + ngo_name)
            render_tags(ngo_areas, heading="Areas of Work", style="inline")
        st.subheader("Requirements")
        qualification = opportunity.get("Qualification", opportunity.get(
                "Qualification (optional but useful)", "Not specified"))
        st.write("**Qualification:**", qualification)
        st.write("**Experience:**", opportunity.get("Experience_Required", "Not specified"))
    with right:
        st.subheader("Opportunity details")
        details = {"📍 Location": opportunity.get("Location", "Not specified"), "💻 Mode": opportunity.get(
                "Mode (Online/Offline/Hybrid)", "Not specified"), "🗓 Duration": opportunity.get("Duration",
                "Not specified"), "⏱ Time commitment": opportunity.get("Time_Commitment", "Not specified"),
            "👥 Volunteers required": opportunity.get("Volunteers_Required", "Not specified"),
            "📅 Deadline": opportunity.get("Application_Deadline", "Not specified")}
        for label, value in details.items():
            st.write("**" + label + ":**", value)
        st.write("**📌 Status:**", status_badge(display_status(opportunity)))
        st.write("**🪑 Slots left:**", slots_left(opportunity))
        note = deadline_note(opportunity)
        if note:
            remaining = days_left(opportunity)
            if remaining is not None and remaining < 0:
                st.error(note)
            elif remaining is not None and remaining <= 7:
                st.warning(note)
            else:
                st.caption(note)
        st.divider()
        match = match_result(volunteer, opportunity, lexicon=lexicon)
        render_why_this_match(match, title=str(match["overall"]) + "% match")

        # AI SKILL-GAP ANALYSIS
        gap = skill_gap_analysis(volunteer, opportunity, lexicon=lexicon)
        st.subheader("AI skill-gap analysis")
        if gap["required"]:
            if gap["matched"]:
                st.write("**Matched skills:** " + ", ".join(skill.title() for skill in gap["matched"]))
            if gap["missing"]:
                st.warning("**Potential skill gaps:** " + ", ".join(skill.title() for skill in gap["missing"])
                )
                st.caption(
                    "These are skills listed in the opportunity that were not detected in your current profile. "
                    "Add them only if you genuinely have the skill or relevant experience.")
            else:
                st.success("Your profile covers all skills detected as required for this opportunity.")
        else:
            st.caption("No specific skills were listed for this opportunity.")
    st.divider()
    action1, action2 = st.columns(2)

    # SAVE / REMOVE SAVED
    if already_saved:
        if action1.button("Remove from saved", key="detail_remove_" + str(oid), use_container_width=True):
            saved = saved[~((saved["Volunteer_ID"].astype(str) == str(st.session_state.id)) & (
                        saved["Opportunity_ID"].astype(str) == str(oid)))]
            if save(saved, "saved_opportunities.csv"):
                st.rerun()
    else:
        if action1.button("♡ Save opportunity", key="detail_save_" + str(oid), use_container_width=True):
            saved.loc[len(saved)] = {"Volunteer_ID": st.session_state.id, "Opportunity_ID": oid}
            if save(saved, "saved_opportunities.csv"):
                st.rerun()

    # APPLY
    current_display_status = display_status(opportunity)
    if current_display_status == "Cancelled":
        action2.button("Opportunity cancelled", disabled=True, key="detail_cancelled_" + str(oid),
            use_container_width=True)
    elif current_display_status == "Expired":
        action2.button("Deadline has passed", disabled=True, key="detail_expired_" + str(oid),
            use_container_width=True)
    elif current_display_status == "Full":
        action2.button("All volunteer slots filled", disabled=True, key="detail_full_" + str(oid),
            use_container_width=True)
    elif current_display_status == "Closed":
        action2.button("Applications closed", disabled=True, key="detail_closed_" + str(oid),
            use_container_width=True)
    elif already_applied:
        action2.button("Already applied", disabled=True, key="detail_applied_" + str(oid),
            use_container_width=True)
    elif active_application_count(st.session_state.id) >= 3:
        action2.button("3 active applications reached", disabled=True, key="detail_limit_" + str(oid),
            use_container_width=True)
    elif action2.button("Apply now", key="detail_apply_" + str(oid), use_container_width=True):
        match = match_result(volunteer, opportunity, lexicon=lexicon)
        applications.loc[len(applications)] = {"Application_ID": new_id(applications, "Application_ID"),
            "Volunteer_ID": st.session_state.id, "Opportunity_ID": oid,
            "Application_Date": pd.Timestamp.now().strftime("%Y-%m-%d"), "Status": "Pending",
            "Match_Score": match["overall"]}
        if save(applications, "application.csv"):
            st.success("Application sent!")
            st.rerun()

def volunteer_browse_opportunities(volunteer):
    st.header("Find opportunities")

    # Only opportunities still accepting applications are discoverable. Cancelled, closed,
    # expired and fully-staffed roles drop out here.
    if opportunities.empty:
        open_roles = opportunities.copy()
    else:
        open_roles = opportunities[opportunities.apply(accepting_applications, axis=1)].copy()

    # Add NGO information to each opportunity for filtering/display.
    ngo_lookup = ngos.set_index(ngos["SrNo"].astype(str))
    def get_ngo_name(ngo_id):
        key = str(ngo_id)
        if key in ngo_lookup.index:
            return str(ngo_lookup.loc[key, "Name"])
        return "NGO"
    open_roles["NGO_Name"] = open_roles["NGO_ID"].apply(get_ngo_name)

    # Opportunity locations come from the NGO's clean locality field.
    open_roles["Location"] = open_roles["NGO_ID"].astype(str).map(
        ngos.set_index(ngos["SrNo"].astype(str))["Location"].to_dict()
    ).fillna(open_roles["Location"]).astype(str).str.strip()
    areas = sorted([x for x in open_roles["Area"].dropna().astype(str).unique() if x.strip()],
        key=lambda x: x.lower())
    locations = sorted([x for x in open_roles["Location"].dropna().astype(str).unique() if x.strip()],
        key=lambda x: x.lower())
    ngo_names = sorted([x for x in open_roles["NGO_Name"].dropna().astype(str).unique() if x.strip()],
        key=lambda x: x.lower())
    filter1, filter2, filter3 = st.columns(3)
    with filter1:
        area_filter = st.selectbox("🌱 Opportunity area", ["All"] + areas)
    with filter2:
        ngo_filter = st.selectbox("🏢 NGO", ["All"] + ngo_names)
    with filter3:
        location_filter = st.selectbox("📍 Location", ["All"] + locations)
    filter4, filter5, filter6 = st.columns(3)
    with filter4:
        mode_filter = st.selectbox("💻 Mode", ["All", "Offline", "Hybrid", "Online"])
    with filter5:
        deadline_filter = st.selectbox("📅 Deadline", ["All", "Next 7 days", "Next 30 days"])
    with filter6:
        sort_by = st.selectbox("Sort by", ["Best match", "Newest", "Deadline"])
    roles = open_roles.copy()
    if area_filter != "All":
        roles = roles[roles["Area"].astype(str).str.lower() == area_filter.lower()]
    if ngo_filter != "All":
        roles = roles[roles["NGO_Name"].astype(str).str.lower() == ngo_filter.lower()]
    if location_filter != "All":
        roles = roles[roles["Location"].astype(str).str.lower() == location_filter.lower()]
    if mode_filter != "All":
        roles = roles[roles["Mode (Online/Offline/Hybrid)"].astype(str).str.lower() == mode_filter.lower()]

    # Deadline filtering and sorting
    roles["Deadline_Date"] = pd.to_datetime(roles["Application_Deadline"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    if deadline_filter == "Next 7 days":
        roles = roles[(roles["Deadline_Date"] >= today) & (roles["Deadline_Date"]
                <= today + pd.Timedelta(days=7))]
    elif deadline_filter == "Next 30 days":
        roles = roles[(roles["Deadline_Date"] >= today) & (roles["Deadline_Date"]
                <= today + pd.Timedelta(days=30))]
    if sort_by == "Deadline":
        roles = roles.sort_values(by="Deadline_Date", ascending=True, na_position="last")
    elif sort_by == "Newest":
        roles = roles.sort_values(by="Opportunity_ID", ascending=False)
    lexicon = skill_lexicon(volunteers, opportunities)
    ranked_roles = []
    if not roles.empty:
        with st.spinner("Ranking opportunities with AI matching..."):
            ranked_roles = rank_opportunities(volunteer, roles,
                lexicon=lexicon)
        if sort_by != "Best match":
            order = {str(row["Opportunity_ID"]): index for index, (_, row) in enumerate(roles.iterrows())}
            ranked_roles.sort(key=lambda item: order.get(str(item[0]["Opportunity_ID"]), 10 ** 6))
    st.caption(str(len(ranked_roles)) + " opportunities found")
    if not ranked_roles:
        st.info("No opportunities match your current filters.")
    for opportunity, result in ranked_roles:
        oid = opportunity["Opportunity_ID"]
        score = result["overall"]
        with st.container(border=True):
            top_left, top_right = st.columns([4, 1])
            with top_left:
                st.subheader(str(opportunity["Role_Title"]))
                st.caption(str(opportunity["NGO_Name"]))
            with top_right:
                st.metric("Match", str(score) + "%")
                st.write("💻 " + str(opportunity.get("Mode (Online/Offline/Hybrid)", "")))
            description = str(opportunity.get("Description", ""))
            if len(description) > 180:
                description = description[:180] + "..."
            st.write(description)
            reasons = (result.get("reasons") or [])[:2]
            if reasons:
                st.caption(" · ".join("✓ " + reason for reason in reasons))
            info1, info2, info3 = st.columns(3)
            info1.caption("📍 " + str(opportunity.get("Location", "Not specified")))
            info2.caption(deadline_note(opportunity) or ("📅 " + str(opportunity.get("Application_Deadline",
                            "No deadline"))))
            info3.caption("🧩 " + (tags_caption(opportunity.get("Skills_Required", ""), limit=3)
                    or "No skills listed"))
            if st.button("View details →", key="view_" + str(oid)):
                st.session_state.selected_opportunity = str(oid)
                st.rerun()

def volunteer_applications(volunteer):
    st.header("My applications")
    my_apps = applications[(applications["Volunteer_ID"].astype(str) == str(st.session_state.id))
        & (applications["Status"].astype(str).str.strip().str.lower() != "withdrawn")].copy()
    active_count = active_application_count(st.session_state.id)
    st.caption("Active applications: " + str(active_count) + " / 3")
    if my_apps.empty:
        st.info("You have not applied for any opportunities yet.")
    else:
        rows = []
        for index, application in my_apps.iterrows():
            role_rows = opportunities[opportunities["Opportunity_ID"].astype(str)
                == str(application["Opportunity_ID"])]
            if role_rows.empty:
                continue
            role = role_rows.iloc[0]
            ngo_rows = ngos[ngos["SrNo"].astype(str) == str(role["NGO_ID"])]
            ngo_name = (str(ngo_rows.iloc[0]["Name"]) if not ngo_rows.empty else "NGO")
            rows.append({"Index": index, "Application_ID": str(application["Application_ID"]),
                "Opportunity": str(role["Role_Title"]), "NGO": ngo_name, "Applied On": str(
                    application["Application_Date"]), "Status": str(application["Status"]),
                "Match": application.get("Match_Score", "")})
        for row in rows:
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1.5, 2])
                c1.write("**" + row["Opportunity"] + "**")
                if str(row.get("Match", "")).strip() not in ["", "nan"]:
                    c1.caption(str(row["Match"]) + "% match when you applied")
                c2.write(row["NGO"])
                c3.write(row["Applied On"])
                status = row["Status"]
                status_lower = status.lower()
                display_label = APPLICATION_STATUS_LABELS.get(status_lower, status)
                c4.write(display_label)
                if status_lower in ["pending", "shortlisted"]:
                    if c5.button("Withdraw", key="withdraw_" + row["Application_ID"]):
                        applications.loc[row["Index"], "Status"] = "Withdrawn"
                        if save(applications, "application.csv"):
                            st.rerun()
                else:
                    c5.caption("No action")

def volunteer_saved(volunteer):
    global saved
    st.header("Saved opportunities")
    my_saved = saved[saved["Volunteer_ID"].astype(str) == str(st.session_state.id)]
    ids = my_saved["Opportunity_ID"].astype(str)
    roles = opportunities[opportunities["Opportunity_ID"].astype(str).isin(ids)]
    if roles.empty:
        st.info("You have not saved any opportunities yet.")
    for _, opportunity in roles.iterrows():
        with st.container(border=True):
            st.subheader(str(opportunity["Role_Title"]))
            st.write(str(opportunity.get("Description", "")))
            if not accepting_applications(opportunity):
                st.warning("This opportunity is no longer accepting applications.")
            save_col1, save_col2 = st.columns(2)
            if save_col1.button("View details", key="saved_view_" + str(opportunity["Opportunity_ID"]),
                use_container_width=True):
                st.session_state.selected_opportunity = str(opportunity["Opportunity_ID"])
                st.session_state.volunteer_page = "Find opportunities"
                st.rerun()
            if save_col2.button("Remove from saved", key="saved_remove_" + str(opportunity["Opportunity_ID"]),
                use_container_width=True):
                oid = str(opportunity["Opportunity_ID"])
                saved = saved[~((saved["Volunteer_ID"].astype(str) == str(st.session_state.id))
                        & (saved["Opportunity_ID"].astype(str) == oid))]
                if save(saved, "saved_opportunities.csv"):
                    st.rerun()

def volunteer_profile(volunteer):
    st.header("My profile")
    completion = profile_completion(volunteer)
    st.progress(completion)
    st.caption("Profile completion: " + str(completion) + "%")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", str(volunteer.get("Name", "")))
        email = st.text_input("Email", str(volunteer.get("Email", "")))
        phone = st.text_input("Phone", str(volunteer.get("Phone", "")))
        location = st.text_input("Location", str(volunteer.get("Location", "")))
        mode_choices = ["", "Offline", "Hybrid", "Online"]
        preferred_mode = st.selectbox("Preferred mode", mode_choices, index=(mode_choices.index(
                    str(volunteer.get("Preferred_Mode", ""))) if str(volunteer.get("Preferred_Mode", ""))
                in mode_choices else 0))
    with col2:
        skills = st.text_area("Skills", str(volunteer.get("Skills", "")))
        interests = st.text_area("Interests", str(volunteer.get("Interests", "")))
        availability = st.text_input("Availability", str(volunteer.get("Availability", "")))
    experience = st.text_area("Experience", str(volunteer.get("Experience", "")))
    qualification = st.text_input("Highest qualification", str(volunteer.get("Qualification", "")),
        placeholder="e.g. B.Ed, B.Sc, MA")
    bio = st.text_area("Bio", str(volunteer.get("Bio", "")), placeholder="Tell NGOs a little about yourself...")

    if st.button("Save profile", use_container_width=True):
        if not str(name).strip() or not str(email).strip():
            st.error("Name and email are required.")
        elif not str(skills).strip():
            st.error("Please add at least one skill so we can match you to opportunities.")
        elif not valid_phone(phone):
            st.error("Phone number must contain exactly 10 digits (numbers only).")
        elif not bio.strip():
            st.error("Please add a short bio.")
        else:
            index = volunteers[volunteers["Volunteer_ID"].astype(str) == str(st.session_state.id)].index[0]
            volunteers.loc[index, ["Name", "Email", "Phone", "Location", "Skills", "Interests", "Availability",
                    "Preferred_Mode", "Experience", "Qualification", "Bio"]] = [name, email, phone.strip(),
                location, skills, interests, availability, preferred_mode, experience, qualification, bio.strip()]
            if save(volunteers, "volunteer.csv"):
                st.success("Profile saved!")
                st.rerun()

def volunteer_dashboard():
    volunteer_rows = volunteers[volunteers["Volunteer_ID"].astype(str) == str(st.session_state.id)]
    if volunteer_rows.empty:
        st.error("Volunteer profile not found.")
        return
    volunteer = volunteer_rows.iloc[0]

    # Page state lets an opportunity open as a proper details page.
    if "volunteer_page" not in st.session_state:
        st.session_state.volunteer_page = "Dashboard"
    st.sidebar.title("Volunteer Menu")
    current_page = (st.session_state.volunteer_page if st.session_state.volunteer_page in VOLUNTEER_PAGES
        else "Dashboard")
    page = st.sidebar.radio("Go to", VOLUNTEER_PAGES, index=VOLUNTEER_PAGES.index(current_page))
    st.session_state.volunteer_page = page
    st.title("🤝 Skill Connect for Social Impact")
    if st.button("Log out"):
        logout()
    st.divider()

    # AI ASSISTANT
    if page == "AI Assistant":
        render_ai_assistant("Volunteer", volunteer, None, opportunities, applications,
            saved, ngos, rank_opportunities, skill_lexicon)
    elif page == "Dashboard":
        volunteer_home(volunteer)
    elif page == "Find opportunities":
        selected_id = st.session_state.get("selected_opportunity", None)
        if selected_id is not None:
            volunteer_opportunity_details(volunteer, selected_id)
        else:
            volunteer_browse_opportunities(volunteer)
    elif page == "My applications":
        volunteer_applications(volunteer)
    elif page == "Saved opportunities":
        volunteer_saved(volunteer)
    elif page == "My profile":
        volunteer_profile(volunteer)

# NGO PAGES (one function per sidebar page)
def ngo_home(ngo, my_roles, my_apps):
    st.header(ngo["Name"])
    statuses = (my_roles.apply(display_status, axis=1) if not my_roles.empty else pd.Series(dtype="object"))
    stored = (my_roles.apply(stored_status, axis=1) if not my_roles.empty else pd.Series(dtype="object"))
    application_status = (my_apps["Status"].astype(str).str.lower() if not my_apps.empty
        else pd.Series(dtype="object"))
    open_count = int((statuses == "Open").sum())
    closed_count = int((stored == "Closed").sum())
    cancelled_count = int((stored == "Cancelled").sum())
    expired_count = int((statuses == "Expired").sum())
    pending_count = int((application_status == "pending").sum())
    shortlisted_count = int((application_status == "shortlisted").sum())
    accepted_count_total = int((application_status == "accepted").sum())
    available_slots = (int(my_roles.apply(slots_left, axis=1).sum()) if not my_roles.empty else 0)
    st.subheader("Opportunity statistics")
    row1 = st.columns(4)
    row1[0].metric("Total opportunities", len(my_roles))
    row1[1].metric("Open", open_count)
    row1[2].metric("Closed", closed_count)
    row1[3].metric("Cancelled", cancelled_count)
    row2 = st.columns(4)
    row2[0].metric("Total applications", len(my_apps))
    row2[1].metric("Pending", pending_count)
    row2[2].metric("Accepted volunteers", accepted_count_total)
    row2[3].metric("Available slots", available_slots)
    extra1, extra2 = st.columns(2)
    extra1.metric("Shortlisted", shortlisted_count)
    extra2.metric("Expired (deadline passed)", expired_count)

    # DEADLINE ALERTS
    st.subheader("Deadlines needing attention")
    alerts = []
    for _, role in my_roles.iterrows():
        if stored_status(role) != "Open":
            continue
        remaining = days_left(role)
        if remaining is None:
            continue
        if remaining <= 7:
            alerts.append((remaining, role))
    if not alerts:
        st.caption("No deadlines in the next 7 days.")
    else:
        alerts.sort(key=lambda item: item[0])
        for remaining, role in alerts:
            label = (str(role.get("Role_Title", "Untitled")) + " — " + deadline_note(role))
            if remaining < 0:
                st.error(label)
            else:
                st.warning(label)
    st.divider()

    # The raw directory value is shown as separate tags. The stored Area_of_Work text itself is left exactly as sourced.
    render_tags(ngo["Area_of_Work"], heading="Areas of Work", style="list",
        empty_text="No areas of work listed in the directory.")

def ngo_manage_opportunities(ngo, my_roles):
    global opportunities
    global applications
    st.header("My opportunities")
    # CREATE OPPORTUNITY
    st.subheader("Create a new opportunity")
    with st.form("create_opportunity_form"):
        create_title = st.text_input("Role title")
        create_description = st.text_area("Description")
        create_skills = st.text_input("Skills required")
        create_area = st.selectbox("Opportunity area", OPPORTUNITY_AREAS)
        create_mode = st.selectbox("Mode", ["Offline", "Hybrid", "Online"])
        create_time = st.text_input("Time commitment", placeholder="e.g. 5 hours per week, weekends")
        create_duration = st.text_input("Duration", placeholder="e.g. 3 months")

        # An opportunity can never be created with a past deadline.
        create_deadline = st.date_input("Application deadline",
            value=(today_date() + pd.Timedelta(days=14)).date(), min_value=today_date().date())
        create_qualification = st.text_input("Qualification (optional but useful)",
            placeholder="Leave blank if no qualification is needed")
        create_experience = st.text_input("Experience required", placeholder="e.g. None, or 1 year of teaching")
        create_volunteers = st.number_input("Volunteers required", min_value=1, value=1, step=1)
        create_submit = st.form_submit_button("Create opportunity")
    if create_submit:

        # Each required field is checked individually so the NGO is told exactly what is missing instead of a single generic error.

        required_fields = {"Role title": create_title.strip(), "Description": create_description.strip(),
            "Skills required": create_skills.strip(), "Opportunity area": str(create_area).strip(),
            "Mode": str(create_mode).strip(), "Time commitment": create_time.strip(),
            "Duration": create_duration.strip()}
        missing = [label for label, value in required_fields.items() if not value]
        deadline_value = parse_date(create_deadline)
        if missing:
            st.error("Please complete the following: " + ", ".join(missing) + ".")
        elif len(create_description.strip()) < 30:
            st.error("Description must be at least 30 characters so volunteers understand the role.")
        elif deadline_value is None:
            st.error("Please choose a valid application deadline.")
        elif deadline_value < today_date():
            st.error("The deadline has already passed. Choose a future date.")
        elif int(create_volunteers) < 1:
            st.error("At least one volunteer is required.")
        else:
            new_opportunity = {"Opportunity_ID": new_id(opportunities, "Opportunity_ID"),
                "NGO_ID": st.session_state.id, "Role_Title": create_title.strip(),
                "Description": create_description.strip(), "Skills_Required": create_skills.strip(),
                "Area": create_area, "Location": str(ngo.get("Location", "Not specified")),
                "Mode (Online/Offline/Hybrid)": create_mode, "Time_Commitment": create_time.strip(),
                "Qualification (optional but useful)": (create_qualification.strip()),
                "Experience_Required": create_experience.strip(), "Duration": create_duration.strip(),
                "Volunteers_Required": str(create_volunteers), "Application_Deadline": str(create_deadline),
                "Status (Open/Closed)": "Open"}
            opportunities.loc[len(opportunities)] = new_opportunity
            if save(opportunities, "opportunities.csv"):
                st.success("Opportunity created successfully.")
                st.rerun()
    st.divider()

    # VIEW / EDIT / CLOSE / REOPEN / CANCEL / DELETE
    st.subheader("Manage your opportunities")
    if my_roles.empty:
        st.info("You have not posted any opportunities yet.")
    else:
        for index, role in my_roles.iterrows():
            oid = str(role["Opportunity_ID"])
            title_text = str(role.get("Role_Title", "Untitled"))
            current_status = str(role.get("Status (Open/Closed)", "Open"))

            # The derived status reflects deadline and capacity, not just the stored
            # Open/Closed/Cancelled value.
            role_display_status = display_status(role)
            with st.expander(title_text + " — " + status_badge(role_display_status), expanded=False):
                # VIEW
                st.write(str(role.get("Description", "")))
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.write("**Area:** " + str(role.get("Area", "Not specified")))
                    st.write("**Location:** " + str(role.get("Location", "Not specified")))
                    st.write("**Mode:** " + str(role.get("Mode (Online/Offline/Hybrid)", "Not specified")))
                    render_tags(role.get("Skills_Required", ""), heading="Skills", style="inline")
                with detail_col2:
                    st.write("**Duration:** " + str(role.get("Duration", "Not specified")))
                    st.write("**Time commitment:** " + str(role.get("Time_Commitment", "Not specified")))
                    st.write("**Deadline:** " + str(role.get("Application_Deadline", "Not specified")))
                    st.write("**Volunteers required:** " + str(role.get("Volunteers_Required", "Not specified")))
                    st.write("**Slots filled:** " + str(accepted_count(oid)) + " / "
                        + str(volunteers_required(role)))
                    note = deadline_note(role)
                    if note:
                        st.write("**Deadline status:** " + note)
                st.write("**Qualification:** " + str(role.get("Qualification (optional but useful)",
                            "Not specified")))
                st.write("**Experience required:** " + str(role.get("Experience_Required", "Not specified")))
                st.divider()

                # EDIT
                st.markdown("**Edit opportunity**")
                with st.form("edit_opportunity_" + oid):
                    edit_title = st.text_input("Role title", value=title_text)
                    edit_description = st.text_area("Description", value=str(role.get("Description", "")))
                    edit_skills = st.text_input("Skills required", value=str(role.get("Skills_Required", "")))
                    current_area = str(role.get("Area", OPPORTUNITY_AREAS[0]))
                    edit_area = st.selectbox("Opportunity area", OPPORTUNITY_AREAS, index=(
                            OPPORTUNITY_AREAS.index(current_area) if current_area in OPPORTUNITY_AREAS else 0))
                    modes = ["Offline", "Hybrid", "Online"]
                    current_mode = str(role.get("Mode (Online/Offline/Hybrid)", "Hybrid"))
                    edit_mode = st.selectbox("Mode", modes, index=(modes.index(current_mode)
                            if current_mode in modes else 1))
                    edit_time = st.text_input("Time commitment", value=str(role.get("Time_Commitment", "")))
                    edit_duration = st.text_input("Duration", value=str(role.get("Duration", "")))

                    # Deadlines can be extended, but never set into the past. An already-expired deadline defaults to today.
                    existing_deadline = parse_date(role.get("Application_Deadline", ""))
                    if (existing_deadline is None or existing_deadline < today_date()):
                        deadline_default = today_date()
                    else:
                        deadline_default = existing_deadline
                    edit_deadline = st.date_input("Application deadline", value=deadline_default.date(),
                        min_value=today_date().date(), key="edit_deadline_" + oid)
                    edit_qualification = st.text_input("Qualification (optional but useful)", value=str(role.get(
                                "Qualification (optional but useful)", "")))
                    edit_experience = st.text_input("Experience required", value=str(role.get(
                                "Experience_Required", "")))
                    try:
                        existing_volunteers = int(float(str(role.get("Volunteers_Required", "1"))))
                    except (ValueError, TypeError):
                        existing_volunteers = 1
                    edit_volunteers = st.number_input("Volunteers required", min_value=1,
                        value=max(1, existing_volunteers), step=1)
                    edit_submit = st.form_submit_button("Save changes")
                if edit_submit:
                    edit_required = {"Role title": edit_title.strip(), "Description": edit_description.strip(),
                        "Skills required": edit_skills.strip(), "Time commitment": edit_time.strip(),
                        "Duration": edit_duration.strip()}
                    edit_missing = [label for label, value in edit_required.items() if not value]
                    already_accepted = accepted_count(oid)
                    if edit_missing:
                        st.error("Please complete the following: " + ", ".join(edit_missing) + ".")
                    elif int(edit_volunteers) < already_accepted:
                        st.error("You have already accepted " + str(already_accepted)
                            + " volunteer(s). The number of volunteers required cannot be lower than that.")
                    else:
                        opportunities.loc[index, "Role_Title"] = edit_title.strip()
                        opportunities.loc[index, "Description"] = edit_description.strip()
                        opportunities.loc[index, "Skills_Required"] = edit_skills.strip()
                        opportunities.loc[index, "Area"] = edit_area
                        opportunities.loc[index, "Mode (Online/Offline/Hybrid)"] = edit_mode
                        opportunities.loc[index, "Time_Commitment"] = edit_time.strip()
                        opportunities.loc[index, "Duration"] = edit_duration.strip()
                        opportunities.loc[index, "Qualification (optional but useful)"
                        ] = edit_qualification.strip()
                        opportunities.loc[index, "Experience_Required"] = edit_experience.strip()
                        opportunities.loc[index, "Volunteers_Required"] = str(edit_volunteers)
                        opportunities.loc[index, "Application_Deadline"] = str(edit_deadline)
                        if save(opportunities, "opportunities.csv"):
                            st.success("Opportunity updated successfully.")
                            st.rerun()
                st.divider()

                # CLOSE / REOPEN / CANCEL
                st.markdown("**Opportunity status**")
                stored = stored_status(role)
                st.write("Current status: " + status_badge(role_display_status))
                if role_display_status == "Expired":
                    st.warning("This opportunity is still marked Open but the "
                        "deadline has passed, so it no longer accepts "
                        "applications. Extend the deadline above to reopen it to volunteers.")
                if role_display_status == "Full":
                    st.info("All volunteer slots are filled, so this "
                        "opportunity no longer appears to volunteers.")
                status_col1, status_col2 = st.columns(2)
                if stored == "Open":
                    if status_col1.button("Close opportunity", key="close_" + oid):
                        opportunities.loc[index, "Status (Open/Closed)"] = "Closed"
                        if save(opportunities, "opportunities.csv"):
                            st.success("Opportunity closed.")
                            st.rerun()
                elif stored == "Closed":
                    if status_col1.button("Reopen opportunity", key="reopen_" + oid):
                        reopen_deadline = parse_date(role.get("Application_Deadline", ""))
                        if (reopen_deadline is not None and reopen_deadline < today_date()):
                            st.error("The deadline has passed. Extend the deadline above before reopening.")
                        else:
                            opportunities.loc[index, "Status (Open/Closed)"] = "Open"
                            if save(opportunities, "opportunities.csv"):
                                st.success("Opportunity reopened.")
                                st.rerun()
                else:
                    # Cancelled is deliberately separate from Closed. Restoring it is an explicit, distinct action.
                    if status_col1.button("Restore cancelled opportunity", key="restore_" + oid):
                        restore_deadline = parse_date(role.get("Application_Deadline", ""))
                        if (restore_deadline is not None and restore_deadline < today_date()):
                            st.error("The deadline has passed. Extend the deadline above before restoring.")
                        else:
                            opportunities.loc[index, "Status (Open/Closed)"] = "Open"
                            if save(opportunities, "opportunities.csv"):
                                st.success("Opportunity restored and reopened.")
                                st.rerun()

                # CANCEL = non-destructive removal from normal discovery. Keep the record for audit/history.
                if stored != "Cancelled":
                    if status_col2.button("Cancel opportunity", key="cancel_" + oid):
                        st.session_state["confirm_cancel_" + oid] = True
                if st.session_state.get("confirm_cancel_" + oid, False):
                    st.warning("Cancelling removes this opportunity from the volunteer discovery list.")
                    confirm_col1, confirm_col2 = st.columns(2)
                    if confirm_col1.button("Confirm cancel", key="confirm_cancel_button_" + oid):
                        opportunities.loc[index, "Status (Open/Closed)"] = "Cancelled"
                        if save(opportunities, "opportunities.csv"):
                            st.session_state.pop("confirm_cancel_" + oid, None)
                            st.success("Opportunity cancelled.")
                            st.rerun()
                    if confirm_col2.button("Keep opportunity", key="keep_cancel_button_" + oid):
                        st.session_state.pop("confirm_cancel_" + oid, None)
                        st.rerun()

                # DELETE = permanent removal, including linked applications.
                st.divider()
                st.markdown("**Permanent deletion**")
                if st.button("Delete opportunity permanently", key="delete_" + oid):
                    st.session_state["confirm_delete_" + oid] = True
                if st.session_state.get("confirm_delete_" + oid, False):
                    st.error("Permanent deletion cannot be undone. Linked applications will also be removed.")
                    delete_col1, delete_col2 = st.columns(2)
                    if delete_col1.button("Confirm permanent delete", key="confirm_delete_button_" + oid):
                        opportunities = opportunities[opportunities["Opportunity_ID"].astype(str) != oid
                        ].reset_index(drop=True)
                        applications = applications[applications["Opportunity_ID"].astype(str) != oid
                        ].reset_index(drop=True)
                        opportunity_saved = save(opportunities, "opportunities.csv")
                        applications_saved = save(applications, "application.csv")
                        if (opportunity_saved and applications_saved):
                            st.session_state.pop("confirm_delete_" + oid, None)
                            st.success("Opportunity permanently deleted.")
                            st.rerun()
                    if delete_col2.button("Keep opportunity", key="keep_delete_button_" + oid):
                        st.session_state.pop("confirm_delete_" + oid, None)
                        st.rerun()

def ngo_review_applications(my_roles, my_apps):
    st.header("Volunteer applications")
    if my_roles.empty:
        st.info("You have not posted any opportunities yet, so there are no applications.")
    else:
        # Withdrawn applications remain stored in the CSV for history, but are not part of the
        # NGO's active applicant list.
        active_my_apps = my_apps[my_apps["Status"].astype(str).str.strip().str.lower() != "withdrawn"].copy()
        if active_my_apps.empty:
            st.info("No active applications received yet.")
            return

        # FILTERS AND SORTING
        role_titles = {str(row["Opportunity_ID"]): str(row.get("Role_Title", "Untitled"))
            for _, row in my_roles.iterrows()}
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            opportunity_filter = st.selectbox("Filter by opportunity",
                ["All opportunities"] + sorted(role_titles.values()))
        with filter_col2:
            status_filter = st.selectbox("Filter by status", ["All", "Pending", "Shortlisted", "Accepted",
                    "Rejected"])
        with filter_col3:
            sort_by = st.selectbox("Sort by", ["Newest first", "Oldest first", "Status", "Best match first"])
        search_name = st.text_input("Search volunteer by name", placeholder="Start typing a name...")

        # BUILD THE APPLICANT LIST
        lexicon = skill_lexicon(volunteers, opportunities)
        applicant_rows = []
        for index, application in active_my_apps.iterrows():
            volunteer_rows = volunteers[volunteers["Volunteer_ID"].astype(str) == str(application["Volunteer_ID"])
            ]
            role_rows = opportunities[opportunities["Opportunity_ID"].astype(str)
                == str(application["Opportunity_ID"])]
            if volunteer_rows.empty or role_rows.empty:
                continue
            volunteer_row = volunteer_rows.iloc[0]
            role_row = role_rows.iloc[0]
            match = applicant_match_result(volunteer_row, role_row,
                lexicon=lexicon)
            applicant_rows.append({"index": index, "application": application, "volunteer": volunteer_row,
                "role": role_row, "match": match, "match_rows": match["rows"], "match_total": match["overall"],
                "status": str(application["Status"]).strip(), "date": parse_date(
                    application.get("Application_Date", ""))})

        # APPLY FILTERS
        filtered = []
        for row in applicant_rows:
            if (opportunity_filter != "All opportunities" and str(row["role"].get("Role_Title", ""))
                != opportunity_filter):
                continue
            if (status_filter != "All" and row["status"].lower() != status_filter.lower()):
                continue
            if (search_name.strip() and search_name.strip().lower()
                not in str(row["volunteer"].get("Name", "")).lower()):
                continue
            filtered.append(row)

        # APPLY SORTING ; A fixed sentinel is used for applications with an unreadable date. pd.Timestamp.min cannot be normalised without overflow.
        oldest = pd.Timestamp("1900-01-01")
        if sort_by == "Newest first":
            filtered.sort(key=lambda row: row["date"] or oldest, reverse=True)
        elif sort_by == "Oldest first":
            filtered.sort(key=lambda row: row["date"] or oldest)
        elif sort_by == "Status":
            status_order = {"pending": 0, "shortlisted": 1, "accepted": 2, "rejected": 3, "withdrawn": 4}
            filtered.sort(key=lambda row: status_order.get(row["status"].lower(), 5))
        else:
            filtered.sort(key=lambda row: row["match_total"], reverse=True)
        st.caption(str(len(filtered)) + " application(s) shown out of " + str(len(applicant_rows)))
        if not filtered:
            st.info("No applications match the current filters.")

        # APPLICANT CARDS
        for row in filtered:
            application = row["application"]
            volunteer_row = row["volunteer"]
            role_row = row["role"]
            index = row["index"]
            application_id = str(application["Application_ID"])
            status = row["status"]
            status_lower = status.lower()
            with st.container(border=True):
                header_left, header_right = st.columns([3, 1])
                with header_left:
                    st.subheader(str(volunteer_row.get("Name", "Volunteer")))
                    st.caption("Applied for " + str(role_row.get("Role_Title", "Untitled")) + " on " + str(
                            application.get("Application_Date", "Unknown date")))
                with header_right:
                    st.write(APPLICATION_STATUS_LABELS.get(status_lower, status))
                    st.caption("AI applicant fit: " + str(row["match_total"]) + "%")

                # VOLUNTEER PROFILE
                with st.expander("View full volunteer profile"):
                    profile_col1, profile_col2 = st.columns(2)
                    with profile_col1:
                        st.write("**Name:** " + str(volunteer_row.get("Name", "Not provided")))
                        st.write("**Location:** " + str(volunteer_row.get("Location", "Not provided")))
                        st.write("**Qualification:** " + str(volunteer_row.get("Qualification", "Not provided")))
                        st.write("**Availability:** " + str(volunteer_row.get("Availability", "Not provided")))
                        st.write("**Preferred mode:** " + str(volunteer_row.get("Preferred_Mode", "Not provided"))
                        )
                    with profile_col2:
                        render_tags(volunteer_row.get("Skills", ""), heading="Skills", style="inline",
                            empty_text="Not provided")
                        render_tags(volunteer_row.get("Interests", ""), heading="Interests", style="inline",
                            empty_text="Not provided")
                        st.write("**Experience:** " + str(volunteer_row.get("Experience", "Not provided")))
                    st.write("**Bio:** " + str(volunteer_row.get("Bio", "Not provided")))

                    # Contact details are only revealed once the volunteer has been accepted.
                    if status_lower == "accepted":
                        st.write("**Email:** " + str(volunteer_row.get("Email", "Not provided")))
                        st.write("**Phone:** " + str(volunteer_row.get("Phone", "Not provided")))
                    else:
                        st.caption("Contact details become visible once you accept this volunteer.")

                # MATCH BREAKDOWN
                with st.expander("Why this volunteer matches"):
                    render_why_this_match(row["match"])
                    applicant_match = row["match"]
                    st.caption("AI ranking considers semantic fit, required skills, interests, "
                        "experience, qualification, location/mode and availability.")
                    gap = applicant_match.get("skill_gap", {})
                    if gap.get("missing"):
                        st.warning("Potential skill gaps: " + ", ".join(
                                skill.title() for skill in gap["missing"][:5]))
                    elif gap.get("required"):
                        st.success("No required skills were missing from the detected profile skills.")
                    st.caption("AI ranking is decision support only. The NGO makes the final selection.")

                # DECISIONS
                if status_lower == "withdrawn":
                    st.info("⚪ This volunteer withdrew their application. No action is available.")
                elif status_lower == "accepted":
                    st.success("🟢 Accepted for this opportunity.")
                elif status_lower == "rejected":
                    st.error("🔴 This application was rejected.")
                else:
                    remaining_slots = slots_left(role_row)
                    st.caption("Slots remaining for this opportunity: " + str(remaining_slots) + " of "
                        + str(volunteers_required(role_row)))
                    action_col1, action_col2, action_col3 = st.columns(3)

                    # ACCEPT
                    if remaining_slots == 0:
                        action_col1.button("No slots left", disabled=True, key="no_slots_" + application_id,
                            use_container_width=True)
                    else:
                        if action_col1.button("Accept", key="accept_" + application_id, use_container_width=True):
                            st.session_state["pending_action_" + application_id] = "Accepted"

                    # SHORTLIST / KEEP PENDING
                    if status_lower == "shortlisted":
                        if action_col2.button("Move back to pending", key="unshortlist_" + application_id,
                            use_container_width=True):
                            applications.loc[index, "Status"] = "Pending"
                            if save(applications, "application.csv"):
                                st.rerun()
                    else:
                        if action_col2.button("Shortlist", key="shortlist_" + application_id,
                            use_container_width=True):
                            applications.loc[index, "Status"] = "Shortlisted"
                            if save(applications, "application.csv"):
                                st.rerun()

                    # REJECT
                    if action_col3.button("Reject", key="reject_" + application_id, use_container_width=True):
                        st.session_state["pending_action_" + application_id] = "Rejected"

                    # CONFIRMATION
                    pending_action = st.session_state.get("pending_action_" + application_id, None)
                    if pending_action == "Accepted":
                        st.warning("Accept " + str(volunteer_row.get("Name", "this volunteer")) + " for "
                            + str(role_row.get("Role_Title", "this role")) + "? This uses one volunteer slot and "
                            "reveals their contact details.")
                    elif pending_action == "Rejected":
                        st.warning("Reject " + str(volunteer_row.get("Name", "this volunteer"))
                            + "? This cannot be undone from this screen.")
                    if pending_action:
                        confirm_col1, confirm_col2 = st.columns(2)
                        if confirm_col1.button("Confirm " + pending_action.lower(),
                            key="confirm_" + application_id, use_container_width=True):
                            # Capacity is re-checked at the moment of confirmation, not only when
                            # the button was first clicked.
                            if (pending_action == "Accepted" and slots_left(role_row) == 0):
                                st.error("All volunteer slots have been filled. This volunteer cannot "
                                    "be accepted.")
                                st.session_state.pop("pending_action_" + application_id, None)
                            else:
                                applications.loc[index, "Status"] = pending_action
                                if save(applications, "application.csv"):
                                    st.session_state.pop("pending_action_" + application_id, None)
                                    st.success("Application " + pending_action.lower() + ".")
                                    st.rerun()
                        if confirm_col2.button("Keep pending", key="keep_pending_" + application_id,
                            use_container_width=True):
                            st.session_state.pop("pending_action_" + application_id, None)
                            st.rerun()

def ngo_profile(ngo):
    st.header("NGO profile")
    name = st.text_input("NGO name", str(ngo["Name"]))
    email = st.text_input("Email", str(ngo["Email_ID"]))
    render_tags(ngo["Area_of_Work"], heading="Areas of Work (as displayed)", style="inline",
        empty_text="No areas of work listed in the directory.")
    area = st.text_input("Area of work", str(ngo["Area_of_Work"]))
    address = st.text_area("Address", str(ngo["Address"]))
    since = st.text_input("Working since", str(ngo["Working_Since"]))
    if st.button("Save NGO profile"):
        index = ngos[ngos["SrNo"].astype(str) == str(st.session_state.id)].index[0]
        ngos.loc[index, ["Name", "Email_ID", "Area_of_Work", "Address", "Working_Since"]] = [name, email, area,
            address, since]
        if save(ngos, "ngo.csv"):
            st.success("NGO profile saved!")

def ngo_dashboard():
    ngo = ngos[ngos["SrNo"].astype(str) == str(st.session_state.id)]
    if ngo.empty:
        st.error("NGO profile not found.")
        return
    ngo = ngo.iloc[0]
    st.sidebar.title("NGO Menu")
    NGO_PAGES = ["Dashboard", "My opportunities", "Applications", "NGO profile", "AI Assistant"]
    if "ngo_page" not in st.session_state:
        st.session_state.ngo_page = "Dashboard"
    current_page = (st.session_state.ngo_page if st.session_state.ngo_page in NGO_PAGES
        else "Dashboard")
    page = st.sidebar.radio("Go to", NGO_PAGES, index=NGO_PAGES.index(current_page))
    st.session_state.ngo_page = page
    st.title("🤝 Skill Connect for Social Impact")
    if st.button("Log out"):
        logout()
    st.divider()
    my_roles = opportunities[opportunities["NGO_ID"].astype(str) == str(st.session_state.id)]
    my_apps = applications[applications["Opportunity_ID"].astype(str).isin(my_roles["Opportunity_ID"].astype(str))]

    # AI ASSISTANT
    if page == "AI Assistant":
        render_ai_assistant("NGO", None, ngo, opportunities, applications,
            saved, ngos, rank_opportunities, skill_lexicon)
    elif page == "Dashboard":
        ngo_home(ngo, my_roles, my_apps)
    elif page == "My opportunities":
        ngo_manage_opportunities(ngo, my_roles)
    elif page == "Applications":
        ngo_review_applications(my_roles, my_apps)
    elif page == "NGO profile":
        ngo_profile(ngo)

# RUN APP
if not st.session_state.logged_in:
    login_page()
elif st.session_state.role == "Volunteer":
    volunteer_dashboard()
else:
    ngo_dashboard()
