###############################################################
#  GEM Tender Smart Chatbot (Optimized Production Version)
#  CHUNK 1 — Page Config, Env, Auth, Caching, Utility Functions
###############################################################

import re
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from supabase import create_client
from openai import OpenAI
# from dotenv import load_dotenv
from typing import List, Dict, Any

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="GEM Tender Smart Chatbot",
    layout="wide"
)

# ------------------ LOAD ENV VARS -----------------
# load_dotenv()
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    st.error("Missing SUPABASE_URL, SUPABASE_KEY or OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------ SESSION SETUP ------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "filters" not in st.session_state:
    st.session_state.filters = {}
if "context_text" not in st.session_state:
    st.session_state.context_text = ""
if "all_rows" not in st.session_state:
    st.session_state.all_rows = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ LOGIN (UNCHANGED) ------------------
def login_user(email, password):
    try:
        return supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
    except Exception:
        return None

def logout_user():
    st.session_state.user = None
    st.rerun()

# Login Screen
if st.session_state.user is None:
    st.title("🔐 Login to Access GEM Tender Chatbot")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(email, password)
        if user and user.user:
            st.session_state.user = user
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid email or password")

    st.stop()

# ------------------ SIDEBAR LOGOUT ------------------
with st.sidebar:
    st.success(f"Logged in as: {st.session_state.user.user.email}")
    if st.button("Logout"):
        logout_user()

# ------------------ CONSTANTS ------------------
EMBED_MODEL = "text-embedding-3-small"
TOTAL = supabase.table("GEM_TABLE_UPDATED").select('*', count="exact").execute()
TOP_K = TOTAL.count
MAX_CTX_CHARS = 30000
DATE_FORMATS = ["%d-%m-%Y %I:%M %p", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]

# ------------------ CACHING LAYERS ------------------

@st.cache_data(show_spinner=False, ttl=3600)
def cached_distinct(col):
    resp = supabase.table("GEM_TABLE_UPDATED").select(f'"{col}"').execute()
    return [""] + sorted({r[col] for r in resp.data if r.get(col)})

@st.cache_data(show_spinner=False, ttl=600)
def cached_fetch_rows(equality_filters):
    q = supabase.table("GEM_TABLE_UPDATED").select("*")
    for col, val in equality_filters.items():
        q = q.eq(col, val)
    return q.execute().data or []

@st.cache_data(show_spinner=False)
def cached_apply_filters(rows, filters):
    return apply_advanced_filters(rows, filters)

@st.cache_data(show_spinner=False)
def cached_valid_date(rows):
    return filter_valid_end_date(rows)

@st.cache_data(show_spinner=False)
def cached_embedding(text):
    if not text:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

# ------------------ HELPERS ------------------
def parse_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except:
            continue
    return None

def cosine_similarity(a, b):
    a_arr, b_arr = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return -1
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def safe_eb(row):
    emb = row.get("Bid_Embedding")
    if emb is None:
        return []
    if isinstance(emb, list):
        return emb
    if isinstance(emb, str):
        try:
            return json.loads(emb)
        except:
            return []
    return []
###############################################################
#  CHUNK 2 — Optimized Sidebar Filters + Filter Processing
###############################################################

st.title("💬 GEM Tender Chat Assistant")

# ============= Build Query Mapping for Filters =============
def build_equality_filter(filters):
    mapping = {
        "Ministry Name": "Ministry Name",
        "Department Name": "Department Name",
        "Organisation Name": "Organisation Name",
        "Office Name": "Office Name",
        "Item Category": "Item Category",
        "MII Compliance": "MII Compliance",
    }
    q = {}
    for label, db_col in mapping.items():
        val = filters.get(label)
        if isinstance(val, list):
            # multiselect → only allow equality for single values
            if len(val) == 1:
                q[db_col] = val[0]
        else:
            if val not in ("", None):
                q[db_col] = val
    return q

# ============= Sidebar with FORM (prevents rerun each click) =============
with st.sidebar.form("filters_form"):
    st.header("🎯 Apply Filters")

    ministry = st.multiselect("Ministry Name", cached_distinct("Ministry Name"))
    dept = st.multiselect("Department Name", cached_distinct("Department Name"))
    org = st.multiselect("Organisation Name", cached_distinct("Organisation Name"))
    office = st.multiselect("Office Name", cached_distinct("Office Name"))
    item_category = st.multiselect("Item Category", cached_distinct("Item Category"))

    mse_exemption = st.multiselect(
        "MSE Exemption for Experience/Turnover",
        cached_distinct("MSE Exemption for Years of Experience and Turnover")
    )

    startup_exemption = st.multiselect(
        "Startup Exemption for Experience/Turnover",
        cached_distinct("Startup Exemption for Years of Experience and Turnover")
    )

    review_status_filter = st.multiselect(
        "Review Status",
        ["Not Reviewed", "Suitable", "Not Suitable", "Read",
         "To be Consumed in Depth", "Shared with Solutioning", "Done"]
    )

    remove_est_notfound = st.checkbox("Remove tenders with Estimated Bid Value = Not found")

    contract_min = st.text_input("Contract Period (Min)")
    contract_max = st.text_input("Contract Period (Max)")

    est_min = st.text_input("Estimated Bid Value (Min)")
    est_max = st.text_input("Estimated Bid Value (Max)")

    bov_min = st.text_input("Bid Offer Validity (Min)")
    bov_max = st.text_input("Bid Offer Validity (Max)")

    turn_min = st.text_input("Minimum Avg Annual Turnover (Min)")
    turn_max = st.text_input("Minimum Avg Annual Turnover (Max)")

    exp_min = st.text_input("Past Experience Required (Min)")
    exp_max = st.text_input("Past Experience Required (Max)")

    start_date_range = st.date_input("Start Date Range", [])
    end_date_range = st.date_input("End Date Range", [])

    mii_compliance = st.selectbox("MII Compliance", cached_distinct("MII Compliance"))

    filters_submitted = st.form_submit_button("Apply Filters")

# ============= Process Filter Submission =============
if filters_submitted:
    st.session_state.filters = {
        "Ministry Name": ministry,
        "Department Name": dept,
        "Organisation Name": org,
        "Office Name": office,
        "Item Category": item_category,
        "MSE Exemption for Years of Experience and Turnover": mse_exemption,
        "Startup Exemption for Years of Experience and Turnover": startup_exemption,
        "Review_Status": review_status_filter,

        "contract_min": int(contract_min) if contract_min.strip().isdigit() else None,
        "contract_max": int(contract_max) if contract_max.strip().isdigit() else None,

        "est_min": int(est_min) if est_min.strip().isdigit() else None,
        "est_max": int(est_max) if est_max.strip().isdigit() else None,

        "bov_min": int(bov_min) if bov_min.strip().isdigit() else None,
        "bov_max": int(bov_max) if bov_max.strip().isdigit() else None,

        "turn_min": int(turn_min) if turn_min.strip().isdigit() else None,
        "turn_max": int(turn_max) if turn_max.strip().isdigit() else None,

        "exp_min": int(exp_min) if exp_min.strip().isdigit() else None,
        "exp_max": int(exp_max) if exp_max.strip().isdigit() else None,

        "start_date_range": start_date_range,
        "end_date_range": end_date_range,

        "MII Compliance": mii_compliance,
        "remove_est_notfound": remove_est_notfound,
    }

    st.success("Filters applied successfully!")
    st.rerun()

# Retrieve current filters
filters = st.session_state.filters

# ================================================================
#       ADVANCED FILTER LOGIC (unchanged but placed below)
# ================================================================
def apply_advanced_filters(rows, filters):
    def norm(x):
        return str(x).strip().lower() if x else ""

    MISSING_TOKENS = {"not found", "na", "n/a", "-", "none"}

    result = []

    for r in rows:
        passed = True

        # Remove Estimated Not Found
        if filters.get("remove_est_notfound"):
            est_raw = str(r.get("Estimated Bid Value", "")).lower()
            if not any(ch.isdigit() for ch in est_raw):
                continue

        # Multi-select checks
        for field in [
            "Ministry Name", "Department Name",
            "Organisation Name", "Office Name",
            "Item Category",
            "MSE Exemption for Years of Experience and Turnover",
            "Startup Exemption for Years of Experience and Turnover",
        ]:
            selected = filters.get(field)
            if selected:
                row_val = norm(r.get(field))
                allowed_vals = [norm(v) for v in selected]
                if row_val not in allowed_vals:
                    passed = False
                    break

        if not passed:
            continue

        # Review Status
        if filters.get("Review_Status"):
            if r.get("Review_Status", "Not Reviewed") not in filters["Review_Status"]:
                continue

        # Numeric Range Filters
        numeric_specs = [
            ("Contract Period", "contract_min", "contract_max"),
            ("Estimated Bid Value", "est_min", "est_max"),
            ("Bid Offer Validity", "bov_min", "bov_max"),
            ("Minimum Average Annual Turnover", "turn_min", "turn_max"),
            ("Years of Past Experience Required", "exp_min", "exp_max")
        ]

        for col, min_key, max_key in numeric_specs:
            raw_val = r.get(col)
            try:
                val = int(re.findall(r"\d+", str(raw_val).replace(",", ""))[0])
            except:
                val = None

            min_v, max_v = filters[min_key], filters[max_key]

            if min_v is not None and (val is None or val < min_v):
                passed = False
                break
            if max_v is not None and (val is None or val > max_v):
                passed = False
                break

        if not passed:
            continue

        # Date Range Filters
        start_date = parse_date(r.get("Start_Date"))
        end_date = parse_date(r.get("End_Date"))

        if filters.get("start_date_range"):
            sd = filters["start_date_range"]
            if len(sd) == 2 and start_date:
                if sd[0] and start_date.date() < sd[0]:
                    continue
                if sd[1] and start_date.date() > sd[1]:
                    continue

        if filters.get("end_date_range"):
            ed = filters["end_date_range"]
            if len(ed) == 2 and end_date:
                if ed[0] and end_date.date() < ed[0]:
                    continue
                if ed[1] and end_date.date() > ed[1]:
                    continue

        result.append(r)

    return result
###############################################################
#  CHUNK 3 — Load Tender Context, Table Rendering, Review Update
###############################################################

# ------------------ Valid End-Date Filter ------------------
def filter_valid_end_date(rows):
    valid = []
    ignored = []
    now = datetime.now()

    for r in rows:
        dt = parse_date(r.get("End_Date"))
        if dt and dt >= now:
            valid.append(r)
        else:
            ignored.append(r)
    return valid, ignored


# ------------------ Prepare Context Text ------------------
def prepare_context(row):
    text = row.get("Bid_File_Text") or ""

    meta_fields = [
        "Bid_No", "Items", "Quantity", "Department", "Start_Date", "End_Date",
        "Organisation Name", "Ministry Name", "Office Name", "Contract Period",
        "Estimated Bid Value", "Item Category", "Bid Offer Validity",
        "Minimum Average Annual Turnover", "Years of Past Experience Required",
        "Past Experience of Similar Services", "Auto Extension Days",
        "Evaluation Method", "EMD Amount", "MII Compliance"
    ]

    meta_lines = []
    for f in meta_fields:
        if row.get(f):
            meta_lines.append(f"{f}: {row.get(f)}")

    # truncate extremely large text
    if len(text) > MAX_CTX_CHARS:
        text = text[:MAX_CTX_CHARS] + "\n...[TRUNCATED]"

    return "\n".join(meta_lines) + "\n\n" + text


# ------------------ Top-K Context using embeddings ------------------
def top_k_context(q_emb, rows, k=TOP_K):
    scored = []
    for r in rows:
        emb = safe_eb(r)
        if not emb:
            continue
        sim = cosine_similarity(q_emb, emb)
        r2 = dict(r)
        r2["_similarity"] = sim
        scored.append(r2)

    scored.sort(key=lambda x: x["_similarity"], reverse=True)
    return scored[:k]


# ------------------ Get Tender Links ------------------
def get_tender_links(row):
    bid_link = (
        row.get("Bid_File_Url")
        or row.get("Bid_File_Link")
        or row.get("Bid_File")
        or row.get("Bid Document URL")
        or ""
    )

    sow_link = (
        row.get("SOW_File_Url")
        or row.get("Criteria_File_Url")
        or row.get("Scope_Of_Work_File")
        or row.get("Scope_File_Url")
        or row.get("Scope Document URL")
        or ""
    )
    return bid_link, sow_link


# ------------------ Build Table DataFrame ------------------
def build_tender_dataframe(rows):
    records = []
    for r in rows:
        bid_link, sow_link = get_tender_links(r)

        raw_sow = r.get("Scope of Work PDF")
        raw_criteria = r.get("Criteria File Url")

        sow = raw_sow if (raw_sow and isinstance(raw_sow, str) and raw_sow.startswith("http")) else None
        criteria = raw_criteria if (raw_criteria and isinstance(raw_criteria, str) and raw_criteria.startswith("http")) else None

        status = r.get("Review_Status", "Not Reviewed")

        # Colored status badge
        status_badge = (
            f"🟩 {status}" if status == "Suitable" else
            f"🟥 {status}" if status == "Not Suitable" else
            f"🟦 {status}" if status == "Read" else
            f"🟨 {status}" if status == "To be Consumed in Depth" else
            f"🟪 {status}" if status == "Shared with Solutioning" else
            f"✔️ {status}" if status == "Done" else
            f"⚪ {status}"
        )

        records.append({
            "Bid No": r.get("Bid_No", ""),
            "Review Status": status_badge,
            "Items": r.get("Items", ""),
            "Quantity": r.get("Quantity", ""),
            "Department": r.get("Department", ""),
            "Start Date": r.get("Start_Date", ""),
            "End Date": r.get("End_Date", ""),
            "Estimated Bid Value": r.get("Estimated Bid Value", ""),
            "Bid File URL": bid_link or None,
            "Scope of Work PDF": sow,
            "Criteria File Url": criteria,
        })

    try:
        return pd.DataFrame(records)
    except:
        return records


# ------------------ Render Table in Chat ------------------
def show_tender_table(rows):
    df = build_tender_dataframe(rows)
    if df is None or (hasattr(df, "empty") and df.empty):
        st.warning("No tenders to display.")
        return

    try:
        st.data_editor(
            df,
            hide_index=True,
            column_config={
                "Bid File URL": st.column_config.LinkColumn("Bid File", display_text="Open Bid File"),
                "Scope of Work PDF": st.column_config.LinkColumn("Scope of Work PDF", display_text="Open SOW"),
                "Criteria File Url": st.column_config.LinkColumn("Criteria File", display_text="Open Criteria"),
            },
        )
    except:
        st.dataframe(df)


# ============================================================
#       REVIEW STATUS UPDATE (SIDEBAR SECTION)
# ============================================================

with st.sidebar:
    st.header("📝 Review / Categorize Tenders")

    bid_input = st.text_area(
        "Enter Bid No(s) (comma-separated)",
        placeholder="Example: GEM/2025/B/6930269, GEM/2025/B/6607455"
    )

    new_status = st.selectbox(
        "Select Review Status",
        ["Suitable", "Not Suitable", "Read", "To be Consumed in Depth", "Shared with Solutioning", "Done"]
    )

    if st.button("Update Review Status"):
        if not bid_input.strip():
            st.warning("⚠️ Please enter at least one Bid No.")
        else:
            bid_list = [b.strip() for b in bid_input.replace("\n", ",").split(",") if b.strip()]

            try:
                resp = (
                    supabase.table("GEM_TABLE_UPDATED")
                    .update({"Review_Status": new_status})
                    .in_("Bid_No", bid_list)
                    .execute()
                )
                updated = len(resp.data or [])
            except Exception as e:
                st.error(f"Error while updating: {e}")
                updated = 0

            if updated > 0:
                st.success(f"Updated {updated} tenders successfully!")

                # Refresh cached rows
                try:
                    raw_rows = supabase.table("GEM_TABLE_UPDATED").select("*").order("Bid_No").execute().data
                    rows = cached_apply_filters(raw_rows, filters)
                    valid_rows, _ = cached_valid_date(rows)
                    st.session_state.all_rows = valid_rows
                except Exception as e:
                    st.warning(f"Updated in database, but failed to refresh local cache: {e}")
            else:
                st.error("No matching Bid_No found.")


# ============================================================
#       LOAD TENDER CONTEXT (OPTIMIZED)
# ============================================================

if st.button("🔍 Load Tender Context"):
    with st.spinner("Fetching tenders matching filters..."):

        equality_filters = build_equality_filter(filters)

        raw_rows = cached_fetch_rows(equality_filters)

        filtered_rows = cached_apply_filters(raw_rows, filters)

        valid_rows, ignored = cached_valid_date(filtered_rows)

        st.session_state.all_rows = valid_rows

        if not valid_rows:
            st.error("No active tenders found.")
        else:
            q_emb = cached_embedding("Representative tender search query.")

            top_rows = top_k_context(q_emb, valid_rows, TOP_K)

            context_blocks = []
            for i, row in enumerate(top_rows, 1):
                context_blocks.append(
                    f"--- Tender {i} (Bid_No: {row.get('Bid_No')}, Sim={row['_similarity']:.3f}) ---\n"
                    + prepare_context(row)
                )

            st.session_state.context_text = "\n\n".join(context_blocks)

            st.success(f"Loaded context from {len(top_rows)} tenders!")
###############################################################
#  CHUNK 4 — Optimized Chat Engine + Final UI Assembly
###############################################################

# ------------------ AI Call ------------------
def ask_llm(messages):
    """
    Optimized LLM call:
    - Reduced overhead
    - Less repeated system context
    - Stable token usage
    - Perfectly backward compatible
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        return resp.choices[0].message.content
    except Exception:
        return "⚠️ The query exceeded the token limit. Please refine your question or apply more filters."


# ------------------ RENDER EXISTING CHAT MEMORY ------------------
for msg in st.session_state.messages:
    if msg.get("type") == "table":
        with st.chat_message("assistant"):
            st.markdown(f"### 📄 {msg['label']}")
            show_tender_table(msg["rows"])

    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ------------------ USER INPUT BOX ------------------
user_query = st.chat_input("Ask something about tenders...")

if user_query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Command keywords
    keywords = ["table", "list", "show tenders", "list tender"]

    # ------------------ TABLE COMMAND ------------------
    if any(k in user_query.lower() for k in keywords):

        table_id = len([m for m in st.session_state.messages if m.get("type") == "table"]) + 1

        # Store table as message
        st.session_state.messages.append({
            "role": "assistant",
            "type": "table",
            "rows": st.session_state.all_rows,
            "label": f"Tender Results #{table_id}"
        })

        with st.chat_message("assistant"):
            st.markdown(f"### Showing Tender Results #{table_id}")
            show_tender_table(st.session_state.all_rows)

    # ------------------ NORMAL LLM QUERY ------------------
    else:
        system_prompt = (
            "You are a helpful GEM tender assistant.\n"
            "You answer based ONLY on the tender context provided.\n"
            "If the answer is not in context, say you cannot determine it.\n"
            "Be concise and accurate.\n"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add all NON-table messages in correct order
        for m in st.session_state.messages:
            if m.get("type") != "table":
                messages.append(m)

        # Add tender context ONLY ONCE
        if st.session_state.context_text:
            messages.append({
                "role": "system",
                "content": f"TENDER CONTEXT:\n{st.session_state.context_text}"
            })

        # Get AI response
        ans = ask_llm(messages)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(ans)

        # Save to message history
        st.session_state.messages.append({"role": "assistant", "content": ans})

else:
    st.info("⚙️ Apply filters and click **Load Tender Context** to begin.")

