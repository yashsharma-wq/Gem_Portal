# Gem_updated.py
"""
Streamlit app: GEM Tender Smart Chatbot (Supabase + Vector Search + LLM)
Updated: Added Email–Password Login using Supabase Authentication
"""

import re
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from supabase import create_client
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
# ------------------ PAGE CONFIG ----------------
st.set_page_config(
    page_title="GEM Tender Smart Chatbot",
    layout="wide"
)

# ------------------------- Load Environment Variables -------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    st.error("Missing SUPABASE_URL, SUPABASE_KEY, or OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------- Authentication Helpers -------------------------
if "user" not in st.session_state:
    st.session_state.user = None

def login_user(email, password):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return res
    except Exception:
        return None

def logout_user():
    st.session_state.user = None
    st.rerun()


# ------------------------- Login Screen -------------------------
if st.session_state.user is None:

    st.title("🔐 Login to Access GEM Tender Chatbot")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(email, password)
        if user and user.user:
            st.session_state.user = user
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid email or password")

    st.stop()  # STOP execution until login is done


# ------------------------- Sidebar Logout -------------------------
with st.sidebar:
    st.success(f"Logged in as: {st.session_state.user.user.email}")
    if st.button("Logout"):
        logout_user()


# ------------------------- Constants -------------------------
EMBEDDING_MODEL = "text-embedding-3-small"

Total=supabase.table('GEM_TABLE_UPDATED').select('*',count='exact').execute()
TOP_K_INTERNAL = Total.count
MAX_CONTEXT_CHARS = 30000
DATE_FORMATS = ["%d-%m-%Y %I:%M %p", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]


# ------------------------- Helper Functions -------------------------
def parse_date(date_str: str):
    if not date_str or not isinstance(date_str, str):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None

def get_embedding(text: str):
    if not text:
        return []
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def cosine_similarity(a, b):
    a_arr, b_arr = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return -1
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def get_distinct_values(col):
    try:
        resp = supabase.table("GEM_TABLE_UPDATED").select(f'"{col}"').execute()
        vals = sorted({r[col] for r in resp.data if r.get(col)})
        return [""] + vals
    except Exception as e:
        st.warning(f"Failed loading values for {col}: {e}")
        return [""]

def build_filter_query(filters):
    mapping = {
        "Ministry Name": "Ministry Name",
        "Department Name": "Department Name",
        "Organisation Name": "Organisation Name",
        "Office Name": "Office Name",
        "Contract Period": "Contract Period",
        "Estimated Bid Value": "Estimated Bid Value",
        "Item Category": "Item Category",
        "Bid Offer Validity": "Bid Offer Validity",
        "Minimum Average Annual Turnover": "Minimum Average Annual Turnover",
        "Years of Past Experience Required": "Years of Past Experience Required",
        "MII Compliance": "MII Compliance",
    }
    q = {}
    for k, col in mapping.items():
        v = filters.get(k)
        if v not in ("", None):
            q[col] = v
    return q

def fetch_all_rows(filters):
    equality_filters = build_filter_query(filters)

    query = supabase.table("GEM_TABLE_UPDATED").select("*")
    for col, val in equality_filters.items():
        query = query.eq(col, val)

    resp = query.execute()
    return resp.data or []

def filter_valid_end_date(rows):
    valid, ignored = [], []
    now = datetime.now()
    for r in rows:
        dt = parse_date(r.get("End_Date"))
        if dt and dt >= now:
            valid.append(r)
        else:
            ignored.append(r)
    return valid, ignored

def safe_eb(row):
    emb = row.get("Bid_Embedding")
    if emb is None:
        return []
    if isinstance(emb, list):
        return [float(x) for x in emb]
    if isinstance(emb, str):
        try:
            parsed = json.loads(emb)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except:
            pass
    return []

def top_k_context(q_emb, rows, k=TOP_K_INTERNAL):
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

def prepare_context(row):
    text = row.get("Bid_File_Text") or ""
    meta_lines = []
    fields = [
        "Bid_No", "Items", "Quantity", "Department", "Start_Date", "End_Date",
        "Organisation Name", "Ministry Name", "Office Name", "Contract Period",
        "Estimated Bid Value", "Item Category", "Bid Offer Validity",
        "Minimum Average Annual Turnover", "Years of Past Experience Required",
        "Past Experience of Similar Services", "Auto Extension Days",
        "Evaluation Method", "EMD Amount", "MII Compliance"
    ]
    for f in fields:
        if row.get(f):
            meta_lines.append(f"{f}: {row.get(f)}")

    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS] + "\n[TRUNCATED]"

    return "\n".join(meta_lines) + "\n\n" + text

def ask_llm(messages):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        return resp.choices[0].message.content
    except Exception:
        return 'The query exceeded the token limit.Please select more filters'


# ------------------------- Utility Functions -------------------------
def to_int(val):
    if val is None:
        return None
    s = str(val)
    nums = re.findall(r"\d+", s.replace(",", ""))
    if not nums:
        return None
    try:
        return int(nums[0])
    except:
        return None

def in_range(value, min_val, max_val):
    if value is None:
        return True
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True

# ------------------------- Table / Display Helpers -------------------------
def get_tender_links(row: Dict[str, Any]):
    """Pick best-guess columns for bid file and scope/criteria URLs."""
    bid_link = (
        row.get("Bid_File_Url")
        or row.get("Bid_File_Link")
        or row.get("Bid_File")
        or row.get("Bid Document URL")
        or ""
    )
    scope_link = (
        row.get("SOW_File_Url")
        or row.get("Criteria_File_Url")
        or row.get("Scope_Of_Work_File")
        or row.get("Scope_File_Url")
        or row.get("Scope Document URL")
        or ""
    )
    return bid_link, scope_link


def build_tender_dataframe(rows: List[Dict[str, Any]]):
    """Build dataframe for display with clickable URL columns."""
    records = []
    for r in rows:
        bid_link, _ = get_tender_links(r)
        raw_sow = r.get("Scope of Work PDF")
        raw_criteria = r.get("Criteria File Url")

        # Convert empty / invalid values to None
        sow = raw_sow if (raw_sow and isinstance(raw_sow, str) and raw_sow.startswith("http")) else None
        criteria = raw_criteria if (raw_criteria and isinstance(raw_criteria, str) and raw_criteria.startswith("http")) else None

        status = r.get("Review_Status", "Not Reviewed")

        # add a visual indicator
        status_badge = f"🟩 {status}" if status == "Suitable" else \
                    f"✔️ {status}" if status == "Done" else \
                    f"🟥 {status}" if status == "Not Suitable" else \
                    f"🟦 {status}" if status == "Read" else \
                    f"🟨 {status}" if status == "To be Consumed in Depth" else \
                    f"🟪 {status}" if status == "Shared with Solutioning" else \
                    f"⚪ {status}"

       

        records.append({
            "Bid No": r.get("Bid_No", ""),
            "Review Status": status_badge,   # NEW HIGHLIGHT LABEL
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
        import pandas as pd  # local import in case global fails
        return pd.DataFrame(records)
    except Exception:
        return records  # graceful fallback


def show_tender_table(rows: List[Dict[str, Any]]):
    """Render tenders with clickable links using Streamlit's data_editor + LinkColumn."""
    df = build_tender_dataframe(rows)
    if df is None or (hasattr(df, "empty") and df.empty):
        st.warning("No tenders loaded.")
        return

    try:
        st.data_editor(
            df,
            hide_index=True,
            column_config={
                "Bid File URL": st.column_config.LinkColumn(
                    "Bid File",
                    display_text="Open Bid File",
                ),
                "Scope of Work PDF": st.column_config.LinkColumn(
                    "Scope of Work PDF",
                    display_text="Open SOW",
                ),
                "Criteria File Url": st.column_config.LinkColumn(
                    "Criteria File",
                    display_text="Open Criteria",
                ),
            },
        )
    except Exception:
        st.dataframe(df)




# ------------------------- Advanced Filters -------------------------
def apply_advanced_filters(rows, filters):

    
    def norm(x):
        return str(x).strip().lower() if x else ""

    MISSING_TOKENS = {"not found", "notavailable", "not available",
                      "na", "n/a", "-", "none", "not provided", "tbd"}

    result = []

    for r in rows:

        if filters.get("remove_est_notfound"):
            est_raw = str(r.get("Estimated Bid Value", "")).strip().lower()
            if not any(ch.isdigit() for ch in est_raw):
                continue

        passed = True

        for field in [
    "Ministry Name",
    "Department Name",
    "Organisation Name",
    "Office Name",
    "Item Category",
    "MSE Exemption for Years of Experience and Turnover",
    "Startup Exemption for Years of Experience and Turnover"
]:

            selected = filters.get(field)
            if selected:
                row_val = norm(r.get(field))
                allowed_vals = [norm(v) for v in selected]
                if row_val not in allowed_vals:
                    passed = False
                    break
                # Review Status filtering
        if filters.get("Review_Status"):
            if r.get("Review_Status", "Not Reviewed") not in filters["Review_Status"]:
                continue

        if not passed:
            continue

        checks = [
            ("Contract Period", "contract_min", "contract_max", False),
            ("Estimated Bid Value", "est_min", "est_max", True),
            ("Bid Offer Validity", "bov_min", "bov_max", False),
            ("Minimum Average Annual Turnover", "turn_min", "turn_max", False),
            ("Years of Past Experience Required", "exp_min", "exp_max", False)
        ]

        for col, min_key, max_key, is_est_val in checks:
            raw_val = r.get(col)
            min_v = filters[min_key]
            max_v = filters[max_key]

            if is_est_val:
                if (min_v is not None or max_v is not None):
                    if str(raw_val).strip().lower() == "not found":
                        passed = False
                        break

            val = to_int(raw_val)

            if not in_range(val, min_v, max_v):
                passed = False
                break

        if not passed:
            continue

        start_date = parse_date(r.get("Start_Date"))
        end_date = parse_date(r.get("End_Date"))

        if filters["start_date_range"]:
            sd_min, sd_max = filters["start_date_range"]
            if start_date:
                if sd_min and start_date.date() < sd_min:
                    continue
                if sd_max and start_date.date() > sd_max:
                    continue

        if filters["end_date_range"]:
            ed_min, ed_max = filters["end_date_range"]
            if end_date:
                if ed_min and end_date.date() < ed_min:
                    continue
                if ed_max and end_date.date() > ed_max:
                    continue

        result.append(r)

    return result


# ------------------------- Streamlit App -------------------------
# st.set_page_config(page_title="GEM Tender AI Assistant", layout="wide")
st.title("💬 GEM Tender Chat Assistant")


# ------------------------- Sidebar Filters -------------------------
with st.sidebar:
    st.header("🎯 Apply Filters")

    ministry = st.multiselect("Ministry Name", get_distinct_values("Ministry Name"))
    dept = st.multiselect("Department Name", get_distinct_values("Department Name"))
    org = st.multiselect("Organisation Name", get_distinct_values("Organisation Name"))
    office = st.multiselect("Office Name", get_distinct_values("Office Name"))
    item_category = st.multiselect("Item Category", get_distinct_values("Item Category"))
    mse_exemption = st.multiselect(
        "MSE Exemption for Years of Experience and Turnover",
        get_distinct_values("MSE Exemption for Years of Experience and Turnover")
    )

    startup_exemption = st.multiselect(
        "Startup Exemption for Years of Experience and Turnover",
        get_distinct_values("Startup Exemption for Years of Experience and Turnover")
    )
    
    review_status_filter = st.multiselect(
        "Review Status",
        ["Not Reviewed", "Suitable", "Not Suitable", "Read",
        "To be Consumed in Depth", "Shared with Solutioning","Done"]
    )
    # filters["Review_Status"] = review_status_filter


    remove_est_notfound = st.checkbox("Remove tenders with Estimated Bid Value = Not found", value=False)

    contract_min = st.text_input("Contract Period (Min)")
    contract_max = st.text_input("Contract Period (Max)")

    est_min = st.text_input("Estimated Bid Value (Min)")
    est_max = st.text_input("Estimated Bid Value (Max)")

    bov_min = st.text_input("Bid Offer Validity (Min)")
    bov_max = st.text_input("Bid Offer Validity (Max)")

    turn_min = st.text_input("Minimum Avg Annual Turnover (Min)")
    turn_max = st.text_input("Minimum Avg Annual Turnover (Max)")

    exp_min = st.text_input("Years of Past Experience Required (Min)")
    exp_max = st.text_input("Years of Past Experience Required (Max)")

    start_date_range = st.date_input("Start Date Range", [],help="Both FROM and TO dates are required for Start Date.")
    end_date_range = st.date_input("End Date Range", [],help="Both FROM and TO dates are required for End Date.")

    mii_compliance = st.selectbox("MII Compliance", get_distinct_values("MII Compliance"))

if len(end_date_range)==1:
    st.warning("⚠️ Please select BOTH FROM and TO date in End Date Range.")
    st.stop()
if len(start_date_range)==1:
    st.warning("⚠️ Please select BOTH FROM and TO date in Start Date Range.")
    st.stop()

filters = {
    "Ministry Name": ministry,
    "Department Name": dept,
    "Organisation Name": org,
    "Office Name": office,
    "Item Category": item_category,
    "MSE Exemption for Years of Experience and Turnover": mse_exemption,
    "Startup Exemption for Years of Experience and Turnover": startup_exemption,


    "contract_min": to_int(contract_min),
    "contract_max": to_int(contract_max),

    "est_min": to_int(est_min),
    "est_max": to_int(est_max),

    "bov_min": to_int(bov_min),
    "bov_max": to_int(bov_max),

    "turn_min": to_int(turn_min),
    "turn_max": to_int(turn_max),

    "exp_min": to_int(exp_min),
    "exp_max": to_int(exp_max),

    "start_date_range": start_date_range,
    "end_date_range": end_date_range,

    "MII Compliance": mii_compliance,
    "remove_est_notfound": remove_est_notfound,
    "Review_Status":review_status_filter,
}


st.sidebar.header("📝 Review / Categorize Tenders")

bid_input = st.sidebar.text_area(
    "Enter Bid No(s) (comma-separated)",
    placeholder="Example: GEM/2025/B/6930269, GEM/2025/B/6607455"
)

new_status = st.sidebar.selectbox(
    "Select Review Status",
    [
        "Suitable",
        "Not Suitable",
        "Read",
        "To be Consumed in Depth",
        "Shared with Solutioning",
        "Done",
    ]
)

if st.sidebar.button("Update Review Status"):
    if not bid_input.strip():
        st.sidebar.warning("⚠️ Please enter at least one Bid No.")
    else:
        # Clean and split bid numbers
        bid_list = [b.strip() for b in bid_input.replace("\n", ",").split(",") if b.strip()]
        if not bid_list:
            st.sidebar.warning("⚠️ No valid Bid No found after parsing. Please check the format.")
        else:
            # Batch update using IN clause so that all bid numbers are updated
            try:
                resp = (
                    supabase
                    .table("GEM_TABLE_UPDATED")
                    .update({"Review_Status": new_status})
                    .in_("Bid_No", bid_list)
                    .execute()
                )
                updated_count = len(resp.data or [])
            except Exception as e:
                updated_count = 0
                st.sidebar.error(f"❌ Error while updating review status: {e}")

            if updated_count > 0:
                st.sidebar.success(f"🎯 Updated {updated_count} tender(s) successfully!")

                # 🔄 Refresh in-memory rows so that next 'table/list' query sees latest statuses
                try:
                    with st.spinner("Refreshing tender data with latest review status..."):
                        raw_rows = supabase.table("GEM_TABLE_UPDATED").select("*").order("Bid_No").execute().data
                        rows = apply_advanced_filters(raw_rows, filters)
                        valid_rows, ignored = filter_valid_end_date(rows)
                        st.session_state.all_rows = valid_rows
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Updated in Supabase, but failed to refresh local table cache: {e}")
            else:
                st.sidebar.error("❌ No valid Bid_No matched. Please check your input.")


# ------------------------- Chat State -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_text" not in st.session_state:
    st.session_state.context_text = ""
if "all_rows" not in st.session_state:
    st.session_state.all_rows = []


# ------------------------- Load Context Trigger -------------------------
if st.button("🔍 Load Tender Context"):
    with st.spinner("Fetching ALL matching tenders..."):
        # raw_rows = supabase.table("GEM_TABLE_UPDATED").select("*").execute().data
        raw_rows = supabase.table("GEM_TABLE_UPDATED").select("*").order("Bid_No").execute().data
        st.session_state.all_rows = raw_rows  
        rows = apply_advanced_filters(raw_rows, filters)

    valid_rows, ignored = filter_valid_end_date(rows)

    if not valid_rows:
        st.error("No active tenders found.")
    else:
        st.session_state.all_rows = valid_rows

        q_emb = get_embedding("Representative tender search query.")
        top_rows = top_k_context(q_emb, valid_rows, TOP_K_INTERNAL)

        context_blocks = []
        for i, row in enumerate(top_rows, start=1):
            ctx = prepare_context(row)
            context_blocks.append(
                f"--- Tender #{i} (Bid_No: {row.get('Bid_No')}, Sim={row['_similarity']:.3f}) ---\n{ctx}"
            )

        st.session_state.context_text = "\n\n".join(context_blocks)
        st.success(f"Loaded context from {len(top_rows)} tenders. Start chatting!")


# ------------------------- Chatbox -------------------------
# ---- Persistent Table Memory ----
if "show_table" not in st.session_state:
    st.session_state.show_table = False
if "table_rows" not in st.session_state:
    st.session_state.table_rows = []

# ---- Render Chat History ----
for msg in st.session_state.messages:
    if msg.get("type") == "table":
        with st.chat_message("assistant"):
            st.markdown(f"### 📄 {msg['label']}")
            show_tender_table(msg["rows"])
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ---- User Input ----
user_query = st.chat_input("Ask something...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    keywords = ["table", "list", "list tender", "show tenders"]

    if any(k in user_query.lower() for k in keywords):

        # Store table as chat message
        table_id = len([m for m in st.session_state.messages if m.get("type") == "table"]) + 1

        st.session_state.messages.append({
            "role": "assistant",
            "type": "table",
            "rows": st.session_state.all_rows,
            "label": f"Tender Results #{table_id}"
        })

        with st.chat_message("assistant"):
            st.markdown(f"Showing Tender Results #{table_id}")
            show_tender_table(st.session_state.all_rows)

    else:
        # Normal AI response
        system_prompt = "You are a helpful tender assistant..."
        messages = [{"role": "system", "content": system_prompt}]

        for m in st.session_state.messages:
            if m.get("type") != "table":
                messages.append(m)

        messages.append({"role": "system", "content": f"Tender Context:\n{st.session_state.context_text}"})

        with st.chat_message("assistant"):
            ans = ask_llm(messages)
            st.markdown(ans)

        st.session_state.messages.append({"role": "assistant", "content": ans})


else:
    st.info("⚙️ Apply filters and click **Load Tender Context** to begin.")
