# GEM_BOT_3_conversational.py
# Streamlit conversational chatbot for GEM_SERVICE_TABLE (filter + semantic search + LLM summary + follow-ups)
# Requirements:
#   pip install streamlit supabase openai
#   Set env vars: SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY
# Run: streamlit run GEM_BOT_3_conversational.py

import os
import re
from typing import List, Dict, Any, Tuple
import json
import streamlit as st
from supabase import create_client
from openai import OpenAI

# -------------------------------
# Streamlit configuration - MUST be the first Streamlit call
st.set_page_config(page_title="GEM Service Tender Chatbot (Conversational)", page_icon="🤖", layout="wide")

# -------------------------------
# Config (env vars)
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))


if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    st.error("Please set SUPABASE_URL, SUPABASE_KEY and OPENAI_API_KEY environment variables.")
    st.stop()

# OpenAI client (new API surface)
client = OpenAI(api_key=OPENAI_API_KEY)

# Embedding models - primary and fallback
EMBED_MODEL_PRIMARY = "text-embedding-3-small"   # 1536 dims
EMBED_MODEL_FALLBACK = "text-embedding-3-large"  # 3072 dims

TABLE_NAME = "GEM_SERVICE_TABLE"

# Supabase client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Helpers
def parse_numeric(text: str) -> float:
    if not text:
        return None
    s = str(text)
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "":
        return None
    try:
        return float(s)
    except:
        return None

def detect_filters_from_query(q: str) -> Dict[str, Any]:
    ql = q.lower()
    filters = {}
    m = re.search(r"(?:bid value|estimated value|value|estimated)\s*(?:is\s*)?(?:greater than|more than|above|>)\s*([0-9,]+)", ql)
    if m:
        filters["estimated_value_gt"] = parse_numeric(m.group(1))
    m = re.search(r"(?:bid value|estimated value|value|estimated)\s*(?:less than|below|<)\s*([0-9,]+)", ql)
    if m:
        filters["estimated_value_lt"] = parse_numeric(m.group(1))
    m = re.search(r"more than\s*([0-9,]+)", ql)
    if m and ("bid" in ql or "value" in ql or "estimated" in ql):
        filters.setdefault("estimated_value_gt", parse_numeric(m.group(1)))
    m = re.search(r"([0-9,]+)\s*(?:or more|\+|and above)", ql)
    if m and ("bid" in ql or "value" in ql or "estimated" in ql):
        filters.setdefault("estimated_value_gt", parse_numeric(m.group(1)))
    m = re.search(r"department[: ]+\s*([a-zA-Z0-9 &-]+)", q, re.IGNORECASE)
    if m:
        filters["Department"] = m.group(1).strip()
    m = re.search(r"ministry[: ]+\s*([a-zA-Z0-9 &-]+)", q, re.IGNORECASE)
    if m:
        filters["Ministry Name"] = m.group(1).strip()
    m = re.search(r"(GEM/[0-9A-Za-z\-/]+)", q)
    if m:
        filters["Bid_No"] = m.group(1)
    m_after = re.search(r"after\s*([0-9]{2,4}[-/][0-9]{1,2}[-/][0-9]{1,4})", q)
    m_before = re.search(r"before\s*([0-9]{2,4}[-/][0-9]{1,2}[-/][0-9]{1,4})", q)
    if m_after:
        filters["date_after"] = m_after.group(1)
    if m_before:
        filters["date_before"] = m_before.group(1)
    return filters

def decide_search_strategy(q: str) -> Tuple[str, Dict[str, Any]]:
    filters = detect_filters_from_query(q)
    if filters:
        return "filter", filters
    tokens = q.split()
    if len(tokens) >= 2:
        return "semantic", {}
    return "semantic", {}

# Embedding helpers using new OpenAI client
def make_embedding_force(model: str, text: str):
    resp = client.embeddings.create(model=model, input=text)
    embedding = resp.data[0].embedding
    return embedding, model, len(embedding)

def make_embedding_with_primary(text: str):
    return make_embedding_force(EMBED_MODEL_PRIMARY, text)

# Structured (filter) search - lightweight client-side
def fetch_by_filters(filters: Dict[str, Any], limit: int = 1000) -> List[Dict[str, Any]]:
    r = sb.table(TABLE_NAME).select("*").limit(limit).execute()
    data = None
    if isinstance(r, dict) and "data" in r:
        data = r["data"]
    elif hasattr(r, "data"):
        data = r.data
    else:
        data = r
    if not data:
        return []
    out = []
    for row in data:
        ok = True
        if "estimated_value_gt" in filters:
            val = parse_numeric(row.get("Estimated Bid Value"))
            if val is None or val <= filters["estimated_value_gt"]:
                ok = False
        if "estimated_value_lt" in filters:
            val = parse_numeric(row.get("Estimated Bid Value"))
            if val is None or val >= filters["estimated_value_lt"]:
                ok = False
        if "Department" in filters:
            if not row.get("Department") or filters["Department"].lower() not in row.get("Department","").lower():
                ok = False
        if "Ministry Name" in filters:
            if not row.get("Ministry Name") or filters["Ministry Name"].lower() not in row.get("Ministry Name","").lower():
                ok = False
        if "Bid_No" in filters:
            if row.get("Bid_No") != filters["Bid_No"]:
                ok = False
        if ok:
            out.append(row)
    return out

# Semantic search (vector) via RPC match_tenders
def semantic_search(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    emb, model_name, dim = make_embedding_with_primary(query)
    def call_rpc(embedding):
        payload = {"match_count": top_k, "query_embedding": embedding}
        r = sb.rpc("match_tenders", payload).execute()
        if isinstance(r, dict) and "data" in r:
            return r["data"]
        elif hasattr(r, "data"):
            return r.data
        else:
            return r
    try:
        return call_rpc(emb)
    except Exception as e:
        err = str(e).lower()
        # try fallback embedding model if dims mismatch
        if "different vector dimensions" in err or ("1536" in err and "3072" in err):
            try:
                emb2, model2, dim2 = make_embedding_force(EMBED_MODEL_FALLBACK, query)
                return call_rpc(emb2)
            except Exception:
                # fallback RPC name
                try:
                    payload = {"match_count": top_k, "query_embedding": emb}
                    r2 = sb.rpc("match_gem_tender_data", payload).execute()
                    if isinstance(r2, dict) and "data" in r2:
                        return r2["data"]
                    elif hasattr(r2, "data"):
                        return r2.data
                    else:
                        return r2
                except Exception as e3:
                    st.error(f"Semantic search error: {e3}")
                    return []
        else:
            # try alternative RPC once
            try:
                payload = {"match_count": top_k, "query_embedding": emb}
                r3 = sb.rpc("match_gem_tender_data", payload).execute()
                if isinstance(r3, dict) and "data" in r3:
                    return r3["data"]
                elif hasattr(r3, "data"):
                    return r3.data
                else:
                    return r3
            except Exception as e4:
                st.error(f"Semantic search RPC failed: {e4}")
                return []

# UI presentation helpers
def get_file_urls(row: Dict[str, Any]):
    urls = []
    for c in ["Bid_File", "Scope of Work PDF", "Criteria File Url"]:
        v = row.get(c)
        if v and isinstance(v, str) and v.strip().lower() not in ("not found","na","n/a",""):
            urls.append((c, v.strip()))
    return urls

# Use LLM to create a concise summary of results and suggested follow-ups
def generate_summary_with_llm(query: str, rows: List[Dict[str, Any]], sample_n: int = 6) -> str:
    sample_rows = rows[:sample_n]
    # Build a compact context for the LLM
    sample_texts = []
    for r in sample_rows:
        sample_texts.append({
            "Bid_No": r.get("Bid_No", "-"),
            "Department": r.get("Department", "-"),
            "Estimated_Bid_Value": r.get("Estimated Bid Value", "-"),
            "Bid_End_Date": r.get("Bid End Date", "-")
        })
    system_prompt = (
        "You are a helpful assistant that summarizes search results for procurement tenders. "
        "Given a user query and a small sample of matching tender rows, produce a short (3-6 sentence) summary describing:\n"
        " - how many tenders matched in total,\n"
        " - top departments or organizations found (brief),\n"
        " - estimated value range if available,\n"
        " - one-line recommended next actions or follow-up questions the user might ask.\n"
        "Answer concisely and in plain language."
    )
    user_prompt = f"User query: {query}\nTotal matches: {len(rows)}\nSample rows: {json.dumps(sample_texts, ensure_ascii=False)}\nProduce the requested short summary and 2 suggested follow-up questions."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    # New OpenAI client chat completions returns choices[0].message.content
    try:
        summary = resp.choices[0].message.content.strip()
    except Exception:
        # fallback: try older attribute
        summary = getattr(resp.choices[0], "message", {}).get("content", "")
    return summary

# Follow-up question handler: uses conversation history + last results context
def answer_followup_with_llm(user_msg: str, history: List[Dict[str,str]], last_summary: str, last_rows: List[Dict[str,Any]]) -> str:
    # Provide a compact context to the LLM (summary + optional 1-2 rows examples)
    sample_rows = last_rows[:4]
    sample_text = []
    for r in sample_rows:
        sample_text.append({
            "Bid_No": r.get("Bid_No"),
            "Department": r.get("Department"),
            "Estimated_Bid_Value": r.get("Estimated Bid Value"),
            "Bid_End_Date": r.get("Bid End Date"),
            "ShortDesc": (r.get("Bid_File_Text") or "")[:200].replace("\n"," ")  # first 200 chars
        })
    system_prompt = (
        "You are a conversational expert assistant that answers follow-up questions about a set of procurement tenders. "
        "Use the provided summary and sample rows to answer the user's follow-up question. If the user asks for details about a specific Bid_No, find it in the sample rows or say you can show the full list (use the UI). Keep answers helpful and concise."
    )
    # Build messages
    messages = [{"role":"system", "content": system_prompt}]
    messages.append({"role":"user", "content": f"Context summary: {last_summary}"})
    messages.append({"role":"user", "content": f"Sample rows: {json.dumps(sample_text, ensure_ascii=False)}"})
    # append previous conversation for continuity
    for turn in history[-6:]:
        role = "user" if turn.get("from") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("text")})
    messages.append({"role":"user", "content": user_msg})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=400
    )
    try:
        ans = resp.choices[0].message.content.strip()
    except Exception:
        ans = getattr(resp.choices[0], "message", {}).get("content", "")
    return ans

def present_result_summary_and_ui(rows: List[Dict[str,Any]], query: str):
    # store rows in session state for follow-ups
    st.session_state['last_rows'] = rows
    st.session_state['last_query'] = query

    # generate an LLM summary (non-blocking short call)
    with st.spinner("Generating summary..."):
        try:
            summary = generate_summary_with_llm(query, rows, sample_n=6)
        except Exception as e:
            summary = f"(Summary failed: {e})"

    st.session_state['last_summary'] = summary

    st.markdown("### Search result")
    st.success(f"Found **{len(rows)}** tender(s) matching your query.")
    st.markdown("**Summary:**")
    st.markdown(summary)

    # small list of top hits (compact)
    st.markdown("**Top matches (compact):**")
    for idx, r in enumerate(rows[:5], start=1):
        st.markdown(f"{idx}. **{r.get('Bid_No','-')}** — {r.get('Department','-')} — {r.get('Estimated Bid Value','-')} — End: {r.get('Bid End Date','-')}")

    # Dropdown to select a single row for detail
    bid_options = [r.get("Bid_No","-") or "-" for r in rows]
    st.markdown("---")
    col1, col2 = st.columns([2,1])
    with col1:
        sel = st.selectbox("Select a tender to view full details (or choose 'Show all results')", options=["-- Select --", "Show all results"] + bid_options, key="select_tender")
    with col2:
        show_files = st.checkbox("Show file URLs in details", value=True)

    if sel == "Show all results":
        with st.expander("All results (table)"):
            # convert to list of dicts -> display
            try:
                # choose visible columns for compactness
                display_cols = ["Bid_No","Department","Estimated Bid Value","EMD Amount","Contract Period","Bid End Date","Bid Opening Date"]
                # prepare rows list of dict with guaranteed keys
                table_rows = []
                for r in rows:
                    row_dict = {c: r.get(c, "") for c in display_cols}
                    table_rows.append(row_dict)
                st.dataframe(table_rows, use_container_width=True)
            except Exception as e:
                st.write("Could not render table:", e)

    elif sel and sel != "-- Select --":
        # find row
        matched = next((r for r in rows if r.get("Bid_No") == sel), None)
        if matched:
            st.markdown(f"### Details for **{sel}**")
            st.markdown(f"- **Department:** {matched.get('Department','-')}")
            st.markdown(f"- **Estimated Value:** {matched.get('Estimated Bid Value','-')}")
            st.markdown(f"- **EMD Amount:** {matched.get('EMD Amount','-')}")
            st.markdown(f"- **Contract Period:** {matched.get('Contract Period','-')}")
            st.markdown(f"- **Bid End Date:** {matched.get('Bid End Date','-')}")
            st.markdown(f"- **Bid Opening Date:** {matched.get('Bid Opening Date','-')}")
            files = get_file_urls(matched)
            if show_files and files:
                st.markdown("**Files:**")
                for label, url in files:
                    st.write(f"- {label}: {url}")
            with st.expander("Show extracted Bid_File_Text"):
                st.text_area("Bid_File_Text", value=matched.get("Bid_File_Text") or "Not available", height=300)
        else:
            st.info("Selected tender not found in the current result set.")

# -------------------------------
# UI layout and conversation memory
st.title("GEM Service Tender Chatbot — Conversational")
st.write("Ask about tenders. The assistant chooses filter vs semantic search, summarizes results with an LLM, and keeps short session memory for follow-ups.")

if "history" not in st.session_state:
    st.session_state.history = []         # list of {"from":"user"/"assistant", "text": "..."}
if "last_rows" not in st.session_state:
    st.session_state.last_rows = []
if "last_summary" not in st.session_state:
    st.session_state.last_summary = "(no summary yet)"

# Query input (main search)
with st.form("search_form"):
    query = st.text_input("Enter your question about tenders (e.g. 'tenders > 100000' or 'HPCL refractory works')", key="main_query")
    submitted = st.form_submit_button("Search")
    if submitted and query:
        st.session_state.history.append({"from":"user", "text": query})
        strategy, filters = decide_search_strategy(query)
        st.write(f"Detected strategy: **{strategy}**")
        if strategy == "filter":
            with st.spinner("Filtering..."):
                rows = fetch_by_filters(filters, limit=1000)
                if not rows:
                    st.info("No rows matched your filters.")
                    st.session_state.last_rows = []
                    st.session_state.last_summary = "No matches found."
                else:
                    present_result_summary_and_ui(rows, query)
        elif strategy == "semantic":
            with st.spinner("Running semantic search..."):
                top_k = 12 if len(query.split()) > 8 else 8
                rows = semantic_search(query, top_k=top_k)
                if not rows:
                    st.info("No semantic matches found. Try rephrasing or add explicit filters (e.g. 'tenders > 100000').")
                    st.session_state.last_rows = []
                    st.session_state.last_summary = "No matches found."
                else:
                    present_result_summary_and_ui(rows, query)
        else:
            st.error("Unknown strategy detected.")

# Conversation / follow-up area
st.markdown("---")
st.markdown("### Chat & follow-ups")
st.write("Ask follow-up questions about the last search (for example: 'Summarize the top 3', 'Which are closing within 7 days?', 'Show details for GEM/2025/B/xxxxx').")

colA, colB = st.columns([4,1])
with colA:
    followup = st.text_input("Your follow-up message", key="followup_input")
with colB:
    if st.button("Send follow-up"):
        if not followup:
            st.info("Type a follow-up message first.")
        else:
            st.session_state.history.append({"from":"user", "text": followup})
            # call LLM with last summary + last_rows context
            if not st.session_state.last_rows:
                resp_text = "No previous search context. Please perform a search first."
            else:
                try:
                    resp_text = answer_followup_with_llm(followup, st.session_state.history, st.session_state.last_summary, st.session_state.last_rows)
                except Exception as e:
                    resp_text = f"(LLM follow-up failed: {e})"
            st.session_state.history.append({"from":"assistant", "text": resp_text})
            st.success("Answer:")
            st.write(resp_text)

# show recent conversation history (compact)
if st.session_state.history:
    st.markdown("---")
    st.markdown("**Conversation (last turns):**")
    for turn in st.session_state.history[-10:]:
        who = "You" if turn["from"] == "user" else "Assistant"
        st.markdown(f"- **{who}:** {turn['text']}")

# Sidebar: quick controls and last summary
st.sidebar.markdown("### Session")
st.sidebar.write(f"Last query: {st.session_state.get('last_query','(none)')}")
st.sidebar.write(f"Last summary (snippet):")
st.sidebar.text_area("summary", value=(st.session_state.get('last_summary') or "")[:600], height=150)
st.sidebar.markdown("---")
st.sidebar.write("Note: Use 'Select a tender' dropdown to view details or 'Show all results' to expand the full table.")

