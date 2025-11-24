import fitz
import pytesseract
from PIL import Image
import requests
import io
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client=OpenAI(api_key=OPENAI_API_KEY)


def extract_criteria_using_llm(raw_text):
    prompt = f"""
You are an expert tender analyst. 
Analyze the following text extracted from a government tender document.

Your job is to extract ONLY these two things and nothing else:

1. Eligibility Criteria
2. Important Qualification Information

Rules:
- Use bullet points.
- Keep it short, clean, factual.
- Remove repetition, page numbers, table formatting, or irrelevant policy text.
- If information is missing, write "Not mentioned".

Here is the extracted text:
------------------------------------
{raw_text}
------------------------------------

Now extract clean, structured information.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content



def extract_text_from_pdf(pdf_path, use_ocr=True, save_txt=True):

    text = ""
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")

        if not page_text.strip() and use_ocr:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)

        text += "\n" + page_text

    doc.close()

    norm_text = re.sub(r"[ \t]+", " ", text)

    dis = re.search(r"\bDisclaimer\b", norm_text, re.IGNORECASE)
    limited_text = norm_text[:dis.start()] if dis else norm_text

    patterns = {
        "Bid End Date": r"Bid End Date.*?(\d{2}-\d{2}-\d{4}\s*\d{2}:\d{2}:\d{2})",
        "Bid Opening Date": r"Bid Opening.*?(\d{2}-\d{2}-\d{4}\s*\d{2}:\d{2}:\d{2})",
        "Bid Offer Validity": r"Bid Offer Validity.*?(\d+\s*\(Days\))",
        "Ministry/State Name": r"Ministry/State Name\s*([A-Za-z\s&]+)",
        "Department Name": r"Department Name\s*([A-Za-z/\s&()\.]+)",
        "Organisation Name": r"Organisation Name\s*([A-Za-z/\s&()\.]+)",
        "Office Name": r"Office Name\s*([A-Za-z0-9\s&()\.]+)",
        "Item Category": r"Item Category\s*([A-Za-z0-9\s/\-,&()]+)",
        "Contract Period": r"Contract Period\s*([0-9]+\s*Year\(s\))",
        "Minimum Average Annual Turnover": r"Minimum Average Annual Turnover.*?([\d,.]+\s*\w+\s*\(s\))",
        "Years of Past Experience Required": r"Years of Past Experience Required.*?(\d+\s*Year\s*\(s\))",
        "Auto Extension Days": r"auto[-– ]extended.*?(\d+)",
        "Estimated Bid Value": r"Estimated Bid Value\s*([\d,]+)",
        "Evaluation Method": r"Evaluation Method\s*([A-Za-z\s]+evaluation)",
        "Arbitration Clause": r"Arbitration Clause\s*(Yes|No)",
        "Mediation Clause": r"Mediation Clause\s*(Yes|No)",
        "EMD Amount": r"EMD Amount\s*([\d,]+)",
        "EMD Bank": r"Advisory Bank\s*([A-Za-z ]+)",
        "ePBG Percentage": r"ePBG Percentage.*?([\d.]+)",
        "ePBG Duration": r"Duration of ePBG.*?(\d+)",
        "MII Compliance": r"MII Compliance.*?\b(Yes|No)\b"
    }

    results = {}
    for k, pat in patterns.items():
        m = re.search(pat, norm_text, re.IGNORECASE | re.DOTALL)
        results[k] = m.group(1).strip() if m else "Not found"


    # ----------------------------------------------------
    # NEW FIELDS (1) MSE EXEMPTION (2) STARTUP EXEMPTION
    # ----------------------------------------------------
    # -------------------------
    # NEW: Accurate extraction for MSE & Startup Exemption
    # -------------------------
    # -------------------------
    # Extract MSE & Startup Exemptions via blocks (most reliable)
    # -------------------------
    mse_value = None
    startup_value = None

    doc = fitz.open(pdf_path)

    for page in doc:
        blocks = page.get_text("blocks")

        for i, block in enumerate(blocks):
            text = block[4]

            # ----- MSE EXEMPTION -----
            if "MSE Exemption for Years Of Experience" in text:
                # Next block contains the value
                if i + 1 < len(blocks):
                    mse_value = blocks[i+1][4].strip()

            # ----- STARTUP EXEMPTION -----
            if "Startup Exemption for Years Of" in text:
                if i + 1 < len(blocks):
                    startup_value = blocks[i+1][4].strip()

    doc.close()

    # Save results
    results["MSE Exemption for Years of Experience and Turnover"] = (
        mse_value if mse_value else "Not found"
    )

    results["Startup Exemption for Years of Experience and Turnover"] = (
        startup_value if startup_value else "Not found"
    )

    # -------------------------
    # Past Experience Parsing
    # -------------------------
    m1 = re.search(r"Past Experience of Similar Services[^\n]{0,120}\n\s*[:\-]?\s*(Yes|No)\b",
                   norm_text, re.IGNORECASE)
    if m1:
        results["Past Experience of Similar Services"] = m1.group(1).strip()
    else:
        fallback_yes = re.search(
            r"(Three similar completed services|similar completed services|One similar completed service|Two similar completed services|must have successfully executed)",
            norm_text, re.IGNORECASE)
        if fallback_yes:
            results["Past Experience of Similar Services"] = "Yes"
        else:
            m2 = re.search(r"Past Experience of Similar Services.*?\b(Yes|No)\b", norm_text, re.IGNORECASE)
            results["Past Experience of Similar Services"] = m2.group(1).strip() if m2 else "Not found"


    # -------------------------
    # Scope of Work PDF detection
    # -------------------------
    scope_file_match = re.search(r"Scope of work.*?(\d+\.pdf)", norm_text, re.IGNORECASE)
    scope_file = scope_file_match.group(1) if scope_file_match else None

    scope_url = None
    if scope_file:
        with open(pdf_path, "rb") as fh:
            raw = fh.read().decode("latin1", errors="ignore")
        url_match = re.search(r"https?://\S*" + re.escape(scope_file), raw)
        if url_match:
            scope_url = url_match.group(0)

        results["Scope of Work PDF"] = scope_url if scope_url else scope_file
        results["Criteria File Url"] = "Not Found"

        try:
            response = requests.get(results["Scope of Work PDF"])
            response.raise_for_status()
            pdf_bytes = response.content

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
            doc.close()
            results["Eligibility Criteria Text"] = extract_criteria_using_llm(full_text)

        except:
            results["Eligibility Criteria Text"] = "Not able to download"
    

    elif "Click here to view the file" in limited_text:

        doc = fitz.open(pdf_path)
        found_criteria_url = None

        for page in doc:
            links = page.get_links()
            for ln in links:
                if "uri" in ln:
                    url = ln["uri"]
                    if re.search(r"\.(pdf|docx?|xlsx?|zip)$", url, re.IGNORECASE):
                        found_criteria_url = url
                        break
            if found_criteria_url:
                break

        doc.close()

        results["Criteria File Url"] = found_criteria_url if found_criteria_url else "Not found"
        results["Scope of Work PDF"] = "Not found"

        if found_criteria_url:
            try:
                response = requests.get(found_criteria_url)
                response.raise_for_status()
                pdf_bytes = response.content

                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                full_text = ""
                for page in doc:
                    full_text += page.get_text("text") + "\n"
                doc.close()

                results["Eligibility Criteria Text"] = extract_criteria_using_llm(full_text)

            except:
                results["Eligibility Criteria Text"] = "Not able to download"


    # -------------------------
    # Save Output File
    # -------------------------
    if save_txt:
        out = os.path.join(os.getcwd(), "Extracted_BID_text.txt")
        with open(out, "w", encoding="utf-8") as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"[INFO] Saved text to {out}")


    return results



# RUN FUNCTION
pdf_path = r"C:\Users\Yash Sharma\Downloads\GeM-Bidding-8599859.pdf"
data = extract_text_from_pdf(pdf_path)
print(data)
