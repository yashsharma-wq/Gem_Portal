from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from supabase import create_client,Client
import numpy as np
from openai import OpenAI
import tiktoken
import re,requests
import fitz
from pdf2image import convert_from_path
import glob
import time,os
import traceback
from pytesseract import pytesseract
from PIL import Image
import base64
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
# --- CONFIGURATION ---
path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # path for .exe file
url=os.getenv("SUPABASE_URL")
key=os.getenv("SUPABASE_KEY")
supabase:Client=create_client(url,key)

OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
TOKEN_LIMIT = 8000
client = OpenAI(api_key=OPENAI_API_KEY)

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
        # "Bid Opening Date": r"Bid Opening.*?(\d{2}-\d{2}-\d{4}\s*\d{2}:\d{2}:\d{2})",
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
    # ADD DEFAULT VALUES TO AVOID KEYERROR
    # -------------------------
    results["Scope of Work PDF"] = "Not found"
    results["Criteria File Url"] = "Not found"

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
        with open(out, "a", encoding="utf-8") as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"[INFO] Saved text to {out}")


    return results




# --- Helper Functions ---
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    enc = get_tokenizer()
    return len(enc.encode(text))

def chunk_text(text: str, max_tokens: int = TOKEN_LIMIT):
    enc = get_tokenizer()
    tokens = enc.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield enc.decode(tokens[i:i + max_tokens])

def create_embedding(text: str):
    """Generate embedding vector from text using OpenAI."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding



# --- Main Process ---
def process_pdf_text(text):
    total_tokens = count_tokens(text)
    print(f"🧾 Characters: {len(text)} | 🔢 Tokens: {total_tokens}")

    if total_tokens <= TOKEN_LIMIT:
        print("✅ Small file — creating single embedding...")
        embedding = create_embedding(text)
    else:
        print("⚙️ Large file — splitting and averaging embeddings...")
        chunks = list(chunk_text(text))
        embeddings = [create_embedding(chunk) for chunk in chunks]
        embedding = np.mean(embeddings, axis=0).tolist()
        print(f"✅ Averaged {len(chunks)} embeddings into one vector.")

    # insert_into_supabase(os.path.basename(file_path), text, embedding)
    print(f"📦 Final embedding length: {len(embedding)}\n")
    return embedding



def launch(driver,download_path, timeout=30):
    
    try:
        wait = WebDriverWait(driver, timeout)
        # FILTER BUTTON : LATEST ENTRIES
        filter_btn = wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div[5]/div[2]/div[1]/div[2]/div/button')))
        filter_btn.click()
        
        
        # SELECTION FOR LATEST FIRST                                   
        opt_latest_first=wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[2]/div[5]/div[2]/div[1]/div[2]/div/ul/li[1]/a')))
        opt_latest_first.click()
        
        #SERVICE BID DATA:
        label_element = wait.until(EC.presence_of_element_located(
                        (By.XPATH, '/html/body/div[2]/div[5]/div[1]/div[7]/label')
                        ))
        if 'Service Bid/RAs' not in label_element.text:
            print("NO SERVICE BID DATA AVAILABLE RIGHT NOW!!", label_element.text)
        else:
            # Click the checkbox
            service_bid_element = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '/html/body/div[2]/div[5]/div[1]/div[7]/label/input')
            ))
            service_bid_element.click()
            time.sleep(5)

        # HIGH VALUE BID
        hgh_label_element=wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[2]/div[5]/div[1]/div[13]/label')))

        if 'High Value Bids' not in hgh_label_element.text:
            print("NO HIGH VALUE BID AVAILABLE!!!",hgh_label_element.text)
        else:
            high_value_element=wait.until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[2]/div[5]/div[1]/div[13]/label/input')))
            high_value_element.click()
            time.sleep(5)    
        #----------------
        # END DATE FROM
        # today = datetime.today().strftime("%Y-%m-%d")
        date_input = wait.until(EC.element_to_be_clickable((By.ID, "fromEndDate")))
        date_input.click()

        today_element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".ui-datepicker-today a")))
        today_element.click()
        #----------------

        # END DATE TO
            # open the datepicker
        end_input = wait.until(EC.element_to_be_clickable((By.ID, "toEndDate")))
        end_input.click()

        # Keep clicking "Next" until the next button becomes disabled (meaning last month)
        while True:
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, ".ui-datepicker-next:not(.ui-state-disabled)")
                next_btn.click()
            except:
                break  # next button is disabled → you're at the last available month

        # Now click the last enabled date in the calendar
        last_day = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "td:not(.ui-datepicker-other-month):not(.ui-state-disabled) a:last-child")
        ))
        last_day.click()     

        #----------------
        
        total_entr_element=wait.until(EC.presence_of_element_located((By.XPATH,'/html/body/div[2]/div[5]/div[2]/div[13]/a[6]')))
                                                                            
        total_entries=total_entr_element.text
        print("total_entries:",total_entries)


        # DOWNLOAD CONTENT ON PAGE 1:
        for j in range(1,3):
            for i in range(1,11):
            # FOR CARD -1:
                ten_card_btn=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]')))
                ten_card_btn.click()

                bid_no_element=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[1]/p[1]/a')))
                Bid_No=bid_no_element.text
                Bid_File_Url=bid_no_element.get_attribute("href")
                print(Bid_No)
                #-----
                # ra_no_element=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[1]/p[3]/a')))
                # RA_No=ra_no_element.text
                # RA_File_Url=ra_no_element.get_attribute("href")
                # print(RA_No)
                #-----
                try:
                    # Try the <a> tag first (most common)
                    items_element = driver.find_element(By.XPATH, f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[3]/div/div[1]/div[1]/a')
                    Items_Data = items_element.get_attribute("data-content") or items_element.text.strip()
                except Exception:
                    # If <a> tag not found, try plain text node
                    try:
                        items_element = driver.find_element(By.XPATH, f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[3]/div/div[1]/div[1]')
                        Items_Data = items_element.text.strip()
                    except Exception:
                        Items_Data = "N/A"
                print(Items_Data)
                #-----
                quantity_element=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[3]/div/div[1]/div[2]')))
                Quantity_Data=quantity_element.text
                print(Quantity_Data)
                #-----
                department_element=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[3]/div/div[2]/div[2]')))
                Department_Data=department_element.text
                print(Department_Data)
                #-----
                startdate_element=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[3]/div/div[3]/div[1]/span')))
                StartDate_Data=startdate_element.text
                print(StartDate_Data) 
                #-----
                enddate_element=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[3]/div/div[3]/div[2]/span')))
                ENDDate_Data=enddate_element.text
                print(ENDDate_Data) 
                #-----
                bid_no_down=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[{i+1}]/div[1]/p[1]/a')))
                bid_no_down.click()
                time.sleep(5)

                #-----Extract Text
                # Check if any new file appeared
                downloaded_files = glob.glob(os.path.join(download_path, "*"))
                if not downloaded_files:
                    driver.quit()
                    raise Exception("No file was downloaded. The site may require session/captcha access.")

                # Get the most recently downloaded file
                BID_latest_file = max(downloaded_files, key=os.path.getctime)
                BID_file_path=rf"{BID_latest_file}"
                
                #------------ADD BID-FILE-URL && START DATE && Bid_No
                file_handler=open('Extracted_BID_text.txt','w')
                file_handler.write(f"Bid No:{Bid_No}\n")
                file_handler.write(f"Bid Start Date:{StartDate_Data}\n")
                file_handler.write(f"Bid File Url:{Bid_File_Url}\n")
                file_handler.close()
                #------------------------------------------
                BID_extracted_dic = extract_text_from_pdf(BID_file_path, use_ocr=True)

                # Combine manual metadata + extracted PDF data before embedding
                manual_metadata = (
                    f"Bid No: {Bid_No}\n"
                    f"Bid Start Date: {StartDate_Data}\n"
                    f"Bid File Url: {Bid_File_Url}\n\n"
                )

                results_text = "\n".join([f"{k}: {v}" for k, v in BID_extracted_dic.items()])

                BID_extracted_text = manual_metadata + results_text

                #----Generate Embedding
                BID_Embedding=process_pdf_text(BID_extracted_text)
    
        # -------------------------------
                
                #-----------Next Page
                # next_pg_btn=wait.until(EC.presence_of_element_located((By.XPATH,f'/html/body/div[2]/div[5]/div[2]/div[13]/a[{6+j}]')))
                

                bid_data={
                    "Bid_No":Bid_No,
                    "Items":Items_Data,
                    "Quantity":Quantity_Data,
                    "Department":Department_Data,
                    "Start_Date":StartDate_Data,
                    "End_Date":ENDDate_Data,
                    "Bid End Date":BID_extracted_dic['Bid End Date'],
                    "Bid Opening Date":BID_extracted_dic['Bid Opening Date'],
                    "Bid Offer Validity":BID_extracted_dic['Bid Offer Validity'],
                    "Ministry Name":BID_extracted_dic['Ministry/State Name'],
                    "Department Name":BID_extracted_dic['Department Name'],
                    "Organisation Name":BID_extracted_dic['Organisation Name'],
                    "Office Name":BID_extracted_dic['Office Name'],
                    "Item Category":BID_extracted_dic['Item Category'],
                    "Contract Period":BID_extracted_dic['Contract Period'],
                    "Minimum Average Annual Turnover":BID_extracted_dic['Minimum Average Annual Turnover'],
                    "Years of Past Experience Required":BID_extracted_dic['Years of Past Experience Required'],
                    "Past Experience of Similar Services":BID_extracted_dic['Past Experience of Similar Services'],
                    "Auto Extension Days":BID_extracted_dic['Auto Extension Days'],
                    "Estimated Bid Value":BID_extracted_dic['Estimated Bid Value'],
                    "Evaluation Method":BID_extracted_dic['Evaluation Method'],
                    "Arbitration Clause":BID_extracted_dic['Arbitration Clause'],
                    "Mediation Clause":BID_extracted_dic['Mediation Clause'],
                    "EMD Amount": BID_extracted_dic["EMD Amount"],
                    "EMD Bank": BID_extracted_dic["EMD Bank"],
                    "ePBG Percentage": BID_extracted_dic["ePBG Percentage"],
                    "ePBG Duration": BID_extracted_dic["ePBG Duration"],
                    "MII Compliance": BID_extracted_dic["MII Compliance"],
                    "MSE Exemption for Years of Experience and Turnover": BID_extracted_dic["MSE Exemption for Years of Experience and Turnover"],
                    "Startup Exemption for Years of Experience and Turnover":BID_extracted_dic["Startup Exemption for Years of Experience and Turnover"],
                    "Scope of Work PDF": BID_extracted_dic["Scope of Work PDF"],
                    "Bid_File":Bid_File_Url,
                    "Bid_File_Text":BID_extracted_text,
                    "Bid_Embedding":BID_Embedding
                    
                }
                # response=supabase.table("GEM_SERVICE_TENDER").insert(bid_data).execute()
                # print(response)
                                
                # 🟢 UPSERT instead of insert
                response = supabase.table("GEM_TABLE_UPDATED").upsert(bid_data, on_conflict=["Bid_No"]).execute()
                if response.data:
                    print(f"✅ Upsert successful for Bid_No: {Bid_No}")
                else:
                    print(f"⚠️ No data returned for Bid_No: {Bid_No}, check response:", response)

            try:
                # wait until the "Next" button appears and is clickable
                next_btn = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.page-link.next"))
                )
                
                # scroll into view if necessary
                driver.execute_script("arguments[0].scrollIntoView(true);", next_btn)
                
                # click it
                next_btn.click()
                print(f"✅ Clicked Next ({j+1}/30)")

                # small delay to allow new content to load
                time.sleep(2)

            except Exception as e:
                print(f"⚠️ Stopped at page {j+1} — {e}")
                break
        return True

    except Exception as e:
        print("Exception in launch():", e)
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Accessing driver")

        
    # Set your custom download path
    download_path = r"D:\YASH STUFF\Awign\GeM_Tenders"

    # Make sure the folder exists
    os.makedirs(download_path, exist_ok=True)

    # Configure Chrome options
    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": download_path,  # <-- Custom folder path
        "download.prompt_for_download": False,        # Auto download, no prompt
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True                  # Avoid "dangerous file" blocking
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=chrome_options)
    print("trying to open")
    driver.get("https://bidplus.gem.gov.in/all-bids")
    print("opened")
    # recommended to use waits inside launch, so small sleep here only for page-level
    time.sleep(2)

    success = launch(driver,download_path)
    if not success:
        print("Launch failed — check stack trace above")
    else:
        print("Launch finished (files saved if found).")


    # Keep the browser open for inspection (press Enter to exit)
    driver.quit()
