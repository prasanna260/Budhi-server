import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    """Extract full text from all pages of the PDF."""
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text("text", sort=True)
        texts.append(text)
    doc.close()
    return "\n".join(texts)

def verify_kyc_fields(pdf_text, kyc_info):
    """
    Given extracted text and a dict of KYC info, verify presence.
    kyc_info = {
      "name": "...",
      "account": "...",
      "ifsc": "..."
    }
    """
    results = {}
    for field, value in kyc_info.items():
        pattern = re.escape(value)
        match = re.search(pattern, pdf_text, re.IGNORECASE)
        results[field] = bool(match)
    return results

if __name__ == "__main__":
    pdf_path = input("Enter path to bank statement PDF: ")

    print("\nEnter the KYC details to verify:")
    name = input("Name: ")
    account = input("Account Number: ")
    ifsc = input("IFSC: ")

    kyc_info = {
        "name": name,
        "account": account,
        "ifsc": ifsc
    }

    print("\n[+] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("[+] Verifying fields...")
    verification = verify_kyc_fields(text, kyc_info)

    print("\n==== Verification Results ====")
    for k, v in verification.items():
        print(f"{k.capitalize()}: {'✔ Found' if v else '✘ Not Found'}")
