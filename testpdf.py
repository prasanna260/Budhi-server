import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path, password=None):
    """Extract text from a password-protected or normal PDF."""
    doc = fitz.open(pdf_path)

    # If PDF needs password, authenticate
    if doc.needs_pass:
        if not password:
            raise Exception("This PDF requires a password!")
        if not doc.authenticate(password):
            raise Exception("Incorrect PDF password!")

    texts = []
    for page in doc:
        text = page.get_text("text", sort=True)
        texts.append(text)
    doc.close()
    return "\n".join(texts)

def verify_kyc_fields(pdf_text, kyc_info):
    results = {}
    for field, value in kyc_info.items():
        pattern = re.escape(value)
        match = re.search(pattern, pdf_text, re.IGNORECASE)
        results[field] = bool(match)
    return results

if __name__ == "__main__":
    pdf_path = input("Enter path to bank statement PDF: ")
    password = input("Enter PDF password (press Enter if not password-protected): ")

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
    try:
        text = extract_text_from_pdf(pdf_path, password.strip() or None)
    except Exception as e:
        print("Error:", e)
        exit()

    print("[+] Verifying fields...")
    verification = verify_kyc_fields(text, kyc_info)

    print("\n==== Verification Results ====")
    for k, v in verification.items():
        print(f"{k.capitalize()}: {'✔ Found' if v else '✘ Not Found'}")
