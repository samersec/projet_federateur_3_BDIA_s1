
import fitz  # PyMuPDF
import json
import re
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = BASE_DIR / "extraction" / "extracted_pages.jsonl"

# Seuil minimum de caractères pour considérer
# une page comme ayant du texte exploitable
MIN_TEXT_CHARS = 50

# Pour signaler un PDF potentiellement scanné :
# si au moins 50% de ses pages sont suspectes
SUSPICIOUS_PAGE_RATIO = 0.50


# =========================
# NETTOYAGE DU TEXTE
# =========================

def clean_text(text):
    # Supprimer certains caractères invisibles
    text = text.replace("\u200f", "")
    text = text.replace("\u200e", "")
    text = text.replace("\ufeff", "")

    # Réduire les espaces horizontaux excessifs
    text = re.sub(r"[ \t]+", " ", text)

    # Réduire les lignes vides multiples
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================
# EXTRACTION D'UN PDF
# =========================

def extract_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    total_pages = len(doc)
    suspicious_pages = []
    extracted_records = []

    for page_num, page in enumerate(doc, start=1):

        # Extraction directe avec PyMuPDF
        raw_text = page.get_text("text")

        # Nettoyage léger
        text = clean_text(raw_text)

        # Détection heuristique des pages suspectes
        if len(text) < MIN_TEXT_CHARS:
            suspicious_pages.append(page_num)

        # Préparation de la donnée
        record = {
            "source": pdf_path.name,
            "page": page_num,
            "text": text
        }

        extracted_records.append(record)

    doc.close()

    # Calcul de la proportion de pages suspectes
    suspicious_ratio = (
        len(suspicious_pages) / total_pages
        if total_pages > 0
        else 0
    )

    # PDF potentiellement scanné
    is_suspicious = (
        total_pages > 0
        and suspicious_ratio >= SUSPICIOUS_PAGE_RATIO
    )

    return {
        "records": extracted_records,
        "total_pages": total_pages,
        "suspicious_pages": suspicious_pages,
        "suspicious_ratio": suspicious_ratio,
        "is_suspicious": is_suspicious
    }


# =========================
# TRAITEMENT DU CORPUS
# =========================

def main():

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Aucun PDF trouvé dans :", PDF_DIR)
        return

    total_pdfs = len(pdf_files)
    total_pages = 0
    total_suspicious_pages = 0

    suspicious_pdfs = []

    print("=" * 60)
    print("EXTRACTION DU CORPUS PDF")
    print("=" * 60)

    # Ouvrir le fichier de sortie
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:

        for index, pdf_path in enumerate(pdf_files, start=1):

            print(
                f"[{index}/{total_pdfs}] "
                f"Extraction : {pdf_path.name}"
            )

            try:
                result = extract_pdf(pdf_path)

                # Enregistrer chaque page en JSONL
                for record in result["records"]:

                    out.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                total_pages += result["total_pages"]

                total_suspicious_pages += len(
                    result["suspicious_pages"]
                )

                # Si le PDF est potentiellement scanné
                if result["is_suspicious"]:

                    suspicious_pdfs.append({
                        "name": pdf_path.name,
                        "pages": result["total_pages"],
                        "suspicious_pages": (
                            result["suspicious_pages"]
                        ),
                        "ratio": result["suspicious_ratio"]
                    })

                print(
                    f"    Pages : {result['total_pages']} | "
                    f"Pages suspectes : "
                    f"{len(result['suspicious_pages'])}"
                )

            except Exception as e:

                print(f"    ERREUR : {e}")

    # =========================
    # RAPPORT FINAL
    # =========================

    print("\n" + "=" * 60)
    print("RAPPORT FINAL")
    print("=" * 60)

    print(f"PDF trouvés : {total_pdfs}")
    print(f"Pages traitées : {total_pages}")
    print(f"Pages suspectes : {total_suspicious_pages}")

    print(
        f"Fichier extrait : {OUTPUT_FILE}"
    )

    print("\nPDF POTENTIELLEMENT SCANNÉS :")

    if suspicious_pdfs:

        for item in suspicious_pdfs:

            percentage = item["ratio"] * 100

            print(
                f"- {item['name']} "
                f"({percentage:.1f}% pages suspectes)"
            )

            print(
                f"  Pages suspectes : "
                f"{item['suspicious_pages']}"
            )

    else:

        print("Aucun PDF suspect détecté.")

    print("\nExtraction terminée.")


if __name__ == "__main__":
    main()
