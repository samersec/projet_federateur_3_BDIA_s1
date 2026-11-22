import json
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "extraction" / "extracted_pages.jsonl"

MIN_TEXT_CHARS = 50


# =========================
# VERIFICATION
# =========================

def main():

    suspicious_pages = []
    total_pages = 0

    if not INPUT_FILE.exists():
        print("Fichier introuvable :", INPUT_FILE)
        return

    with INPUT_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)
            total_pages += 1

            text = record.get("text", "").strip()

            if len(text) < MIN_TEXT_CHARS:

                suspicious_pages.append({
                    "source": record.get("source"),
                    "page": record.get("page"),
                    "text": text,
                    "length": len(text)
                })

    # =========================
    # AFFICHAGE
    # =========================

    print("=" * 60)
    print("VERIFICATION DES PAGES SUSPECTES")
    print("=" * 60)

    print(f"Pages analysées : {total_pages}")
    print(f"Pages suspectes : {len(suspicious_pages)}")

    if not suspicious_pages:
        print("\nAucune page suspecte détectée.")
        return

    print("\nDETAIL DES PAGES SUSPECTES :")

    for item in suspicious_pages:

        print("\n" + "-" * 60)

        print(f"PDF     : {item['source']}")
        print(f"Page    : {item['page']}")
        print(f"Caractères : {item['length']}")

        print("Texte extrait :")

        if item["text"]:
            print(item["text"])
        else:
            print("[AUCUN TEXTE EXTRAIT]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
