
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pdfplumber

# Bornes horizontales des colonnes (page de 612 pt de large).
X_SHIP_TO = 170     # début de la colonne "Ship To"
X_RIGHT = 380       # début de la colonne de droite (Ship Mode, Balance Due, totaux)
X_QUANTITY = 360    # début des colonnes numériques du tableau d'articles

PRODUCT_CODE_RE = re.compile(r"\b[A-Z]{3}-[A-Z]{2}-\d+\b")
MONEY_RE = re.compile(r"-?\$?\(?-?[\d,]+\.\d{2}\)?")


def _money(value: str | None) -> float | None:
    """'$3,960.79' -> 3960.79"""
    if not value:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _rows(words: list[dict], tolerance: float = 3.0) -> list[list[dict]]:
    """Regroupe les mots en lignes (même ordonnée à `tolerance` près)."""
    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows and abs(rows[-1][0]["top"] - word["top"]) <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(row, key=lambda w: w["x0"]) for row in rows]


def _text(words: list[dict]) -> str:
    return " ".join(w["text"] for w in words).strip()


def _find_row(rows: list[list[dict]], label: str) -> int | None:
    for index, row in enumerate(rows):
        if label in _text(row):
            return index
    return None


def _after(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def _parse_ship_to(ship_to: str) -> dict:
    """'98103, Seattle, Washington, United States' -> code postal, ville, état, pays."""
    parts = [p.strip() for p in ship_to.split(",") if p.strip()]
    out: dict = {}
    if parts and re.fullmatch(r"\d{4,6}", parts[0]):
        out["ship_to_postal_code"] = parts.pop(0)
    if parts:
        out["ship_to_country"] = parts[-1]
    if len(parts) >= 2:
        out["ship_to_state"] = parts[-2]
    if len(parts) >= 3:
        out["ship_to_city"] = ", ".join(parts[:-2])
    return out


def extract_invoice(pdf_path: str | Path) -> dict:
    """Retourne {"text": <texte de la facture>, "metadata": {...}}.

    Les champs absents de la facture sont absents du dictionnaire (ou à 0.0 pour
    `discount` / `discount_pct`, afin que les filtres numériques restent utilisables).
    `metadata["is_empty"]` vaut True pour les gabarits vides présents dans le dataset.
    """
    pdf_path = Path(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        words: list[dict] = []
        offset = 0.0
        for page in pdf.pages:
            for word in page.extract_words():
                word = dict(word)
                word["top"] += offset
                words.append(word)
            offset += float(page.height)

    rows = _rows(words)
    full_text = "\n".join(_text(row) for row in rows)
    # Deux fichiers du dataset ont un nom mal encodé : on le rend affichable/stockable.
    source = pdf_path.name.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    meta: dict = {"source": source}

    # --- En-tête -----------------------------------------------------------
    invoice_number = _after(r"#\s*(\d+)", full_text)
    if invoice_number is None:  # repli sur le nom du fichier : invoice_<client>_<numéro>.pdf
        invoice_number = _after(r"_(\d+)\.pdf$", pdf_path.name)
    if invoice_number:
        meta["invoice_number"] = invoice_number

    date_raw = _after(r"Date:\s*([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})", full_text)
    if date_raw:
        date = datetime.strptime(date_raw.replace(",", ""), "%b %d %Y")
        meta["date"] = date.strftime("%Y-%m-%d")
        meta["date_int"] = int(date.strftime("%Y%m%d"))  # pour les filtres d'intervalle
        meta["year"] = date.year
        meta["month"] = date.month

    ship_mode = _after(r"Ship Mode:\s*(.+?)(?:\s{2,}|\n|$)", "\n".join(
        _text([w for w in row if w["x0"] >= X_RIGHT]) for row in rows))
    if ship_mode:
        meta["ship_mode"] = ship_mode

    order_id = _after(r"Order ID\s*:\s*(\S+)", full_text)
    if order_id:
        meta["order_id"] = order_id

    # --- Bill To / Ship To (colonnes séparées par position) ----------------
    bill_row = _find_row(rows, "Bill To:")
    item_header = _find_row(rows, "Quantity")
    if bill_row is not None and item_header is not None:
        band = [w for row in rows[bill_row + 1:item_header] for w in row]
        bill_to = _text([w for w in band if w["x0"] < X_SHIP_TO])
        ship_to = _text([w for w in band if X_SHIP_TO <= w["x0"] < X_RIGHT])
        if bill_to:
            meta["bill_to"] = bill_to
        if ship_to:
            meta["ship_to"] = ship_to
            meta.update(_parse_ship_to(ship_to))

    # --- Ligne d'article ----------------------------------------------------
    subtotal_row = _find_row(rows, "Subtotal:")
    if item_header is not None and subtotal_row is not None:
        item_rows = rows[item_header + 1:subtotal_row]
        description: list[str] = []
        for row in item_rows:
            left = _text([w for w in row if w["x0"] < X_QUANTITY])
            right = [w["text"] for w in row if w["x0"] >= X_QUANTITY]
            if len(right) >= 3 and "quantity" not in meta:
                meta["quantity"] = int(float(right[0].replace(",", "")))
                meta["rate"] = _money(right[1])
                meta["amount"] = _money(right[2])
            code = PRODUCT_CODE_RE.search(left)
            if code:
                meta["product_code"] = code.group(0)
                categories = [c.strip() for c in left[:code.start()].split(",") if c.strip()]
                if len(categories) >= 2:
                    meta["sub_category"], meta["category"] = categories[-2], categories[-1]
            elif left:
                description.append(left)
        if description:
            meta["item"] = " ".join(description)

    # --- Totaux -------------------------------------------------------------
    def labelled_amount(label: str) -> float | None:
        index = _find_row(rows, label)
        if index is None:
            return None
        amounts = MONEY_RE.findall(_text(rows[index]).split(label, 1)[1])
        return _money(amounts[-1]) if amounts else None

    for key, label in (("subtotal", "Subtotal:"), ("shipping", "Shipping:"),
                       ("total", "Total:"), ("balance_due", "Balance Due:")):
        value = labelled_amount(label)
        if value is not None:
            meta[key] = value

    discount = re.search(r"Discount\s*\((\d+(?:\.\d+)?)%\):\s*(\S+)", full_text)
    meta["discount"] = _money(discount.group(2)) if discount else 0.0
    meta["discount_pct"] = float(discount.group(1)) if discount else 0.0

    meta["is_empty"] = "item" not in meta and not meta.get("total")
    meta = {k: v for k, v in meta.items() if v is not None}
    return {"text": full_text, "metadata": meta}


def build_document(extracted: dict) -> str:
    """Texte indexé dans la collection : un résumé lisible, suivi du texte brut."""
    m = extracted["metadata"]
    lines = [
        f"Invoice #{m.get('invoice_number', '?')} dated {m.get('date', '?')}",
        f"Bill to: {m.get('bill_to', '?')}",
        f"Ship to: {m.get('ship_to', '?')} (ship mode: {m.get('ship_mode', '?')})",
        f"Item: {m.get('item', '?')} | {m.get('sub_category', '?')}, {m.get('category', '?')}"
        f" | product code {m.get('product_code', '?')}",
        f"Quantity {m.get('quantity', '?')} x rate {m.get('rate', '?')} = amount {m.get('amount', '?')}",
        f"Subtotal {m.get('subtotal', '?')}, discount {m.get('discount', 0)} ({m.get('discount_pct', 0)}%),"
        f" shipping {m.get('shipping', '?')}, total {m.get('total', '?')}",
        f"Order ID: {m.get('order_id', '?')}",
    ]
    return "\n".join(lines) + "\n\n--- Raw text ---\n" + extracted["text"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python metadata_extractor.py <facture.pdf>")
    print(json.dumps(extract_invoice(sys.argv[1])["metadata"], indent=2, ensure_ascii=False))