
from __future__ import annotations

import os
import re
import sys

from qdrant_client import models

from ingestion import COLLECTION_NAME, DEFAULT_DB_PATH, embed, get_client

TOP_K = 5

NUMERIC_FIELDS = {"total", "shipping", "discount", "discount_pct", "amount", "subtotal",
                  "balance_due", "quantity", "rate", "year", "month", "date_int"}
STRING_FIELDS = {"bill_to", "ship_to_country", "ship_to_state", "ship_to_city",
                 "ship_to_postal_code", "ship_mode", "category", "sub_category",
                 "product_code", "invoice_number", "order_id", "date", "item"}
# Champs dont on charge les valeurs pour les reconnaître dans la question.
VOCAB_FIELDS = ("bill_to", "ship_to_country", "ship_to_state", "ship_to_city",
                "ship_mode", "category", "sub_category")

ALIASES = {
    # numériques
    "total": "total", "shipping": "shipping", "livraison": "shipping", "port": "shipping",
    "discount": "discount", "remise": "discount", "réduction": "discount", "reduction": "discount",
    "amount": "amount", "montant": "amount", "subtotal": "subtotal", "sous-total": "subtotal",
    "quantity": "quantity", "quantité": "quantity", "quantite": "quantity", "qty": "quantity",
    "rate": "rate", "prix": "rate", "year": "year", "année": "year", "annee": "year",
    "month": "month", "mois": "month", "discount_pct": "discount_pct", "balance_due": "balance_due",
    # texte
    "bill_to": "bill_to", "client": "bill_to", "customer": "bill_to",
    "country": "ship_to_country", "pays": "ship_to_country",
    "city": "ship_to_city", "ville": "ship_to_city",
    "state": "ship_to_state", "état": "ship_to_state", "etat": "ship_to_state",
    "postal_code": "ship_to_postal_code", "ship_mode": "ship_mode", "mode": "ship_mode",
    "category": "category", "catégorie": "category", "categorie": "category",
    "sub_category": "sub_category", "code": "product_code", "product_code": "product_code",
    "invoice": "invoice_number", "facture": "invoice_number", "invoice_number": "invoice_number",
    "order_id": "order_id", "date": "date",
}
for _field in NUMERIC_FIELDS | STRING_FIELDS:
    ALIASES.setdefault(_field, _field)

FRENCH_COUNTRIES = {
    "chine": "China", "états-unis": "United States", "etats-unis": "United States", "usa": "United States",
    "allemagne": "Germany", "espagne": "Spain", "italie": "Italy", "royaume-uni": "United Kingdom",
    "inde": "India", "mexique": "Mexico", "australie": "Australia", "brésil": "Brazil", "bresil": "Brazil",
    "japon": "Japan", "indonésie": "Indonesia", "indonesie": "Indonesia", "turquie": "Turkey",
    "pays-bas": "Netherlands", "russie": "Russia", "égypte": "Egypt", "egypte": "Egypt", "maroc": "Morocco",
}

OPERATORS = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "==": "$eq", "!=": "$ne"}
COMPARATIVES = [  # (tournure, opérateur) — les plus longues d'abord
    (r"supérieure?s? ou égale?s? à|superieure?s? ou egale?s? a|au moins|at least|minimum", "$gte"),
    (r"inférieure?s? ou égale?s? à|inferieure?s? ou egale?s? a|au plus|at most|maximum", "$lte"),
    (r"supérieure?s? à|superieure?s? a|plus de|plus grande?s? que|greater than|more than|over|above|dépassant|depassant", "$gt"),
    (r"inférieure?s? à|inferieure?s? a|moins de|plus petite?s? que|less than|under|below", "$lt"),
    (r"égale?s? à|egale?s? a|equal to|de exactement|exactly", "$eq"),
]
NUMBER = r"\$?\s*(\d[\d\s,]*(?:\.\d+)?)\s*(%)?"
FIELD_WORDS = "|".join(sorted((re.escape(a) for a, f in ALIASES.items() if f in NUMERIC_FIELDS),
                              key=len, reverse=True))


def _number(raw: str) -> float:
    return float(re.sub(r"[\s,$]", "", raw))


def _numeric_field(alias: str, percent: bool) -> str:
    field = ALIASES[alias.lower()]
    return "discount_pct" if field == "discount" and percent else field


def fetch_payloads(client, query_filter=None) -> list[dict]:
    """Toutes les métadonnées (payloads) des factures qui passent le filtre."""
    payloads, offset = [], None
    while True:
        points, offset = client.scroll(COLLECTION_NAME, scroll_filter=query_filter, limit=500,
                                       offset=offset, with_payload=True, with_vectors=False)
        payloads += [point.payload for point in points]
        if offset is None:
            return payloads


def load_vocabulary(client) -> dict[str, list[tuple[str, str]]]:
    """{valeur en minuscules: [(champ, valeur exacte), ...]} pour les champs texte."""
    vocab: dict[str, list[tuple[str, str]]] = {}
    for meta in fetch_payloads(client):
        for field in VOCAB_FIELDS:
            value = meta.get(field)
            if isinstance(value, str) and len(value) >= 3:
                entries = vocab.setdefault(value.lower(), [])
                if (field, value) not in entries:
                    entries.append((field, value))
    return vocab


def to_qdrant_filter(where: dict | None) -> models.Filter | None:
    """Convertit le filtre neutre ({'$and': [...]}, {'total': {'$gt': 100}}...) en filtre Qdrant."""
    if where is None:
        return None
    must, must_not = [], []
    for key, value in where.items():
        if key == "$and":
            must += [to_qdrant_filter(condition) for condition in value]
        elif key == "$or":
            must.append(models.Filter(should=[to_qdrant_filter(condition) for condition in value]))
        else:
            (operator, operand), = value.items()
            if operator == "$in":
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(operand))))
            elif operator in ("$eq", "$ne"):
                if isinstance(operand, float):  # égalité sur un nombre décimal -> intervalle [v, v]
                    condition = models.FieldCondition(key=key, range=models.Range(gte=operand, lte=operand))
                else:
                    condition = models.FieldCondition(key=key, match=models.MatchValue(value=operand))
                (must if operator == "$eq" else must_not).append(condition)
            else:  # $gt, $gte, $lt, $lte
                must.append(models.FieldCondition(key=key, range=models.Range(**{operator[1:]: operand})))
    return models.Filter(must=must or None, must_not=must_not or None)


def parse_question(question: str, vocab: dict) -> tuple[dict | None, str]:
    """Retourne (filtre neutre ou None, texte restant pour la recherche sémantique)."""
    conditions: list[dict] = []
    text = question
    for french, english in FRENCH_COUNTRIES.items():  # les pays sont en anglais dans les factures
        text = re.sub(rf"(?<!\w){french}(?!\w)", english, text, flags=re.IGNORECASE)

    def consume(match: re.Match) -> None:
        nonlocal text
        text = text[:match.start()] + " " * (match.end() - match.start()) + text[match.end():]

    # 1. Expressions explicites : champ op valeur
    explicit = re.compile(r"([\w\-éèà]+)\s*(>=|<=|!=|==|=|>|<)\s*(\"[^\"]+\"|'[^']+'|[^\s,;]+)")
    for match in list(explicit.finditer(text)):
        alias, operator, raw = match.group(1).lower(), match.group(2), match.group(3).strip("\"'")
        if alias not in ALIASES:
            continue
        field = ALIASES[alias]
        if field in NUMERIC_FIELDS:
            try:
                value: float | str = _number(raw.rstrip("%"))
            except ValueError:
                continue
            field = _numeric_field(alias, raw.endswith("%"))
            if field in {"year", "month", "quantity", "date_int"}:
                value = int(value)
        else:
            value = raw
            if operator in ("=", "==") and raw.lower() in vocab:  # insensible à la casse
                value = next((v for f, v in vocab[raw.lower()] if f == field), raw)
        conditions.append({field: {OPERATORS[operator]: value}})
        consume(match)

    # 2. "<champ> entre X et Y" / "<champ> between X and Y"
    between = re.compile(rf"\b({FIELD_WORDS})\b[^\d\n]{{0,25}}?\b(?:entre|between)\s*{NUMBER}\s*(?:et|and)\s*{NUMBER}",
                         re.IGNORECASE)
    for match in list(between.finditer(text)):
        field = _numeric_field(match.group(1), bool(match.group(3) or match.group(5)))
        low, high = sorted((_number(match.group(2)), _number(match.group(4))))
        conditions += [{field: {"$gte": low}}, {field: {"$lte": high}}]
        consume(match)

    # 3. "<champ> supérieur à X", "<champ> over X"...
    for pattern, operator in COMPARATIVES:
        comparative = re.compile(rf"\b({FIELD_WORDS})\b[^\d\n]{{0,25}}?(?:{pattern})\s*{NUMBER}", re.IGNORECASE)
        for match in list(comparative.finditer(text)):
            field = _numeric_field(match.group(1), bool(match.group(3)))
            conditions.append({field: {operator: _number(match.group(2))}})
            consume(match)

    # 4. Code produit, numéro de facture, année
    for match in list(re.finditer(r"\b[A-Z]{3}-[A-Z]{2}-\d+\b", text)):
        conditions.append({"product_code": {"$eq": match.group(0)}})
        consume(match)
    for match in list(re.finditer(r"(?:#|\b(?:facture|invoice)\s+(?:n°|no\.?|numéro|number)?\s*#?)\s*(\d{4,6})\b",
                                  text, re.IGNORECASE)):
        conditions.append({"invoice_number": {"$eq": match.group(1)}})
        consume(match)
    years = [int(m.group(0)) for m in re.finditer(r"\b(?:19|20)\d{2}\b", text)]
    if years:
        conditions.append({"year": {"$eq": years[0]} if len(years) == 1 else {"$in": sorted(set(years))}})
        text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)

    # 5. Valeurs connues de la collection (les plus longues d'abord)
    by_field: dict[tuple, list] = {}
    for value_lower in sorted(vocab, key=len, reverse=True):
        match = re.search(rf"(?<!\w){re.escape(value_lower)}(?!\w)", text, re.IGNORECASE)
        if not match:
            continue
        fields = tuple(sorted(f for f, _ in vocab[value_lower]))
        by_field.setdefault(fields, []).append(dict(vocab[value_lower]))
        consume(match)
    for fields, matches in by_field.items():
        # Une valeur présente dans plusieurs champs (ex. ville et état) -> $or entre ces champs.
        alternatives = []
        for field in fields:
            values = sorted({m[field] for m in matches})
            alternatives.append({field: {"$eq": values[0]} if len(values) == 1 else {"$in": values}})
        conditions.append(alternatives[0] if len(alternatives) == 1 else {"$or": alternatives})

    where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}
    return where, re.sub(r"\s+", " ", text).strip()


# --- Réponse ciblée : "quelle est la quantité de la facture 36258 ?" -> seulement la quantité -----------
# (motif, champ, libellé) — les motifs les plus précis d'abord.
REQUEST_FIELDS = [
    (r"mode de livraison|mode d'expédition|ship mode|ship_mode", "ship_mode", "Mode de livraison"),
    (r"code produit|product code|product_code|code", "product_code", "Code produit"),
    (r"sous-catégories?|sous-categories?|sub-category|subcategory|sub_category", "sub_category", "Sous-catégorie"),
    (r"catégories?|categories?|category", "category", "Catégorie"),
    (r"quantités?|quantites?|quantity|quantities|qty", "quantity", "Quantité"),
    (r"sous-total|subtotal", "subtotal", "Sous-total"),
    (r"montant total|total amount", "total", "Total"),
    (r"frais de port|frais de livraison|shipping|livraison|port", "shipping", "Livraison"),
    (r"remises?|discounts?|réductions?|reductions?", "discount", "Remise"),
    (r"prix unitaire|unit price|prix|rate", "rate", "Prix unitaire"),
    (r"montants?|amounts?", "amount", "Montant"),
    (r"total|totaux", "total", "Total"),
    (r"dates?|quand|when", "date", "Date"),
    (r"clients?|customers?|bill to|bill_to|facturée? à", "bill_to", "Client"),
    (r"ship to|ship_to|adresses?|address|destinations?", "ship_to", "Adresse de livraison"),
    (r"pays|country|countries", "ship_to_country", "Pays"),
    (r"villes?|city|cities", "ship_to_city", "Ville"),
    (r"articles?|produits?|items?|products?", "item", "Article"),
    (r"order id|order_id|n° de commande|numéro de commande|commande", "order_id", "Order ID"),
]
FIELD_LABELS = {field: label for _, field, label in REQUEST_FIELDS}
MONEY_FIELDS = {"total", "amount", "subtotal", "shipping", "discount", "rate"}
SUMMABLE_FIELDS = MONEY_FIELDS | {"quantity"}
AGGREGATES = [  # (motif, opération)
    (r"moyennes?|moyens?|average|avg|mean", "avg"),
    (r"sommes?|sum|totales?|totals|cumulée?s?|cumul|au total|en tout|overall", "sum"),
    (r"maximum|maximale?s?|max|plus grande?s?|plus élevée?s?|highest|largest|biggest", "max"),
    (r"minimum|minimale?s?|min|plus petite?s?|plus basse?s?|plus bas|lowest|smallest", "min"),
]
COUNT_RE = r"combien de factures|combien y a|nombre de factures|how many invoices|number of invoices|how many"
AGG_LABELS = {"sum": "Somme", "avg": "Moyenne", "max": "Maximum", "min": "Minimum"}
MAX_LINES = 30


def _word(pattern: str) -> str:
    return rf"(?<!\w)(?:{pattern})(?!\w)"


def detect_request(remaining_text: str) -> tuple[list[str], str | None]:
    """Champs demandés et opération (sum/avg/max/min/count) dans le texte restant après les filtres.

    "quantité de la facture 36258" -> (["quantity"], None)
    "total des remises en France"  -> (["discount"], "sum")
    "combien de factures en 2013"  -> ([], "count")
    """
    text = remaining_text.lower()
    aggregate = None
    if re.search(_word(COUNT_RE), text):
        return [], "count"

    # "total des remises", "somme des quantités" : ici "total" est une opération, pas un champ
    others = "|".join(pattern for pattern, field, _ in REQUEST_FIELDS if field != "total")
    total_of = re.search(rf"(?<!\w)(?:total|totaux)\s+(?:des|du|de la|de l'|de|d'|of the|of)\s*(?={others})", text)
    if total_of:
        aggregate = "sum"
        text = text[:total_of.start()] + " " + text[total_of.end():]

    fields: list[str] = []
    for pattern, field, _ in REQUEST_FIELDS:
        match = re.search(_word(pattern), text)
        if match:
            if field not in fields:
                fields.append(field)
            text = text[:match.start()] + " " + text[match.end():]
    for pattern, operation in AGGREGATES:
        if aggregate is None and re.search(_word(pattern), text):
            aggregate = operation
    if aggregate and not fields:  # "la plus grande facture de 2013" -> on compare les totaux
        fields = ["total"]
    return fields, aggregate


def format_value(field: str, meta: dict) -> str:
    value = meta.get(field)
    if value is None:
        return "?"
    if field == "discount":
        return f"${value:,.2f} ({meta.get('discount_pct', 0):g}%)"
    if field in MONEY_FIELDS:
        return f"${value:,.2f}"
    return str(value)


def _format_number(field: str, value: float) -> str:
    return f"${value:,.2f}" if field in MONEY_FIELDS else f"{value:,.2f}".rstrip("0").rstrip(".")


def concise_answer(rows: list[dict], fields: list[str], aggregate: str | None) -> list[str]:
    """Lignes de réponse limitées à ce qui est demandé (pas tout le détail des factures)."""
    if aggregate == "count":
        return [f"{len(rows)} facture(s)"]
    if not rows:
        return ["Aucune facture trouvée."]
    numeric = [f for f in fields if f in SUMMABLE_FIELDS]

    if aggregate and numeric:  # une opération demandée : une ligne par champ
        lines = []
        for field in numeric:
            valued = [(row.get(field, 0), row) for row in rows]
            if aggregate == "sum":
                result, suffix = sum(v for v, _ in valued), ""
            elif aggregate == "avg":
                result, suffix = sum(v for v, _ in valued) / len(valued), ""
            else:
                result, best = (max if aggregate == "max" else min)(valued, key=lambda pair: pair[0])
                suffix = f" (facture #{best.get('invoice_number')})"
            lines.append(f"{AGG_LABELS[aggregate]} — {FIELD_LABELS[field]} : {_format_number(field, result)}{suffix}"
                         f" | sur {len(rows)} facture(s)")
        return lines

    if len(rows) == 1:  # une seule facture : "Quantité : 3"
        return [f"Facture #{rows[0].get('invoice_number')}"] + \
               [f"{FIELD_LABELS[f]} : {format_value(f, rows[0])}" for f in fields]

    if not numeric:  # champs texte sur plusieurs factures : valeurs distinctes
        counts: dict[tuple, int] = {}
        for row in rows:
            key = tuple(format_value(f, row) for f in fields)
            counts[key] = counts.get(key, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        lines = [f"{' | '.join(key)}  ({n} facture{'s' if n > 1 else ''})" for key, n in ordered[:MAX_LINES]]
        if len(ordered) > MAX_LINES:
            lines.append(f"... et {len(ordered) - MAX_LINES} autres valeurs")
        return [f"{' / '.join(FIELD_LABELS[f] for f in fields)} — {len(ordered)} valeur(s) distincte(s) :"] + lines

    lines = [f"#{row.get('invoice_number')} — " + " | ".join(f"{FIELD_LABELS[f]} : {format_value(f, row)}" for f in fields)
             for row in rows[:MAX_LINES]]
    if len(rows) > MAX_LINES:
        lines.append(f"... et {len(rows) - MAX_LINES} autres factures")
    for field in numeric:
        if field != "rate":
            lines.append(f"Somme — {FIELD_LABELS[field]} : {_format_number(field, sum(r.get(field, 0) for r in rows))}"
                         f" | sur {len(rows)} facture(s)")
    return lines


def search(client, question: str, vocab: dict, top_k: int = TOP_K) -> dict:
    where, semantic_text = parse_question(question, vocab)
    query_filter = to_qdrant_filter(where)
    fields, aggregate = detect_request(semantic_text)
    result = {"where": where, "stats": None, "fields": fields, "aggregate": aggregate, "matching": None}

    if where or aggregate:  # statistiques exactes sur TOUTES les factures qui passent le filtre
        matching = fetch_payloads(client, query_filter)
        matching.sort(key=lambda m: int(m.get("invoice_number", 0)))
        result["matching"] = matching
        totals = [m.get("total", 0.0) for m in matching]
        result["stats"] = {"count": len(matching), "sum_total": round(sum(totals), 2),
                           "avg_total": round(sum(totals) / len(totals), 2) if totals else 0.0}
        if not matching:
            result["hits"] = []
            return result

    found = client.query_points(COLLECTION_NAME, query=embed([semantic_text or question])[0],
                                query_filter=query_filter, limit=top_k, with_payload=True).points
    result["hits"] = [{"metadata": point.payload, "score": point.score} for point in found]
    return result


def format_hit(meta: dict) -> str:
    discount = f" | discount ${meta['discount']:,.2f} ({meta['discount_pct']:g}%)" if meta.get("discount") else ""
    return (f"#{meta.get('invoice_number')} | {meta.get('date')} | bill to: {meta.get('bill_to')} | "
            f"ship to: {meta.get('ship_to')} | {meta.get('ship_mode')}\n"
            f"     {meta.get('item')} [{meta.get('product_code')}] x{meta.get('quantity')} | "
            f"amount ${meta.get('amount', 0):,.2f}{discount} | shipping ${meta.get('shipping', 0):,.2f} | "
            f"total ${meta.get('total', 0):,.2f}")


def llm_answer(question: str, result: dict) -> str | None:
    """Réponse rédigée (optionnelle). Retourne None si aucun LLM n'est configuré."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    context = "\n".join(f"- {format_hit(h['metadata'])}" for h in result["hits"])
    stats = f"Statistiques exactes sur toutes les factures filtrées : {result['stats']}\n" if result["stats"] else ""
    message = anthropic.Anthropic().messages.create(
        model=os.environ.get("CHAT_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=600,
        system="Tu réponds à des questions sur des factures, uniquement à partir du contexte fourni. "
               "Si le contexte ne suffit pas, dis-le. Réponds dans la langue de la question.",
        messages=[{"role": "user", "content": f"Filtre appliqué : {result['where']}\n{stats}"
                                              f"Factures les plus pertinentes :\n{context}\n\nQuestion : {question}"}],
    )
    return message.content[0].text


def answer(client, question: str, vocab: dict, top_k: int = TOP_K) -> None:
    result = search(client, question, vocab, top_k)
    print(f"\nFiltre : {result['where'] or 'aucun'}")
    if result["fields"] or result["aggregate"]:  # question ciblée : on ne répond qu'à ce qui est demandé
        rows = result["matching"] if result["matching"] is not None else [h["metadata"] for h in result["hits"]]
        for line in concise_answer(rows, result["fields"], result["aggregate"]):
            print(f" {line}")
        print()
        return
    if result["stats"]:
        stats = result["stats"]
        print(f"{stats['count']} facture(s) correspondent | somme des totaux ${stats['sum_total']:,.2f} | "
              f"total moyen ${stats['avg_total']:,.2f}")
    if not result["hits"]:
        print("Aucune facture trouvée.\n")
        return
    print(f"Top {len(result['hits'])} :")
    for rank, hit in enumerate(result["hits"], 1):
        print(f" {rank}. {format_hit(hit['metadata'])}")
    generated = llm_answer(question, result)
    if generated:
        print(f"\nRéponse : {generated}")
    print()


def main() -> None:
    client = get_client(os.environ.get("QDRANT_DB_PATH", DEFAULT_DB_PATH))
    if not client.collection_exists(COLLECTION_NAME) or client.count(COLLECTION_NAME).count == 0:
        sys.exit("La collection 'invoices' est vide : lancez d'abord `python ingestion.py`.")
    vocab = load_vocabulary(client)

    if len(sys.argv) > 1:
        answer(client, " ".join(sys.argv[1:]), vocab)
        return

    print(f"Collection 'invoices' : {client.count(COLLECTION_NAME).count} factures. "
          "Posez une question (/champs pour la liste des filtres, /quit pour sortir).")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("/quit", "/exit", "quit", "exit"):
            break
        if question == "/champs":
            print("Numériques :", ", ".join(sorted(NUMERIC_FIELDS)))
            print("Texte      :", ", ".join(sorted(STRING_FIELDS)))
            print('Exemples   : total>1000 country=France | bill_to="Joy Smith" | discount over 20%\n')
            continue
        answer(client, question, vocab)


if __name__ == "__main__":
    main()