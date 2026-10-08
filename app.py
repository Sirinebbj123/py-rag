
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from chat import (FIELD_LABELS, concise_answer, detect_request, fetch_payloads, load_vocabulary,
                  parse_question, to_qdrant_filter)
from ingestion import COLLECTION_NAME, DEFAULT_DATA_DIR, DEFAULT_DB_PATH, embed, get_client

# (champ, libellé) des filtres à choix multiples et des curseurs numériques
CHOICE_FILTERS = [
    ("bill_to", "Client (Bill To)"),
    ("ship_to_country", "Pays (Ship To)"),
    ("ship_to_state", "État / région"),
    ("ship_to_city", "Ville"),
    ("category", "Catégorie"),
    ("sub_category", "Sous-catégorie"),
    ("ship_mode", "Mode de livraison"),
]
RANGE_FILTERS = [
    ("total", "Total ($)"),
    ("amount", "Montant / Amount ($)"),
    ("shipping", "Livraison / Shipping ($)"),
    ("discount", "Remise / Discount ($)"),
    ("discount_pct", "Remise (%)"),
    ("quantity", "Quantité"),
]
TABLE_COLUMNS = ["invoice_number", "date", "bill_to", "ship_to", "ship_mode", "item", "product_code",
                 "category", "sub_category", "quantity", "rate", "amount", "discount", "discount_pct",
                 "shipping", "total", "order_id"]


# --------------------------------------------------------------------------- données
@st.cache_resource(show_spinner="Ouverture de la base Qdrant...")
def get_qdrant():
    return get_client(os.environ.get("QDRANT_DB_PATH", DEFAULT_DB_PATH))


@st.cache_data(show_spinner="Chargement des factures...")
def load_all() -> pd.DataFrame:
    """Toutes les métadonnées, pour alimenter les listes et les bornes des filtres."""
    return pd.DataFrame(fetch_payloads(get_qdrant()))


@st.cache_data(show_spinner=False)
def get_vocabulary() -> dict:
    return load_vocabulary(get_qdrant())


# --------------------------------------------------------------------------- filtres
def build_where(choices: dict, ranges: dict, bounds: dict, dates: tuple | None,
                product_code: str, invoice_number: str, text_where: dict | None) -> dict | None:
    """Assemble les filtres de l'interface (+ ceux déduits de la question) en un filtre neutre."""
    conditions: list[dict] = []
    for field, values in choices.items():
        if values:
            conditions.append({field: {"$eq": values[0]} if len(values) == 1 else {"$in": list(values)}})
    for field, (low, high) in ranges.items():
        full_low, full_high = bounds[field]
        if low > full_low:   # un curseur laissé sur toute la plage n'ajoute aucun filtre
            conditions.append({field: {"$gte": low}})
        if high < full_high:
            conditions.append({field: {"$lte": high}})
    if dates and len(dates) == 2:
        conditions.append({"date_int": {"$gte": int(dates[0].strftime("%Y%m%d"))}})
        conditions.append({"date_int": {"$lte": int(dates[1].strftime("%Y%m%d"))}})
    if product_code.strip():
        conditions.append({"product_code": {"$eq": product_code.strip().upper()}})
    if invoice_number.strip():
        conditions.append({"invoice_number": {"$eq": invoice_number.strip().lstrip("#")}})
    if text_where:
        conditions += text_where["$and"] if "$and" in text_where else [text_where]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def run_search(client, where: dict | None, semantic_text: str, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (toutes les factures qui passent le filtre, résultats à afficher).

    Avec un texte de recherche : les `top_k` factures les plus proches, classées par score.
    Sans texte : toutes les factures filtrées, triées par total décroissant.
    """
    query_filter = to_qdrant_filter(where)
    matching = pd.DataFrame(fetch_payloads(client, query_filter))
    if matching.empty:
        return matching, matching
    if not semantic_text:
        return matching, matching.sort_values("total", ascending=False).reset_index(drop=True)
    points = client.query_points(COLLECTION_NAME, query=embed([semantic_text])[0], query_filter=query_filter,
                                 limit=top_k, with_payload=True).points
    shown = pd.DataFrame([{**point.payload, "score": round(point.score, 3)} for point in points])
    return matching, shown


# --------------------------------------------------------------------------- interface
def sidebar(data: pd.DataFrame) -> tuple[dict, dict, dict, tuple | None, str, str, int]:
    st.sidebar.header("Filtres")
    if st.sidebar.button("Réinitialiser les filtres"):
        for key in [k for k in st.session_state if k.startswith("f_")]:
            del st.session_state[key]
        st.rerun()

    with st.sidebar.expander("Facture et produit", expanded=True):
        invoice_number = st.text_input("N° de facture", key="f_invoice", placeholder="ex. 36258")
        product_code = st.text_input("Code produit", key="f_code", placeholder="ex. FUR-CH-4421")

    choices: dict = {}
    with st.sidebar.expander("Client et destination", expanded=True):
        for field, label in CHOICE_FILTERS[:4]:
            choices[field] = st.multiselect(label, sorted(data[field].dropna().unique()), key=f"f_{field}")
    with st.sidebar.expander("Produit et livraison", expanded=True):
        for field, label in CHOICE_FILTERS[4:]:
            choices[field] = st.multiselect(label, sorted(data[field].dropna().unique()), key=f"f_{field}")

    ranges, bounds = {}, {}
    with st.sidebar.expander("Montants", expanded=True):
        for field, label in RANGE_FILTERS:
            if field == "quantity":
                low, high = int(data[field].min()), int(data[field].max())
            else:
                low, high = float(data[field].min()), float(data[field].max())
            bounds[field] = (low, high)
            ranges[field] = st.slider(label, min_value=low, max_value=high, value=(low, high), key=f"f_{field}")

    with st.sidebar.expander("Période", expanded=False):
        first, last = date.fromisoformat(data["date"].min()), date.fromisoformat(data["date"].max())
        picked = st.date_input("Date de facture", value=(first, last), min_value=first, max_value=last, key="f_date")
        dates = tuple(picked) if isinstance(picked, (tuple, list)) and len(picked) == 2 else None
        if dates == (first, last):
            dates = None

    top_k = st.sidebar.slider("Résultats pour la recherche sémantique", 5, 50, 10, step=5)
    return choices, ranges, bounds, dates, product_code, invoice_number, top_k


def money(value: float) -> str:
    return f"${value:,.2f}"


def show_detail(row: pd.Series) -> None:
    st.subheader(f"Facture #{row['invoice_number']}")
    left, right = st.columns(2)
    left.markdown(f"**Bill To**  \n{row['bill_to']}")
    left.markdown(f"**Ship To**  \n{row['ship_to']}")
    left.markdown(f"**Mode de livraison**  \n{row['ship_mode']}")
    right.markdown(f"**Date**  \n{row['date']}")
    right.markdown(f"**Order ID**  \n{row['order_id']}")
    right.markdown(f"**Code produit**  \n{row['product_code']} ({row['sub_category']}, {row['category']})")
    st.markdown(f"**Article**  \n{row['item']}")
    a, b, c, d = st.columns(4)
    a.metric("Montant", money(row["amount"]), help=f"{row['quantity']} x {money(row['rate'])}")
    b.metric("Remise", money(row["discount"]), help=f"{row['discount_pct']:g} %")
    c.metric("Livraison", money(row["shipping"]))
    d.metric("Total", money(row["total"]))

    pdf = Path(os.environ.get("INVOICES_DIR", DEFAULT_DATA_DIR)) / str(row["source"])
    if pdf.exists():
        st.download_button("Télécharger le PDF d'origine", pdf.read_bytes(), file_name=pdf.name,
                           mime="application/pdf")
    with st.expander("Texte extrait du PDF"):
        st.text(row.get("document", ""))


def main() -> None:
    st.set_page_config(page_title="Factures SuperStore", page_icon="🧾", layout="wide")
    st.title("Recherche de factures")

    client = get_qdrant()
    if not client.collection_exists(COLLECTION_NAME) or client.count(COLLECTION_NAME).count == 0:
        st.error("La collection « invoices » est vide. Lancez d'abord : `python ingestion.py`")
        st.stop()
    data = load_all()
    st.caption(f"Collection Qdrant « {COLLECTION_NAME} » : {len(data)} factures")

    choices, ranges, bounds, dates, product_code, invoice_number, top_k = sidebar(data)

    question = st.text_input(
        "Recherche",
        placeholder="ex. chaises de bureau ergonomiques  |  total supérieur à 5 000 en Chine en 2013  |  discount over 20%",
        help="Texte libre : recherche sémantique. Les conditions reconnues dans la phrase "
             "(montants, pays, client, année...) sont ajoutées aux filtres.",
    )
    text_where, semantic_text = parse_question(question, get_vocabulary()) if question.strip() else (None, "")
    where = build_where(choices, ranges, bounds, dates, product_code, invoice_number, text_where)
    matching, shown = run_search(client, where, semantic_text, top_k)

    with st.expander("Filtre Qdrant appliqué", expanded=False):
        st.json(where or {"filtre": "aucun"})
        if semantic_text:
            st.write(f"Texte utilisé pour la recherche sémantique : « {semantic_text} »")

    if matching.empty:
        st.warning("Aucune facture ne correspond à ces filtres.")
        st.stop()

    # Question ciblée ("quantité de la facture 36258") : on répond seulement à ce qui est demandé.
    fields, aggregate = detect_request(semantic_text)
    if fields or aggregate:
        focus = matching if (where or aggregate) else shown
        focus = focus.sort_values("invoice_number", key=lambda col: col.astype(int)).reset_index(drop=True)
        with st.container(border=True):
            st.markdown("**Réponse**")
            st.text("\n".join(concise_answer(focus.to_dict("records"), fields, aggregate)))
        if fields and len(focus) > 1:
            st.dataframe(focus[["invoice_number"] + fields], hide_index=True,
                         column_config={"invoice_number": "N° facture", **{f: FIELD_LABELS[f] for f in fields}})
        if not st.toggle("Afficher tous les détails", value=False):
            st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Factures", f"{len(matching):,}".replace(",", " "))
    k2.metric("Somme des totaux", money(matching["total"].sum()))
    k3.metric("Total moyen", money(matching["total"].mean()))
    k4.metric("Remises accordées", money(matching["discount"].sum()))

    tab_results, tab_charts = st.tabs(["Résultats", "Analyse"])

    with tab_results:
        if semantic_text:
            st.caption(f"{len(shown)} factures les plus proches de la recherche, parmi les {len(matching)} filtrées. "
                       "Cliquez sur une ligne pour voir le détail.")
        else:
            st.caption("Triées par total décroissant. Cliquez sur une ligne pour voir le détail.")
        columns = (["score"] if "score" in shown.columns else []) + [c for c in TABLE_COLUMNS if c in shown.columns]
        event = st.dataframe(
            shown[columns], width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row",
            column_config={
                "invoice_number": "N° facture", "date": "Date", "bill_to": "Bill To", "ship_to": "Ship To",
                "ship_mode": "Livraison", "item": "Article", "product_code": "Code produit",
                "category": "Catégorie", "sub_category": "Sous-catégorie", "quantity": "Qté",
                "rate": st.column_config.NumberColumn("Prix unitaire", format="$%.2f"),
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "discount": st.column_config.NumberColumn("Discount", format="$%.2f"),
                "discount_pct": st.column_config.NumberColumn("Discount %", format="%d%%"),
                "shipping": st.column_config.NumberColumn("Shipping", format="$%.2f"),
                "total": st.column_config.NumberColumn("Total", format="$%.2f"),
                "order_id": "Order ID", "score": "Score",
            },
        )
        st.download_button("Exporter ces résultats (CSV)", shown[columns].to_csv(index=False).encode("utf-8-sig"),
                           file_name="factures_filtrees.csv", mime="text/csv")
        selected = event.selection.rows
        if selected:
            st.divider()
            show_detail(shown.iloc[selected[0]])

    with tab_charts:
        st.caption(f"Calculé sur les {len(matching)} factures filtrées.")
        left, right = st.columns(2)
        left.markdown("**Total par catégorie ($)**")
        left.bar_chart(matching.groupby("category")["total"].sum())
        right.markdown("**Total par mois ($)**")
        right.bar_chart(matching.groupby(matching["date"].str[:7])["total"].sum())
        left.markdown("**10 premiers pays par total ($)**")
        left.bar_chart(matching.groupby("ship_to_country")["total"].sum().nlargest(10), horizontal=True)
        right.markdown("**10 premiers clients par total ($)**")
        right.bar_chart(matching.groupby("bill_to")["total"].sum().nlargest(10), horizontal=True)


if __name__ == "__main__":
    main()