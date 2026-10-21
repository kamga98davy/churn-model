"""Churn Studio : qui va quitter la banque, pourquoi, et qui appeler en premier."""
from itertools import count
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

# ---------- Constantes ----------
ROOT = Path(__file__).resolve().parent
NUM = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
CAT = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"]
FEATURES = NUM + CAT
LABELS = {"CreditScore": "Score de crédit", "Age": "Âge", "Tenure": "Ancienneté (années)", "Balance": "Solde du compte",
          "EstimatedSalary": "Salaire estimé", "Geography": "Pays", "Gender": "Genre",
          "NumOfProducts": "Nombre de produits", "HasCrCard": "Carte de crédit", "IsActiveMember": "Membre actif"}
MODALITES = {"Geography": {"Germany": "Allemagne", "Spain": "Espagne"}, "Gender": {"Male": "Homme", "Female": "Femme"},
             "HasCrCard": {0: "non", 1: "oui"}, "IsActiveMember": {0: "non", 1: "oui"}}      # affichage en français
DESCRIPTION = ["Effectif", "Moyenne", "Écart-type", "Min", "Q1", "Médiane", "Q3", "Max"]   # colonnes de describe()

# Couleurs de mes carrousels : encre, gris, filet, fond, puis brun, bleu, vert, alerte et bleu nuit
INK, GREY, LINE = "#14181C", "#6B7280", "#E2E1DC"
C1, C2, C3, ALERT, DEEP = "#8C5A2B", "#1F5C8B", "#2E7D4F", "#B03030", "#123240"
CLAIR = "#C9C6BD"                                         # barres sous la moyenne
STATUT = {0: "Resté", 1: "Parti"}
COULEURS = {"Resté": C2, "Parti": ALERT}
SEGMENTS = ["Risque élevé", "Risque moyen", "Risque faible"]
COULEURS_SEGMENTS = {"Risque élevé": ALERT, "Risque moyen": C1, "Risque faible": C3}
CLASSE_SEGMENTS = {"Risque élevé": "a", "Risque moyen": "b", "Risque faible": "i"}
PAGES = ["Vue d'ensemble", "Explorer les clients", "Analyses inférentielles", "Ciblage commercial",
         "Simuler un client", "Le modèle", "Documentation"]
SANS_DONNEES = ("Résumé des données absent : lance l'application une fois en local, avec data/train_data.csv, pour créer "
                "modeles/resume_donnees.joblib, puis publie ce fichier avec le reste.")
TRAIN, TEST, MODELE, RESUME = ("data/train_data.csv", "data/test_data.csv", "modeles/modele_churn.joblib",
                               "modeles/resume_donnees.joblib")
VERSION_RESUME = 1                                         # à changer si la structure du résumé change
SANS_VALIDATION = "Relance la cellule de sauvegarde du notebook (version avec la validation) pour afficher cette partie."

# Tranches lisibles pour expliquer le départ : bornes (None = modalités), libellés, début et fin de phrase
FACTEURS = {
    "Age": ([0, 30, 40, 50, 60, np.inf], ["moins de 30 ans", "30 à 39 ans", "40 à 49 ans", "50 à 59 ans", "60 ans ou plus"],
            "les ", ""),
    "NumOfProducts": (None, {1: "1 produit", 2: "2 produits", 3: "3 produits", 4: "4 produits"}, "les clients à ", ""),
    "IsActiveMember": (None, {0: "inactifs", 1: "actifs"}, "les clients ", ""),
    "Geography": (None, {"France": "France", "Germany": "Allemagne", "Spain": "Espagne"}, "les clients en ", ""),
    "Gender": (None, {"Female": "femmes", "Male": "hommes"}, "les ", ""),
    "Balance": ([-np.inf, 0.01, 100_000, 150_000, np.inf], ["nul", "inférieur à 100 000", "de 100 000 à 150 000",
                                                            "de 150 000 ou plus"], "les clients au solde ", ""),
    "CreditScore": ([0, 580, 670, 740, 800, np.inf], ["inférieur à 580", "de 580 à 669", "de 670 à 739", "de 740 à 799",
                                                      "de 800 ou plus"], "les clients au score de crédit ", ""),
    "Tenure": ([0, 3, 6, 9, np.inf], ["0 à 2 ans", "3 à 5 ans", "6 à 8 ans", "9 ans ou plus"], "les clients de ",
               " d'ancienneté"),
    "EstimatedSalary": ([-np.inf, 50_000, 100_000, 150_000, np.inf], ["inférieur à 50 000", "de 50 000 à 100 000",
                                                                      "de 100 000 à 150 000", "de 150 000 ou plus"],
                        "les clients au salaire estimé ", ""),
    "HasCrCard": (None, {0: "sans carte de crédit", 1: "avec carte de crédit"}, "les clients ", ""),
}
# Ce que le marketing peut en faire (valable quel que soit le sens de l'écart observé)
LECTURES = {
    "Age": "L'âge ne se change pas : il sert à choisir le discours et le canal.",
    "Geography": "Un écart entre pays renvoie plutôt à la concurrence ou aux conditions locales : à creuser avec les équipes sur place.",
    "Gender": "Le genre fait partie des données de la compétition. Pour une vraie campagne, je ne m'en servirais pas pour cibler : "
              "je vérifierais plutôt que le modèle traite hommes et femmes de la même façon.",
    "Balance": "Le solde mesure aussi ce que la banque perd au départ du client : c'est lui qui sert à prioriser les appels.",
}
EN_LETTRES = {2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept", 8: "huit", 9: "neuf", 10: "dix"}
# Traits utilisés pour faire le portrait des segments
TRAITS = {"clients inactifs": lambda d: d["IsActiveMember"] == 0,
          "clients en Allemagne": lambda d: d["Geography"] == "Germany",
          "clients de 50 ans ou plus": lambda d: d["Age"] >= 50,
          "clients à un seul produit": lambda d: d["NumOfProducts"] == 1,
          "clients à 3 produits ou plus": lambda d: d["NumOfProducts"] >= 3,
          "clients avec un solde non nul": lambda d: d["Balance"] > 0}
# Hyperparamètres principaux, expliqués en clair (XGBoost, gradient boosting, forêt, logistique, k plus proches voisins)
REGLAGES = {"n_estimators": "Nombre d'arbres", "max_iter": "Nombre d'arbres", "learning_rate": "Pas d'apprentissage",
            "max_depth": "Profondeur maximale d'un arbre", "max_leaf_nodes": "Feuilles maximum par arbre",
            "min_child_weight": "Poids minimum d'une feuille", "min_samples_leaf": "Clients minimum par feuille",
            "subsample": "Part des clients tirés pour chaque arbre",
            "colsample_bytree": "Part des variables tirées pour chaque arbre", "max_features": "Part des variables tirées",
            "gamma": "Gain minimum pour couper une feuille", "reg_alpha": "Pénalité L1", "reg_lambda": "Pénalité L2",
            "l2_regularization": "Pénalité L2", "C": "Inverse de la pénalité", "n_neighbors": "Nombre de voisins"}
TRAITEMENTS = {"OneHotEncoder": 'Indicatrices 0/1 (<span style="white-space:nowrap">one-hot</span>)',
               "OrdinalEncoder": "Codage en entiers",
               "StandardScaler": "Centrage-réduction", "passthrough": "Valeurs brutes", "drop": "Ignorées par le modèle"}
# Les variables côté métier : ce que c'est, et ce que ça dit du client
METIER = [
    ("CreditScore", "Score de crédit", "La solidité financière du client vue par la banque : sa capacité à rembourser."),
    ("Age", "Âge", "L'étape de vie : premier emploi, installation, patrimoine, retraite. Les besoins changent avec elle."),
    ("Tenure", "Ancienneté, en années", "Depuis combien de temps le client est à la banque."),
    ("Balance", "Solde du compte", "Ce que la banque perd si le client part. La devise n'est pas précisée par la source."),
    ("EstimatedSalary", "Salaire estimé", "Le revenu estimé du client ; la source ne dit pas comment il est calculé."),
    ("Geography", "Pays", "Le marché du client (France, Allemagne ou Espagne), avec sa concurrence et ses offres."),
    ("Gender", "Genre", "Donnée sensible, à réserver au contrôle de l'équité du modèle."),
    ("NumOfProducts", "Nombre de produits (1 à 4)", "Le degré d'équipement : compte, carte, épargne, crédit."),
    ("HasCrCard", "Carte de crédit (0 ou 1)", "Un usage quotidien de la banque."),
    ("IsActiveMember", "Membre actif (0 ou 1)",
     "Le client utilise-t-il encore ses comptes ? La source ne définit pas « actif »."),
    ("Exited", "Départ, la cible (0 ou 1)", "1 si le client a quitté la banque. C'est ce que le modèle prédit."),
    ("ID, CustomerId, Surname", "Identifiants et nom", "Ils ne disent rien du comportement du client."),
]
# La page est en français mais Streamlit la déclare en anglais : le navigateur la traduit et casse l'affichage (removeChild)
SANS_TRADUCTION = """<script>
const d = window.parent.document;
d.documentElement.lang = "fr";
d.documentElement.setAttribute("translate", "no");
d.documentElement.classList.add("notranslate");
if (!d.querySelector('meta[name="google"]')) {
  const m = d.createElement("meta"); m.name = "google"; m.content = "notranslate"; d.head.appendChild(m);
}
</script>"""

# ---------- Style : le système de mes carrousels, adapté à l'écran ----------
CSS = """<style>
:root{--ink:#14181C;--grey:#6B7280;--line:#E2E1DC;--bg:#FBFAF7;--c1:#8C5A2B;--c2:#1F5C8B;--c3:#2E7D4F;--alert:#B03030;--deep:#123240;}
.stApp{background:var(--bg);}
.block-container{padding-top:2rem;max-width:1280px;}
.kicker{font-size:.76rem;letter-spacing:.15em;text-transform:uppercase;color:var(--grey);font-weight:700;}
.hero{background:var(--deep);border-radius:18px;padding:40px 40px 34px;margin-bottom:18px;position:relative;overflow:hidden;}
.hero:before{content:"";position:absolute;top:0;left:0;right:0;height:9px;background:var(--c2);}
.hero .kicker{color:#9FC3D9;}
.hero .titre{color:#fff;font-size:2.7rem;line-height:1.06;font-weight:700;letter-spacing:-.02em;margin:10px 0 14px;}
.hero p{color:#D3E2EA;font-size:1.08rem;max-width:860px;margin:0;}
.entete{border-left:9px solid var(--deep);padding:2px 0 2px 18px;margin:4px 0 22px;}
.entete .titre{font-size:2rem;line-height:1.12;font-weight:700;letter-spacing:-.015em;color:var(--ink);margin:4px 0 6px;}
.entete p{color:#2C3238;font-size:1.03rem;margin:0;max-width:900px;}
.reponse{display:flex;gap:14px;margin:2px 0 20px;flex-wrap:wrap;}
.card{flex:1;min-width:220px;background:#fff;border:2px solid var(--line);border-radius:16px;padding:16px 20px;}
.card.a{border-color:var(--alert)} .card.s2{border-color:var(--c2)} .card.i{border-color:var(--c3)} .card.b{border-color:var(--c1)} .card.d{border-color:var(--deep)}
.card .v{font-size:1.9rem;font-weight:700;line-height:1.12;color:var(--ink);margin:8px 0 4px;}
.card .v.long{font-size:1.4rem;line-height:1.2;}
.card .t{font-size:.95rem;color:#2C3238;line-height:1.38;}
.tag{display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:4px 11px;border-radius:999px;color:#fff;white-space:nowrap;}
.tag.a{background:var(--alert)} .tag.b{background:var(--c1)} .tag.i{background:var(--c3)} .tag.s2{background:var(--c2)} .tag.d{background:var(--deep)}
.chain{display:flex;flex-direction:column;gap:9px;margin:6px 0 12px;}
.ch{display:flex;gap:14px;align-items:flex-start;background:#fff;border:1.5px solid var(--line);border-radius:12px;padding:11px 16px;}
.ch .st{flex:0 0 28px;height:28px;border-radius:50%;background:var(--deep);color:#fff;font-size:.85rem;font-weight:700;display:flex;align-items:center;justify-content:center;}
.ch div.x{line-height:1.42;padding-top:2px;}
.quote{font-size:1.32rem;line-height:1.3;font-weight:700;letter-spacing:-.01em;margin:16px 0 8px;color:var(--ink);}
.hl{background:linear-gradient(transparent 62%,#FFE08A 62%);}
.mini{display:flex;flex-direction:column;gap:8px;margin:6px 0 10px;}
.mini div{padding-left:26px;position:relative;line-height:1.45;}
.mini div:before{content:"→";position:absolute;left:0;color:var(--grey);font-weight:700;}
table.t{width:100%;border-collapse:collapse;font-size:.92rem;margin:4px 0 8px;}
table.t th{background:var(--deep);color:#fff;font-weight:700;text-align:left;padding:9px 11px;}
table.t td{padding:8px 11px;border-bottom:1px solid var(--line);vertical-align:top;background:#fff;}
table.t tr:nth-child(even) td{background:#F4F3EF;}
table.t .r{text-align:right;font-variant-numeric:tabular-nums;}
[class*="st-key-carte"]{background:#fff;border-radius:16px;}
[data-testid="stMetric"]{background:#fff;border:1.5px solid var(--line);border-radius:14px;padding:14px 18px;}
[data-testid="stMetricLabel"] p{color:var(--grey);font-weight:600;}
[data-testid="stSidebar"]{background:var(--deep);}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] .marque{color:#E6EDF1;}
[data-testid="stSidebar"] .marque{font-size:1.45rem;font-weight:700;letter-spacing:-.01em;margin:0 0 2px;}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
[data-testid="stSidebar"] [data-testid="stRadio"],[data-testid="stSidebar"] [data-testid="stRadioGroup"],
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div{width:100% !important;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]{padding:.5rem .75rem;border-radius:10px;width:100%;margin-bottom:2px;}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover{background:rgba(255,255,255,.08);}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]{background:rgba(255,255,255,.14);box-shadow:inset 3px 0 0 #FFE08A;}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child{display:none;}
.stButton button[kind="primary"]{background:var(--deep);border-color:var(--deep);}
</style>"""

pio.templates["carrousel"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Source Sans Pro, Source Sans 3, sans-serif", size=13, color=INK),
    title=dict(font=dict(size=16, color=INK), x=0.01, xanchor="left"),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", colorway=[C2, ALERT, C1, C3, DEEP, GREY],
    xaxis=dict(gridcolor="#ECEAE4", linecolor=LINE, zeroline=False, title_font=dict(color=GREY)),
    yaxis=dict(gridcolor="#ECEAE4", linecolor=LINE, zeroline=False, title_font=dict(color=GREY)),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title_text=""),
    separators=", ", margin=dict(l=10, r=10, t=64, b=10), height=380))
pio.templates.default = px.defaults.template = "plotly_white+carrousel"      # go.Figure et plotly express


# ---------- Petits outils d'affichage ----------
NBSP = " "
_cartes = count()


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace(".", ",") + NBSP + "%"


def nombre(x):
    return f"{x:,.0f}".replace(",", NBSP)


def decimal(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def euros(x):
    return ("-" if x < 0 else "") + nombre(abs(x)) + NBSP + "€"


def majuscule(s):
    return s[:1].upper() + s[1:]


def enumerer(elements):
    return elements[0] if len(elements) == 1 else ", ".join(elements[:-1]) + " et " + elements[-1]


def libelle(col, v):
    return str(MODALITES.get(col, {}).get(v, v))


def html(texte):
    st.markdown(texte, unsafe_allow_html=True)


def entete(kicker, titre, chapeau=""):
    html(f'<div class="entete"><div class="kicker">{kicker}</div><div class="titre">{titre}</div>'
         + (f"<p>{chapeau}</p>" if chapeau else "") + "</div>")


def carte():
    return st.container(border=True, key=f"carte_{next(_cartes)}")


def fleches(lignes):
    html('<div class="mini">' + "".join(f"<div>{l}</div>" for l in lignes) + "</div>")


def chute(texte):
    html(f'<div class="quote"><span class="hl">{texte}</span></div>')


def cartes_reponse(cartes):
    html('<div class="reponse">' + "".join(
        f'<div class="card {c}"><span class="tag {c}">{tag}</span><div class="v{" long" if len(v) > 12 else ""}">{v}</div>'
        f'<div class="t">{t}</div></div>' for c, tag, v, t in cartes) + "</div>")


def tableau_html(df, droite=()):
    th = "".join(f'<th class="{"r" if c in droite else ""}">{c}</th>' for c in df.columns)
    lignes = "".join("<tr>" + "".join(f'<td class="{"r" if c in droite else ""}">{v}</td>' for c, v in zip(df.columns, row))
                     + "</tr>" for row in df.itertuples(index=False))
    html(f'<table class="t"><thead><tr>{th}</tr></thead><tbody>{lignes}</tbody></table>')


def tableau(df, **kwargs):
    """st.dataframe avec les nombres au format de la langue du navigateur (virgule décimale en français)."""
    formats = {c: st.column_config.NumberColumn(format="localized") for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c])}
    st.dataframe(df, column_config=formats, **kwargs)


def plot(fig, hauteur=None):
    if hauteur:
        fig.update_layout(height=hauteur)
    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")    # sinon Streamlit impose le fond du thème
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def download(df, name, texte="Télécharger le tableau CSV"):
    st.download_button(texte, df.to_csv(index=False).encode("utf-8-sig"), name, "text/csv")


def barres_taux(t, base, titre, hauteur=300):
    """Taux de départ par tranche ; en rouge les tranches au-dessus de la moyenne du portefeuille."""
    fig = go.Figure(go.Bar(x=[str(i) for i in t.index], y=t["taux"], marker_color=[ALERT if v > base else CLAIR for v in t["taux"]],
                           text=[pct(v, 0) for v in t["taux"]], textposition="outside", customdata=t["clients"],
                           hovertemplate="%{x}<br>%{y:.1%} de départs<br>%{customdata:,} clients<extra></extra>"))
    fig.add_hline(y=base, line_dash="dash", line_color=GREY)
    fig.update_layout(title=f"{titre}<br><sup>pointillés : moyenne du portefeuille ({pct(base, 0)})</sup>", showlegend=False)
    fig.update_yaxes(tickformat=".0%", range=[0, max(t["taux"].max() * 1.25, base * 1.4)])
    fig.update_xaxes(type="category")
    plot(fig, hauteur)


# ---------- Calculs, indépendants de l'interface ----------
def validate_data(df):
    df = df.copy()
    missing = set(FEATURES + ["Exited"]) - set(df.columns)
    if missing:
        raise ValueError("Colonnes absentes : " + ", ".join(sorted(missing)))
    for c in NUM + ["Exited"]:
        original = df[c]
        df[c] = pd.to_numeric(original, errors="coerce")
        if (original.notna() & df[c].isna()).any():
            raise ValueError(f"{c} contient des valeurs non numériques.")
    if not df["Exited"].dropna().isin([0, 1]).all():
        raise ValueError("Exited doit être codée 0/1.")
    if np.isinf(df[NUM].to_numpy(dtype=float)).any():
        raise ValueError("Les variables numériques contiennent des valeurs infinies.")
    return df


def wilson(k, n):
    """Intervalle de confiance à 95 % d'une proportion (méthode de Wilson)."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(0.975)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    demi = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - demi), min(1.0, centre + demi)


def segment_rates(df, col):
    d = df.dropna(subset=["Exited"]).copy()
    d[col] = d[col].map(lambda v: libelle(col, v) if pd.notna(v) else "Non renseigné")
    out = d.groupby(col, observed=True)["Exited"].agg(n="size", departs="sum").reset_index()
    out["taux"] = out.departs / out.n
    intervalles = [wilson(k, n) for k, n in zip(out.departs, out.n)]
    out["borne_basse"] = [x[0] for x in intervalles]
    out["borne_haute"] = [x[1] for x in intervalles]
    return out.sort_values("taux", ascending=False)


def associations(df):
    rows = []
    d = df.dropna(subset=["Exited"])
    for c in NUM:
        a = d.loc[d.Exited == 1, c].dropna().to_numpy()
        b = d.loc[d.Exited == 0, c].dropna().to_numpy()
        if min(len(a), len(b)) < 2 or np.ptp(np.r_[a, b]) == 0:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        rows.append(dict(variable=c, test="Mann-Whitney", n=len(a) + len(b), p=p,
                         effet=2 * u / (len(a) * len(b)) - 1, mesure="Corrélation bisérielle de rang",
                         remarque="Effet positif : valeurs plus élevées chez les départs."))
    for c in CAT:
        tab = pd.crosstab(d[c], d.Exited)
        if min(tab.shape) < 2:
            continue
        chi, p, _, expected = stats.chi2_contingency(tab, correction=False)
        n = tab.to_numpy().sum()
        v = np.sqrt(chi / (n * min(tab.shape[0] - 1, tab.shape[1] - 1)))
        note, test = "Approximation du khi² ; effectifs attendus ≥ 5.", "Khi² de Pearson"
        if (expected < 5).any():
            if tab.shape == (2, 2):
                _, p = stats.fisher_exact(tab.to_numpy())
                test, note = "Fisher exact", "V de Cramér descriptif ; p exacte de Fisher."
            else:
                p = np.nan
                note = "Effectifs attendus faibles : p non publiée ; regrouper les modalités avec justification."
        rows.append(dict(variable=c, test=test, n=n, p=p, effet=v, mesure="V de Cramér", remarque=note))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_BH"] = np.nan                                  # p-valeurs ajustées par Benjamini-Hochberg
    idx = out.index[out.p.notna()]
    if len(idx):
        ordre = out.loc[idx, "p"].sort_values().index
        rangs = out.loc[ordre, "p"].to_numpy()
        q = np.minimum.accumulate((rangs * len(rangs) / np.arange(1, len(rangs) + 1))[::-1])[::-1]
        out.loc[ordre, "q_BH"] = np.minimum(q, 1)
    return out.sort_values("q_BH", na_position="last")


def format_p(p):
    if pd.isna(p):
        return "non calculée"
    if p < 1e-300:
        return "< 1e-300"
    return decimal(p, 3) if p >= 0.001 else f"{p:.1e}".replace(".", ",")


def tranches(df, col):
    bornes, libelles, _, _ = FACTEURS[col]
    if bornes is None:
        return df[col].map(lambda v: libelles.get(v))
    return pd.cut(df[col], bornes, labels=libelles, right=False)


def facteurs(data):
    """Taux de départ par tranche pour chaque variable, classées par écart entre tranche la plus et la moins exposée."""
    res = []
    for col, (bornes, libelles, _, _) in FACTEURS.items():
        t = data.groupby(tranches(data, col), observed=True)["Exited"].agg(clients="size", taux="mean")
        if bornes is None:
            t = t.reindex([l for l in libelles.values() if l in t.index])
        grandes = t[t["clients"] >= 0.01 * len(data)]           # tranches de moins de 1 % : hors classement
        if len(grandes) >= 2:
            res.append(dict(variable=col, detail=t, pire=grandes["taux"].idxmax(), t_pire=grandes["taux"].max(),
                            meilleur=grandes["taux"].idxmin(), t_meilleur=grandes["taux"].min(),
                            ecart=grandes["taux"].max() - grandes["taux"].min()))
    return sorted(res, key=lambda f: -f["ecart"])


def phrase_facteur(f, debut_phrase=True):
    _, _, debut, fin = FACTEURS[f["variable"]]
    phrase = (f"{debut}{f['pire']}{fin} partent à {pct(f['t_pire'], 0)}, contre {pct(f['t_meilleur'], 0)} "
              f"pour {debut}{f['meilleur']}{fin}.")
    return majuscule(phrase) if debut_phrase else phrase


def lecture(f):
    """Ce que le marketing peut faire de l'écart ; les deux leviers commerciaux sont lus dans les chiffres."""
    taux = f["detail"]["taux"]
    if f["variable"] == "NumOfProducts":
        morceaux = []
        if {"1 produit", "2 produits"} <= set(taux.index) and taux["2 produits"] < taux["1 produit"]:
            morceaux.append(f"Passer d'un à deux produits va de pair avec un départ bien plus rare "
                            f"({pct(taux['1 produit'], 0)} contre {pct(taux['2 produits'], 0)}) : c'est le terrain de la vente croisée.")
        if any(l in taux.index and taux[l] > taux.get("2 produits", 1) for l in ("3 produits", "4 produits")):
            morceaux.append("Au-delà de deux, le taux remonte : un client suréquipé est peut-être un client mal conseillé, "
                            "à vérifier avant de lui vendre davantage.")
        return " ".join(morceaux)
    if f["variable"] == "IsActiveMember" and f["pire"] == "inactifs":
        return ("Un client qui n'utilise plus son compte a souvent déjà décidé de partir : la relance doit arriver "
                "avant la clôture.")
    return LECTURES.get(f["variable"], "")


def table_deciles(y, p):
    val = pd.DataFrame({"proba": p, "parti": y})
    val["décile"] = pd.qcut(val["proba"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    t = val.groupby("décile").agg(clients=("parti", "size"), partis=("parti", "sum"), proba_moyenne=("proba", "mean"))
    t["taux"] = t["partis"] / t["clients"]
    t["lift"] = t["taux"] / y.mean()
    t["capte"] = t["partis"].cumsum() / t["partis"].sum()   # part des départs touchés, en cumulé
    return t


def validation(bundle):
    """Réponses et probabilités du modèle appris sans les clients de validation, et son seuil."""
    if bundle is None or "validation" not in bundle:
        return None
    v = bundle["validation"]
    return np.asarray(v["y"]).astype(int), np.asarray(v["proba"], dtype=float), float(v["seuil"])


def aligner(df, modalites):
    """Met les colonnes au format vu à l'entraînement (0/1 devient non/oui si le modèle a appris non/oui)."""
    df = df.copy()
    for col, valeurs in modalites.items():
        if col in df and all(isinstance(v, str) for v in valeurs) and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map({0: "non", 1: "oui"})
    return df


@st.cache_data
def scorer_portefeuille(_modele, test, modalites, cle_modele):
    """Probabilité de départ, décile, segment et perte attendue de chaque client du portefeuille."""
    idcol = "ID" if "ID" in test else test.columns[0]
    s = test[[idcol] + FEATURES].rename(columns={idcol: "ID"})
    s["proba"] = _modele.predict_proba(aligner(test[FEATURES], modalites))[:, 1]
    s["décile"] = pd.qcut(s["proba"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    s["segment"] = pd.cut(s["décile"], bins=[0, 2, 5, 10], labels=SEGMENTS).astype(str)
    s["perte_attendue"] = s["proba"] * s["Balance"]      # risque × valeur : pour prioriser dans un segment
    return s


def preparation(prep):
    """Traitement appliqué à chaque groupe de variables par le ColumnTransformer du pipeline."""
    lignes = []
    for _, trans, cols in getattr(prep, "transformers_", []):
        cols = [prep.feature_names_in_[c] if isinstance(c, (int, np.integer)) else c for c in cols]
        if len(cols):
            quoi = trans if isinstance(trans, str) else type(trans).__name__
            if quoi == "FunctionTransformer" and trans.func is None:     # « passthrough » une fois le pipeline entraîné
                quoi = "passthrough"
            lignes.append((TRAITEMENTS.get(quoi, quoi), ", ".join(map(str, cols))))
    return pd.DataFrame(lignes, columns=["Traitement", "Variables"])


def etapes_modele(bundle):
    pipe = getattr(bundle["modele"], "estimator_", bundle["modele"])     # pipeline sans la couche « seuil »
    return getattr(pipe, "steps", [("modele", pipe)])


def reglages(algo):
    params = algo.get_params()
    lignes = [(texte, k, f"{params[k]:.4g}" if isinstance(params[k], float) else str(params[k]))
              for k, texte in REGLAGES.items() if params.get(k) is not None]
    return pd.DataFrame(lignes, columns=["Réglage", "Paramètre", "Valeur"])


# ---------- Chargements ----------
@st.cache_resource
def load_bundle(path, modified):                          # « modified » recharge le modèle quand le fichier change
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "modele" not in obj:
        raise ValueError("Le joblib doit être un dictionnaire contenant la clé 'modele'.")
    seuil = float(obj.get("seuil", 0.5))
    if not np.isfinite(seuil) or not 0 <= seuil <= 1:
        raise ValueError("Seuil invalide dans le joblib.")
    return obj


def lire_csv(chemin):
    df = pd.read_csv(chemin)
    if df.shape[1] == 1:                                  # fichier séparé par des points-virgules
        df = pd.read_csv(chemin, sep=";")
    return df


@st.cache_data
def lire_donnees(chemin, modifie):                        # « modifie » relit le fichier quand il change
    return validate_data(lire_csv(chemin))


@st.cache_data
def lire_test(chemin, modifie):
    df = lire_csv(chemin)
    manque = set(FEATURES) - set(df.columns)
    if manque:
        raise ValueError("Colonnes absentes du fichier test : " + ", ".join(sorted(manque)))
    return df


def charger(nom, lecteur):
    chemin = ROOT / nom
    if chemin.exists():
        try:
            return lecteur(str(chemin), chemin.stat().st_mtime_ns)
        except Exception as e:
            st.error(f"Lecture de {nom} impossible : {e}")
    return None


def mtime(nom):
    chemin = ROOT / nom
    return chemin.stat().st_mtime_ns if chemin.exists() else None


# ---------- Résumé agrégé : ce qui permet aux pages de s'afficher en ligne, sans publier une seule ligne client ----------
def boite(x):
    """Quartiles et moustaches de Tukey : de quoi tracer une boîte à moustaches sans garder les données."""
    x = np.sort(x[~np.isnan(x)])
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    bas, haut = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return dict(q1=q1, median=med, q3=q3, lowerfence=x[x >= bas].min(), upperfence=x[x <= haut].max())


def histogramme(data, col):
    """Densité par statut sur des classes communes ; une classe par valeur pour les petites variables entières."""
    x = data[col].dropna().to_numpy(dtype=float)
    if np.all(x % 1 == 0) and len(np.unique(x)) <= 40:
        bords = np.arange(x.min() - 0.5, x.max() + 1.5)
    else:
        bords = np.histogram_bin_edges(x, bins=35)
    largeur = np.diff(bords)
    h = {"centres": (bords[:-1] + bords[1:]) / 2, "largeur": largeur}
    for code, statut in STATUT.items():
        v = data.loc[data["Exited"] == code, col].dropna().to_numpy(dtype=float)
        h[statut] = np.histogram(v, bins=bords)[0] / (len(v) * largeur)
    return h


@st.cache_data
def resume_train(chemin, modifie):
    """Tout ce que les pages montrent des clients dont on connaît l'issue, sous forme agrégée."""
    data = lire_donnees(chemin, modifie)
    statut = data["Exited"].map(STATUT).rename("Statut")
    explorer = {}
    for col in NUM:
        desc = data.groupby(statut)[col].describe().reindex(list(STATUT.values()))
        desc.columns = DESCRIPTION
        explorer[col] = {"histo": histogramme(data, col), "desc": desc,
                         "boites": {s: boite(data.loc[data["Exited"] == c, col].to_numpy(dtype=float))
                                    for c, s in STATUT.items()}}
    for col in CAT:
        explorer[col] = {"rates": segment_rates(data, col)}
    resume_num = data[NUM].describe().T.rename(index=LABELS)
    resume_num.columns = DESCRIPTION
    return {"n": len(data), "k": int(data["Exited"].sum()), "facteurs": facteurs(data),
            "associations": associations(data), "explorer": explorer, "resume_num": resume_num,
            "spearman": data[NUM].corr(method="spearman").rename(index=LABELS, columns=LABELS),
            "qualite": pd.DataFrame({"variable": data.columns, "manquants": data.isna().sum().values,
                                     "manquants (%)": data.isna().mean().values * 100}),
            "doublons": int(data.duplicated().sum()),
            "repetitions": {c: int(data[c].dropna().duplicated().sum()) for c in ["ID", "CustomerId"] if c in data}}


@st.cache_data
def resume_test(_s, cle):
    """Segments du portefeuille à traiter, sous forme agrégée ; la liste nominative, elle, reste en local."""
    s = _s
    seg = s.groupby("segment").agg(clients=("proba", "size"), proba=("proba", "mean"), departs=("proba", "sum"),
                                   perte=("perte_attendue", "sum")).reindex(SEGMENTS)
    for c in ["clients", "departs", "perte"]:
        seg[f"part_{c}"] = seg[c] / seg[c].sum()
    traits = pd.DataFrame({nom: f(s) for nom, f in TRAITS.items()})
    groupes = {g: s.index[s["segment"] == g] for g in SEGMENTS} | {"Portefeuille": s.index}
    profil = {g: {"age": s.loc[i, "Age"].median(), "solde": s.loc[i, "Balance"].mean(), **traits.loc[i].mean().to_dict()}
              for g, i in groupes.items()}
    return {"n": len(s), "seg": seg, "profil": profil}


@st.cache_resource
def lire_resume(chemin, modifie):
    return joblib.load(chemin)


def synchroniser_resume(train, portefeuille, cle):
    """En local, enregistre le résumé dans modeles/ ; en ligne, sans les données, relit celui qui a été publié."""
    fichier = ROOT / RESUME
    ancien = {}
    if fichier.exists():
        try:
            ancien = lire_resume(str(fichier), fichier.stat().st_mtime_ns)
        except Exception:
            ancien = {}
    if train is None and portefeuille is None:
        return ancien.get("train"), ancien.get("test")
    nouveau = {"cle": cle, "train": train if train is not None else ancien.get("train"),
               "test": portefeuille if portefeuille is not None else ancien.get("test")}
    if ancien.get("cle") != cle:
        try:
            joblib.dump(nouveau, fichier)
        except OSError:
            pass
    return nouveau["train"], nouveau["test"]


# ---------- Pages ----------
def page_accueil():
    html("""<div class="hero"><div class="kicker">Projet data science | Churn bancaire</div>
<div class="titre">Qui va partir ?<br>Pourquoi ?<br>Qui appeler d'abord ?</div>
<p>Un score de départ pour chaque client d'une banque, et ce que le marketing peut en faire.</p></div>""")
    if R is None:
        st.info(SANS_DONNEES)
        return
    n, k = R["n"], R["k"]
    taux = k / n
    fac = R["facteurs"]
    val = validation(bundle)
    cartes = [("d", "Le constat", pct(taux), f"des clients sont partis : {nombre(k)} sur {nombre(n)}.")]
    if val is not None:
        g20 = table_deciles(*val[:2])["capte"].loc[2]
        cartes.append(("s2", "Le modèle", pct(g20, 0), "des départs se trouvent dans les 20 % de clients que le modèle "
                                                         "classe en tête. Au hasard, ce serait 20 %."))
    if fac:
        cartes.append(("a", "Le facteur n°1", LABELS[fac[0]["variable"]], phrase_facteur(fac[0])))
    cartes_reponse(cartes)

    gauche, droite = st.columns([1.15, 1], gap="large")
    with gauche:
        html('<div class="kicker">Le contexte</div>')
        un_sur = EN_LETTRES.get(round(1 / taux), str(round(1 / taux)))
        st.markdown(f"""Dans ce portefeuille, {pct(taux)} des clients sont partis. Un sur {un_sur}.

Pour une banque, chaque départ coûte deux fois. Le solde du client sort de la maison, puis il faut payer pour attirer \
quelqu'un qui le remplace. Retenir revient moins cher, à condition de savoir qui appeler. Appeler tout le monde \
épuise le budget de la campagne, et appeler au hasard, c'est tomber sur un partant tous les {un_sur} appels.

J'ai déjà fait ce travail pendant mon stage chez Orange Cameroun, sur des abonnés télécom : un score de départ par \
client, rangé en déciles, et une courbe de lift pour montrer au marketing où concentrer ses appels. Je refais ici la \
démarche pour une banque, avec les données de la compétition Kaggle *mlpro-classification-bank-churn-2025* : \
{nombre(n)} clients en France, en Allemagne et en Espagne, dont on sait s'ils sont partis.""")
        if val is not None and bundle is not None:
            premier = table_deciles(*val[:2])["taux"].loc[1]
            st.markdown(f"Le modèle retenu, un {bundle['nom']}, donne à chaque client une probabilité de départ. "
                        f"Je l'ai testé sur {nombre(len(val[0]))} clients qu'il n'avait jamais vus.")
            chute(f"Dans les 10 % de clients en tête de liste, {round(premier * 10)} appels sur{NBSP}10 tombent sur un "
                  f"futur partant. Au hasard, {round(taux * 10)} sur{NBSP}10.")
        st.caption("Deux limites, rappelées là où elles comptent : la source ne précise ni la devise des montants, ni la "
                   "période sur laquelle le départ est observé. Et le modèle mesure des associations : il dit qui risque "
                   "de partir, une campagne testée avec un groupe témoin dira ce qui le retient.")
    with droite:
        with carte():
            cible = pd.DataFrame({"Statut": list(STATUT.values()), "Clients": [n - k, k]})
            cible["Proportion"] = cible["Clients"] / n
            fig = go.Figure(go.Bar(x=cible["Statut"], y=cible["Proportion"], marker_color=[COULEURS[s] for s in cible["Statut"]],
                                   text=[pct(v) for v in cible["Proportion"]], textposition="outside",
                                   customdata=cible["Clients"], hovertemplate="%{x} : %{customdata:,} clients<extra></extra>"))
            fig.update_layout(title="Répartition de la cible", showlegend=False)
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            plot(fig, 300)
            lo, hi = wilson(k, n)
            st.caption(f"Taux de départ : {pct(taux)}, intervalle de confiance à 95 % (Wilson) de {pct(lo)} à {pct(hi)}. "
                       "Il suppose des observations indépendantes.")
        with carte():
            rates = R["explorer"]["Geography"]["rates"].set_index("Geography").rename(columns={"n": "clients"})
            barres_taux(rates, taux, "Taux de départ selon le pays", 280)

    html('<div class="kicker" style="margin-top:14px">Ce que contient l\'application</div>')
    fleches(["<b>Explorer les clients</b> : distributions et taux de départ, une variable à la fois.",
             "<b>Analyses inférentielles</b> : tests statistiques, tailles d'effet et correction des tests multiples.",
             "<b>Ciblage commercial</b> : pourquoi les clients partent, ce que le modèle apporte, qui appeler et combien.",
             "<b>Simuler un client</b> : le score d'un profil et les leviers qui le font baisser.",
             "<b>Le modèle</b> : performances sur des clients jamais vus, calibration.",
             "<b>Documentation</b> : variables côté métier, préparation des données, méthodologie."])
    with st.expander("Qualité des données"):
        tableau(R["qualite"], hide_index=True)
        st.caption(f"{R['doublons']} lignes entièrement dupliquées. Les identifiants ne servent pas à prédire. "
                   "Les doublons ne sont pas supprimés automatiquement.")
        for idcol, repetitions in R["repetitions"].items():
            st.caption(f"{idcol} : {nombre(repetitions)} répétitions au-delà de la première occurrence ; vérifier "
                       "l'indépendance des observations.")


def page_explorer():
    entete("Exploration", "Profils, distributions et segments",
           "Une variable à la fois : comment se répartissent les clients restés et les clients partis.")
    if R is None:
        st.info(SANS_DONNEES)
        return
    variable = st.selectbox("Variable à explorer", NUM + CAT, format_func=LABELS.get)
    E = R["explorer"][variable]
    if variable in NUM:
        gauche, droite = st.columns(2)
        with gauche, carte():
            fig = go.Figure()
            for statut in STATUT.values():
                fig.add_bar(x=E["histo"]["centres"], y=E["histo"][statut], width=E["histo"]["largeur"], name=statut,
                            marker_color=COULEURS[statut], opacity=0.65,
                            hovertemplate=f"{statut}<br>%{{x:,.0f}}<br>densité %{{y:.3g}}<extra></extra>")
            fig.update_layout(barmode="overlay", bargap=0, title="Distribution par statut")
            fig.update_xaxes(title=LABELS[variable])
            fig.update_yaxes(title="Densité")
            plot(fig)
        with droite, carte():
            fig = go.Figure()
            for statut in STATUT.values():
                q = E["boites"][statut]
                fig.add_trace(go.Box(x=[statut], q1=[q["q1"]], median=[q["median"]], q3=[q["q3"]],
                                     lowerfence=[q["lowerfence"]], upperfence=[q["upperfence"]], name=statut,
                                     marker_color=COULEURS[statut], boxpoints=False))
            fig.update_layout(title="Dispersion par statut<br><sup>boîte : quartiles ; moustaches : 1,5 écart "
                                    "interquartile</sup>", showlegend=False)
            fig.update_yaxes(title=LABELS[variable])
            plot(fig)
        # La cible sert seulement à former les deux groupes : ce tableau décrit la variable choisie, pas la cible
        desc = E["desc"]
        st.markdown(f"**{LABELS[variable]} : clients restés et partis**")
        tableau(desc.round(2))
        download(desc.reset_index(), f"description_{variable}.csv")
    else:
        rates = E["rates"]
        with carte():
            base = R["k"] / R["n"]
            fig = go.Figure(go.Bar(x=rates[variable], y=rates["taux"], marker_color=[ALERT if v > base else CLAIR for v in rates["taux"]],
                                   error_y=dict(type="data", symmetric=False, array=rates["borne_haute"] - rates["taux"],
                                                arrayminus=rates["taux"] - rates["borne_basse"], color=GREY),
                                   customdata=rates["n"], hovertemplate="%{x}<br>%{y:.1%} de départs<br>%{customdata:,} clients<extra></extra>"))
            fig.add_hline(y=base, line_dash="dash", line_color=GREY)
            fig.update_layout(title=f"Taux de départ selon : {LABELS[variable].lower()}<br><sup>barres d'erreur : IC à 95 % "
                                    f"de Wilson ; pointillés : moyenne du portefeuille ({pct(base, 0)})</sup>")
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(type="category", title=LABELS[variable])
            plot(fig)
        tableau(rates, hide_index=True)
        download(rates, f"segments_{variable}.csv")
    st.markdown("#### Statistiques numériques et corrélations")
    tableau(R["resume_num"].round(2))
    corr = R["spearman"]
    with carte():
        plot(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=[[0, C2], [0.5, "#FFFFFF"], [1, ALERT]],
                       title="Corrélations de Spearman entre variables numériques"), 440)
    st.caption("Corrélations calculées par paires disponibles. Une association ne démontre pas une relation causale.")


def page_analyses():
    entete("Statistique", "Quelles variables sont associées au départ ?",
           "Les tests comparent les clients partis aux clients restés. Ils décrivent des associations marginales, "
           "sans ajustement sur les autres caractéristiques.")
    if R is None:
        st.info(SANS_DONNEES)
    else:
        results = R["associations"]
        vue = pd.DataFrame({"Variable": results["variable"].map(LABELS), "Test": results["test"],
                            "Effectif": results["n"].map(nombre), "p-valeur": results["p"].map(format_p),
                            "q (BH)": results["q_BH"].map(format_p), "Taille d'effet": results["effet"].map(lambda e: decimal(e, 3)),
                            "Mesure": results["mesure"], "Remarque": results["remarque"]})
        tableau_html(vue, droite=("Effectif", "p-valeur", "q (BH)", "Taille d'effet"))
        download(results, "tests_association.csv")
        st.info("Lire ensemble la taille d'effet, les effectifs et q (BH). Avec plus de 140 000 clients, une différence "
                "minuscule devient significative : c'est la taille d'effet qui dit si elle compte.")
    with st.expander("Méthode et interprétation", expanded=True):
        st.markdown("""- **Mann-Whitney** : compare les distributions des variables numériques. Il ne teste une différence de médiane que sous des hypothèses supplémentaires sur la forme des distributions.
- **Effet de rang signé** : de -1 à +1 ; une valeur positive indique des valeurs généralement plus élevées chez les clients partis. Les ex æquo sont pris en compte.
- **Khi² / Fisher** : recherche une association pour les variables catégorielles. Fisher remplace le khi² en tableau 2×2 si des effectifs attendus sont faibles. Dans les autres tableaux clairsemés, la p-valeur est laissée vide.
- **V de Cramér** : intensité descriptive de l'association, de 0 à 1, sans direction ; l'estimateur n'est pas corrigé du biais de petit échantillon.
- **q (BH)** : p-valeur ajustée par Benjamini-Hochberg sur les tests calculables. Un seuil de 5 % contrôle le taux de fausses découvertes sous les conditions usuelles d'indépendance ou de dépendance positive.
- Les analyses supposent des observations indépendantes. Les données manquantes sont exclues variable par variable. Les tests ne corrigent ni un biais de sélection, ni une dépendance entre clients, ni les différences de composition des groupes.""")


def page_ciblage():
    val = validation(bundle)
    t = table_deciles(*val[:2]) if val is not None else None
    g20 = t["capte"].loc[2] if t is not None else None
    entete("Note à l'équipe marketing", "Qui appeler en premier, et pourquoi",
           f"Les 20 % de clients que le modèle classe en tête concentrent {pct(g20, 0)} des départs. "
           "La campagne de rétention doit commencer par eux." if g20 is not None else
           "D'où vient le risque de départ, comment le modèle le repère, et qui appeler.")

    # 1. Les facteurs de départ, lus sur les clients dont on connaît l'issue
    st.markdown("### 1. Pourquoi nos clients partent")
    if R is None:
        st.info(SANS_DONNEES)
    else:
        fac = R["facteurs"][:4]
        base = R["k"] / R["n"]
        st.markdown(f"J'ai coupé chaque variable en tranches et calculé le taux de départ de chacune, sur les "
                    f"{nombre(R['n'])} clients dont on connaît l'issue. L'écart entre tranches est le plus fort pour "
                    "quatre variables :")
        fleches([f"<b>{LABELS[f['variable']]}</b> : {phrase_facteur(f, debut_phrase=False)} {lecture(f)}" for f in fac])
        colonnes = st.columns(2)
        for i, f in enumerate(fac):
            with colonnes[i % 2], carte():
                barres_taux(f["detail"], base, LABELS[f["variable"]], 270)
        st.caption(f"Taux de départ moyen : {pct(base)}. En rouge, les tranches au-dessus de la moyenne. Les tranches de "
                   "moins de 1 % des clients restent sur les graphiques mais sortent du classement. Chaque écart est lu "
                   "une variable à la fois, sans tenir compte des autres : c'est une association, à confirmer.")
        chute("Le risque se concentre sur quelques profils faciles à repérer. C'est ce qui rend le ciblage rentable.")

    # 2. Ce que le modèle apporte : lift et gain sur la validation
    st.markdown("### 2. Ce que le modèle apporte : la courbe de lift")
    if t is None:
        st.info(SANS_VALIDATION)
    else:
        y, p, _ = val
        base_v = y.mean()
        st.markdown(f"Pour tester le modèle, j'ai mis de côté {nombre(len(y))} clients dont on connaît l'issue et qu'il "
                    "n'a jamais vus. Je les range du plus risqué au moins risqué, puis je les coupe en dix paquets de même "
                    f"taille, les déciles. Dans le premier, {pct(t['taux'].loc[1])} des clients partent, contre "
                    f"{pct(base_v)} en moyenne. Le lift vaut donc {decimal(t['lift'].loc[1])} : ce paquet contient "
                    f"{decimal(t['lift'].loc[1])} fois plus de partants qu'un paquet tiré au hasard.")
        gauche, droite = st.columns(2)
        with gauche, carte():
            fig = go.Figure(go.Bar(x=t.index, y=t["lift"], marker_color=[DEEP if v >= 1 else CLAIR for v in t["lift"]],
                                   text=[decimal(v) for v in t["lift"]], textposition="outside",
                                   hovertemplate="décile %{x}<br>lift %{y:.2f}<extra></extra>"))
            fig.add_hline(y=1, line_dash="dash", line_color=GREY, annotation_text="hasard", annotation_position="top right",
                          annotation_font_color=GREY)
            fig.update_layout(title="Lift par décile", showlegend=False)
            fig.update_xaxes(title="Décile (1 = les 10 % les plus risqués)", dtick=1)
            fig.update_yaxes(title="Lift", range=[0, t["lift"].max() * 1.2])
            plot(fig)
        with droite, carte():
            x = np.r_[0, np.arange(10, 101, 10)] / 100
            fig = go.Figure()
            grille = np.linspace(0, 1, 101)
            fig.add_scatter(x=grille, y=np.minimum(grille / base_v, 1), mode="lines", name="modèle parfait",
                            line=dict(color=C3, dash="dot"))
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="au hasard", line=dict(color=GREY, dash="dash"))
            fig.add_scatter(x=x, y=np.r_[0, t["capte"]], mode="lines+markers", name="avec le modèle", line=dict(color=C2, width=3))
            fig.add_annotation(x=0.2, y=g20, text=f"20 % des clients → {pct(g20, 0)} des départs", showarrow=True,
                               arrowhead=2, ax=150, ay=120, font=dict(color=INK), bgcolor="#FFFFFF")
            fig.update_layout(title="Courbe de gain")
            fig.update_xaxes(title="Part des clients appelés, les plus risqués d'abord", tickformat=".0%")
            fig.update_yaxes(title="Part des départs touchés", tickformat=".0%")
            plot(fig)
        st.markdown(f"La courbe de gain cumule les déciles. Avec les 10 % du haut de la liste, on touche déjà "
                    f"{pct(t['capte'].loc[1], 0)} des départs ; avec 30 %, {pct(t['capte'].loc[3], 0)}.")
        captes = np.cumsum(y[np.argsort(-p, kind="stable")]) / y.sum()
        moitie = int(np.argmax(captes >= 0.5)) + 1                 # nombre de clients à appeler pour toucher 50 %
        chute(f"Pour toucher la moitié des départs, il suffit d'appeler {pct(moitie / len(y), 0)} des clients, "
              "ceux du haut de la liste.")

    # 3. Le portefeuille à traiter, scoré puis segmenté
    st.markdown("### 3. Qui appeler : trois segments")
    if T is None:
        st.info("Cette partie a besoin du modèle et du fichier data/test_data.csv (le portefeuille à traiter, sans réponse), "
                "ou du résumé publié dans modeles/.")
    else:
        seg = T["seg"]
        st.markdown(f"Le portefeuille à traiter compte {nombre(T['n'])} clients, dont on ne connaît pas encore l'issue. "
                    "Chacun reçoit une probabilité de départ, puis je les classe : les 20 % les plus risqués, "
                    "les 30 % suivants, et la moitié restante.")
        cartes_reponse([(CLASSE_SEGMENTS[g], g, f"{nombre(seg.loc[g, 'clients'])} clients",
                         f"Probabilité moyenne de départ : {pct(seg.loc[g, 'proba'], 0)}<br>"
                         f"Départs attendus : {nombre(seg.loc[g, 'departs'])}, soit {pct(seg.loc[g, 'part_departs'], 0)} du total<br>"
                         f"Perte attendue : {pct(seg.loc[g, 'part_perte'], 0)} du total") for g in SEGMENTS])
        with carte():
            fig = go.Figure()
            mesures = ["Clients", "Départs attendus", "Perte attendue"]
            for g in SEGMENTS:
                v = [seg.loc[g, "part_clients"], seg.loc[g, "part_departs"], seg.loc[g, "part_perte"]]
                fig.add_bar(y=mesures, x=v, orientation="h", name=g, marker_color=COULEURS_SEGMENTS[g],
                            text=[pct(x, 0) for x in v], textposition="inside", insidetextanchor="middle",
                            textfont=dict(color="white"), hovertemplate=f"{g}<br>%{{y}} : %{{x:.1%}}<extra></extra>")
            fig.update_layout(barmode="stack", title="Où se trouvent les départs et la perte attendue",
                              legend=dict(traceorder="normal"))
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            fig.update_yaxes(autorange="reversed")
            plot(fig, 290)
        h = seg.loc["Risque élevé"]
        st.markdown(f"Le segment à risque élevé pèse {pct(h['part_clients'], 0)} des clients, mais "
                    f"{pct(h['part_departs'], 0)} des départs attendus et {pct(h['part_perte'], 0)} de la perte attendue.")
        st.caption("Départs attendus : somme des probabilités du segment (la page Le modèle vérifie qu'elles sont bien "
                   "calibrées). Perte attendue : probabilité × solde, additionnée sur le segment.")

        # Portrait du segment à risque élevé, comparé à tout le portefeuille
        haut, tout = T["profil"]["Risque élevé"], T["profil"]["Portefeuille"]
        sur = sorted((n for n in TRAITS if tout[n] > 0 and haut[n] / tout[n] > 1.2 and haut[n] >= 0.2),
                     key=lambda n: -haut[n] / tout[n])[:3]
        if sur:
            morceaux = []
            for j, nom in enumerate(sur):
                dans, partout = pct(haut[nom], 0), pct(tout[nom], 0)
                morceaux.append(f"des {nom} ({dans} du segment, contre {partout} sur tout le portefeuille)" if j == 0
                                else f"des {nom} ({dans} contre {partout})")
            st.markdown("**Qui sont-ils ?** Le segment à risque élevé compte surtout " + enumerer(morceaux) + ".")
        profil = pd.DataFrame({"Indicateur": ["Âge médian", "Solde moyen"] + [majuscule(n) for n in TRAITS]})
        for g in SEGMENTS + ["Portefeuille"]:
            p = T["profil"][g]
            profil[g] = [f"{decimal(p['age'], 0)} ans", nombre(p["solde"])] + [pct(p[n], 0) for n in TRAITS]
        tableau_html(profil, droite=tuple(SEGMENTS + ["Portefeuille"]))

        st.markdown("**Le plan d'action**")
        tableau_html(pd.DataFrame([
            ('<span class="tag a">Risque élevé</span>', "Appel d'un conseiller dans les deux semaines, avec une offre de fidélisation.",
             "Trier par perte attendue : les gros soldes d'abord."),
            ('<span class="tag b">Risque moyen</span>', "Relance par e-mail ou dans l'application ; un deuxième produit pour les clients mono-produit.",
             "Campagne automatisée, coût faible."),
            ('<span class="tag i">Risque faible</span>', "Pas de campagne de rétention.", "Relation habituelle."),
            ('<span class="tag d">Groupe témoin</span>', "10 % de chaque segment, tirés au hasard, ne sont pas contactés.",
             "Seul moyen de mesurer ce que la campagne retient vraiment.")],
            columns=["Segment", "Action", "Priorité"]))

    # 4. Combien appeler : le profit attendu selon la profondeur du ciblage
    st.markdown("### 4. Combien appeler : le simulateur de campagne")
    if val is None:
        st.info(SANS_VALIDATION)
    else:
        y, p, _ = val
        taille = T["n"] if T is not None else 10_000
        st.markdown("Trois hypothèses, à remplacer par les vrais chiffres de la banque. Le simulateur appelle les clients "
                    "dans l'ordre du score et cherche la part du portefeuille qui rapporte le plus.")
        a, b, c = st.columns(3)
        cout = a.slider("Coût d'un contact (€)", 1, 50, 10)
        valeur = b.slider("Marge perdue quand un client part (€)", 50, 1000, 300, step=10)
        succes = c.slider("Part des partants que l'action retient (%)", 5, 60, 25) / 100
        ordre = np.argsort(-p, kind="stable")
        captes = np.cumsum(y[ordre])
        k = np.arange(1, 101)
        appeles = np.ceil(k / 100 * len(y)).astype(int)
        echelle = taille / len(y)
        gains = captes[appeles - 1] * succes * valeur * echelle
        couts = appeles * cout * echelle
        profit = gains - couts
        i = int(np.argmax(profit))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Part du portefeuille à appeler", f"{k[i]}{NBSP}%")
        m2.metric("Clients à appeler", nombre(appeles[i] * echelle))
        m3.metric("Profit attendu", euros(profit[i]))
        m4.metric("Départs évités", nombre(captes[appeles[i] - 1] * succes * echelle))
        with carte():
            fig = go.Figure()
            fig.add_scatter(x=k / 100, y=profit, mode="lines", line=dict(color=C2, width=3), name="profit",
                            hovertemplate="%{x:.0%} des clients appelés<br>profit %{y:,.0f} €<extra></extra>")
            fig.add_scatter(x=[k[i] / 100], y=[profit[i]], mode="markers", marker=dict(size=14, color=ALERT), name="optimum",
                            hovertemplate="optimum<extra></extra>")
            fig.add_hline(y=0, line_color=GREY)
            fig.update_layout(title=f"Profit de la campagne selon la part de clients appelés (optimum à {k[i]} %)",
                              showlegend=False)
            fig.update_xaxes(title="Part des clients appelés, les plus risqués d'abord", tickformat=".0%")
            fig.update_yaxes(title="Profit attendu (€)", tickformat=",.0f")
            plot(fig, 340)
        if profit[i] <= 0:
            st.warning("Avec ces hypothèses, aucune profondeur de campagne n'est rentable : le coût d'un contact dépasse "
                       "ce que rapporte un départ évité.")
        st.caption(f"Calcul sur les {nombre(len(y))} clients de validation, ramené à un portefeuille de {nombre(taille)} "
                   "clients. La part des partants que l'action retient est l'hypothèse la plus fragile : seule une "
                   "campagne avec groupe témoin la mesure.")

    # 5. La liste à transmettre
    st.markdown("### 5. La liste pour l'équipe commerciale")
    if s is None:
        st.info("La liste nominative n'est disponible qu'en local, avec data/test_data.csv : elle contient des lignes "
                "clients, qui ne sont pas publiées.")
    else:
        a, b = st.columns([2, 1])
        choix = a.multiselect("Segments", SEGMENTS, default=["Risque élevé"])
        tri = b.radio("Trier par", ["Probabilité de départ", "Perte attendue"], horizontal=True)
        liste = s[s["segment"].isin(choix)].sort_values("proba" if tri == "Probabilité de départ" else "perte_attendue",
                                                         ascending=False)
        vue = pd.DataFrame({"ID": liste["ID"], "Probabilité": liste["proba"] * 100, "Segment": liste["segment"],
                            "Solde": liste["Balance"].round(), "Perte attendue": liste["perte_attendue"].round(),
                            "Âge": liste["Age"],
                            "Pays": liste["Geography"].map(lambda v: libelle("Geography", v)),
                            "Produits": liste["NumOfProducts"],
                            "Actif": liste["IsActiveMember"].map(lambda v: libelle("IsActiveMember", v))})
        st.dataframe(vue.head(200), hide_index=True, column_config={
            "Probabilité": st.column_config.ProgressColumn("Probabilité", min_value=0, max_value=100, format="%.0f %%"),
            "Solde": st.column_config.NumberColumn(format="localized"),
            "Perte attendue": st.column_config.NumberColumn(format="localized"),
            "Âge": st.column_config.NumberColumn(format="%d")})
        st.caption(f"{nombre(len(liste))} clients dans la sélection ; les 200 premiers sont affichés, le fichier contient tout.")
        download(liste.drop(columns=["Surname"], errors="ignore").rename(columns={"proba": "proba_depart"}),
                 "clients_a_appeler.csv", "Télécharger la liste (CSV)")


def page_simulation():
    entete("Simulation", "Un profil, une estimation",
           "Renseigner un profil, lire sa probabilité de départ et ce qui la ferait baisser.")
    if bundle is None:
        st.info("Placez le modèle sauvegardé dans modeles/modele_churn.joblib pour activer la simulation.")
        return
    seuil = bundle["seuil"]
    st.caption(f"Modèle : {bundle['nom']} | seuil de décision : {pct(seuil)}")
    # Saisie du profil : curseurs et montants à gauche, choix à droite. Les identifiants (ID, nom) ne servent pas.
    client = {}
    with carte():
        gauche, droite = st.columns(2)
        with gauche:
            for col, (mini, maxi, mediane) in bundle["continues"].items():
                if col == "Age":                          # âge : boutons + et - de 10 ans
                    client[col] = st.number_input(LABELS[col], min_value=float(mini), max_value=float(maxi),
                                                  value=float(mediane), step=10.0, format="%.0f")
                elif col in bundle["entiers"]:
                    client[col] = st.slider(LABELS[col], int(mini), int(maxi), int(mediane))
                else:
                    client[col] = st.number_input(LABELS[col], min_value=0.0, value=float(mediane), step=1000.0,
                                                  format="%.0f")
        with droite:
            for col, valeurs in bundle["modalites"].items():
                client[col] = st.selectbox(LABELS[col], valeurs, format_func=lambda v, col=col: libelle(col, v))
        estimer = st.button("Estimer le risque", type="primary")

    if estimer:
        modele = bundle["modele"]
        proba = float(modele.predict_proba(pd.DataFrame([client]))[0, 1])
        part = (bundle["probas_portefeuille"] < proba).mean()       # part des clients moins risqués que lui
        segment = "Risque élevé" if part >= 0.8 else "Risque moyen" if part >= 0.5 else "Risque faible"
        with carte():
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilité de départ", pct(proba))
            c2.metric("Plus risqué que", f"{pct(part, 0)} des clients")
            c3.metric("Segment", segment)
            st.progress(proba)
            if proba >= seuil:
                st.error(f"Client à risque : sa probabilité dépasse le seuil de {pct(seuil)}.")
            else:
                st.success(f"Client plutôt fidèle : sa probabilité reste sous le seuil de {pct(seuil)}.")
            # Leviers commerciaux : ce que donne le modèle si l'on change l'activité ou l'équipement du client
            actif, inactif = bundle["modalites"]["IsActiveMember"][-1], bundle["modalites"]["IsActiveMember"][0]
            scenarios = []
            if client["IsActiveMember"] == inactif:
                scenarios.append(("Si le client redevient actif", {**client, "IsActiveMember": actif}))
            if client["NumOfProducts"] == 1 and 2 in bundle["modalites"]["NumOfProducts"]:
                scenarios.append(("S'il prend un deuxième produit", {**client, "NumOfProducts": 2}))
            if len(scenarios) == 2:
                scenarios.append(("S'il fait les deux", {**client, "IsActiveMember": actif, "NumOfProducts": 2}))
            if scenarios:
                st.markdown("**Ce que donne le modèle si l'on joue sur un levier commercial**")
                lignes = []
                for texte, profil in scenarios:
                    q = float(modele.predict_proba(pd.DataFrame([profil]))[0, 1])
                    lignes.append(f"{texte} : {pct(q)} au lieu de {pct(proba)} "
                                  f"({'+' if q >= proba else '-'}{decimal(abs(q - proba) * 100)} points).")
                fleches(lignes)
                st.caption("Le modèle mesure des associations : ces écarts montrent ce qui pèse dans le score. "
                           "L'effet réel d'une action se vérifie avec un groupe témoin.")
            else:
                st.caption("Ce client est déjà actif et a plusieurs produits : les deux leviers commerciaux du modèle "
                           "sont déjà utilisés.")


def page_modele():
    entete("Performance", "Le modèle et ses performances", "Les chiffres sur des clients que le modèle n'a jamais vus.")
    if bundle is None:
        st.info("Aucun modèle disponible. Ajoutez le joblib pour lire ses performances.")
        return
    etapes = etapes_modele(bundle)
    a, b, c = st.columns(3)
    a.metric("Algorithme", bundle["nom"])
    b.metric("Seuil de décision", pct(bundle["seuil"]))
    c.metric("Variables explicatives", len(bundle["continues"]) + len(bundle["modalites"]))
    st.caption("Seuil choisi par validation croisée pour maximiser le F1 : au-dessus, le client est classé à risque.")
    gauche, droite = st.columns(2)
    with gauche:
        st.markdown("**Préparation des variables**")
        if len(etapes) > 1:
            tableau_html(preparation(etapes[0][1]))
    with droite:
        st.markdown("**Réglages principaux** (choisis par validation croisée)")
        tableau_html(reglages(etapes[-1][1]), droite=("Valeur",))

    val = validation(bundle)
    if val is None:
        st.info(SANS_VALIDATION)
        return
    # Validation : 20 % du fichier mis de côté au départ, avec la vraie réponse, jamais vus par le modèle évalué
    y, p, seuil = val
    pred = (p >= seuil).astype(int)
    st.markdown("### Performances sur la validation")
    st.caption(f"{nombre(len(y))} clients mis de côté au départ, dont on connaît la réponse. Le modèle évalué a été "
               f"appris sans eux, seuil compris ({pct(seuil)}) ; le modèle déployé est le même, réappris sur tout le fichier.")
    m = st.columns(5)
    m[0].metric("Accuracy", pct(accuracy_score(y, pred)))
    m[1].metric("Précision", pct(precision_score(y, pred, zero_division=0)))
    m[2].metric("Rappel", pct(recall_score(y, pred)))
    m[3].metric("F1", decimal(f1_score(y, pred), 3))
    m[4].metric("AUC", decimal(roc_auc_score(y, p), 3))
    st.caption("Précision : part de vrais départs parmi les clients signalés. Rappel : part des départs détectés. "
               "AUC : probabilité qu'un partant tiré au hasard ait un score plus élevé qu'un client resté.")
    gauche, droite = st.columns(2)
    with gauche, carte():
        fpr, tpr, _ = roc_curve(y, p)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=C2, width=3), name="modèle",
                        hovertemplate="faux positifs %{x:.1%}<br>rappel %{y:.1%}<extra></extra>")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"), name="hasard")
        fig.update_layout(title=f"Courbe ROC (AUC = {decimal(roc_auc_score(y, p), 3)})")
        fig.update_xaxes(title="Taux de faux positifs")
        fig.update_yaxes(title="Taux de vrais positifs (rappel)")
        plot(fig)
    with droite, carte():
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig = px.imshow(cm, x=["Prédit resté", "Prédit parti"], y=["Réel resté", "Réel parti"], text_auto=",d",
                        color_continuous_scale=[[0, "#F4F3EF"], [1, DEEP]], title=f"Matrice de confusion (seuil {pct(seuil)})")
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textfont_size=15)
        plot(fig)
    t = table_deciles(y, p)
    with carte():
        fig = go.Figure()
        fig.add_scatter(x=[0, t["proba_moyenne"].max()], y=[0, t["proba_moyenne"].max()], mode="lines",
                        line=dict(color=GREY, dash="dash"), name="calibration parfaite")
        fig.add_scatter(x=t["proba_moyenne"], y=t["taux"], mode="lines+markers", line=dict(color=C2, width=3), name="modèle",
                        text=[f"décile {d}" for d in t.index],
                        hovertemplate="%{text}<br>annoncé %{x:.1%}<br>observé %{y:.1%}<extra></extra>")
        fig.update_layout(title="Calibration : probabilité annoncée et taux de départ observé, par décile")
        fig.update_xaxes(title="Probabilité moyenne annoncée par le modèle", tickformat=".0%")
        fig.update_yaxes(title="Taux de départ observé", tickformat=".0%")
        plot(fig, 360)
    ecart = (t["proba_moyenne"] - t["taux"]).abs().max()
    st.caption(f"Écart maximal entre probabilité annoncée et taux observé : {decimal(ecart * 100)} points. Des points "
               "proches de la diagonale autorisent à additionner les probabilités, comme le fait la page Ciblage "
               "commercial pour les départs attendus.")


def page_documentation():
    entete("Documentation", "Données, préparation et méthode", "Ce qu'il faut savoir pour relire le projet, ou le refaire.")
    st.markdown("### Le projet en bref")
    st.markdown("L'objectif : estimer, pour chaque client d'une banque, la probabilité qu'il parte, puis s'en servir pour "
                "cibler une campagne de rétention. Les données viennent de la compétition Kaggle "
                "*mlpro-classification-bank-churn-2025* : un fichier d'entraînement où l'on sait qui est parti, et un "
                "fichier test sans la réponse, qui joue ici le rôle du portefeuille à traiter.")

    st.markdown("### Les variables, côté métier")
    traitement = {}
    if bundle is not None:
        etapes = etapes_modele(bundle)
        if len(etapes) > 1:
            for _, ligne in preparation(etapes[0][1]).iterrows():
                for v in ligne["Variables"].split(", "):
                    traitement[v] = ligne["Traitement"]
    vue = pd.DataFrame([(v, quoi, sens, "Variable à prédire" if v == "Exited" else
                         "Retirés avant l'apprentissage" if v.startswith("ID") else traitement.get(v, ""))
                        for v, quoi, sens in METIER], columns=["Variable", "Ce que c'est", "Ce que ça dit du client", "Dans le modèle"])
    tableau_html(vue)

    st.markdown("### La préparation, en quelques lignes")
    st.markdown("""Les données arrivent propres : aucune valeur manquante. Les valeurs extrêmes repérées par la règle de \
l'écart interquartile (surtout des âges élevés, et quelques scores de crédit très bas) restent dans les données : ce \
sont des clients plausibles, et les arbres y sont peu sensibles.

J'ai rangé les variables en trois groupes. Les continues (score de crédit, âge, ancienneté, solde, salaire) gardent \
leur valeur brute pour les arbres, et sont centrées-réduites pour les modèles sensibles à l'échelle (régression \
logistique, k plus proches voisins, SVM). Le nombre de produits est une variable ordinale, codée en entiers dans \
l'ordre. Pays, genre, carte de crédit et activité deviennent des indicatrices 0/1 (one-hot), une modalité servant de \
référence.

Tout vit dans un pipeline scikit-learn. La préparation s'apprend sur les données d'entraînement seulement, puis \
s'applique telle quelle aux nouveaux clients. Aucune information de la validation ne passe donc dans le modèle. Le \
détail exact du pipeline retenu est affiché dans la page Le modèle.""")

    st.markdown("### La méthodologie")
    etapes_methode = [
        "Mettre de côté 20 % des clients, par tirage stratifié sur la cible. Ils servent à comparer les modèles et à "
        "contrôler le résultat final, jamais à régler les hyperparamètres ni le seuil.",
        "Explorer : distributions, valeurs extrêmes, tests de Mood et de Mann-Whitney pour les variables continues, "
        "khi² et V de Cramér pour les qualitatives.",
        "Comparer onze classifieurs classiques, chacun avec la préparation qui lui convient, sur le F1 : régression "
        "logistique, analyses discriminantes linéaire et quadratique, bayésien naïf, k plus proches voisins, SVM linéaire, "
        "arbre de décision, forêt aléatoire, gradient boosting, AdaBoost et XGBoost.",
        "Régler deux modèles de boosting, XGBoost et le gradient boosting de scikit-learn en version histogramme (plus "
        "rapide), par recherche aléatoire à élimination successive : validation croisée à 3 plis sur l'AUC, en "
        "surveillant l'écart entre le score d'entraînement et celui de validation croisée.",
        "Choisir le seuil de décision qui maximise le F1, lui aussi par validation croisée, plutôt que le 0,5 par défaut.",
        "Contrôler sur les 20 % mis de côté : métriques, courbe ROC, calibration, lift.",
        "Réentraîner le modèle retenu sur tout le fichier, scorer le portefeuille test, puis le segmenter pour le "
        "marketing : déciles, segments, perte attendue."]
    html('<div class="chain">' + "".join(f'<div class="ch"><div class="st">{i}</div><div class="x">{e}</div></div>'
                                         for i, e in enumerate(etapes_methode, 1)) + "</div>")

    st.markdown("### Limites")
    st.markdown("La provenance, la licence, les dates de collecte et l'horizon du départ restent à documenter. "
                "Les analyses marginales ne remplacent pas une analyse multivariée ajustée. Le modèle fournit une "
                "fonction prédictive, pas des effets causaux.")


# ---------- Mise en page et navigation ----------
st.set_page_config(page_title="Churn Studio · Banque", page_icon="🏦", layout="wide")
html(CSS)
with st.sidebar:
    html('<div class="marque">Churn Studio</div>')
    st.caption("Qui va partir, pourquoi, et qui appeler en premier.")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Davy Romaric Kamga Bopda · Master en sciences actuarielles, UCLouvain")
    st.caption("Données : compétition Kaggle mlpro-classification-bank-churn-2025")
    try:
        st.html(SANS_TRADUCTION, unsafe_allow_javascript=True)
    except TypeError:                                     # anciennes versions de Streamlit
        components.html(SANS_TRADUCTION, height=0)

# Données : en local, les CSV de data/ ; en ligne, le résumé agrégé publié dans modeles/ (aucune ligne client)
bundle = charger(MODELE, load_bundle)
test = charger(TEST, lire_test)
s = None                                                   # portefeuille scoré, ligne par ligne : en local seulement
if test is not None and bundle is not None:
    s = scorer_portefeuille(bundle["modele"], test, bundle["modalites"], mtime(MODELE))
R, T = synchroniser_resume(charger(TRAIN, resume_train), resume_test(s, (mtime(TEST), mtime(MODELE))) if s is not None else None,
                           (VERSION_RESUME, mtime(TRAIN), mtime(TEST), mtime(MODELE)))

{"Vue d'ensemble": page_accueil, "Explorer les clients": page_explorer, "Analyses inférentielles": page_analyses,
 "Ciblage commercial": page_ciblage, "Simuler un client": page_simulation, "Le modèle": page_modele,
 "Documentation": page_documentation}[page]()

st.divider()
st.caption("Churn Studio | Davy Romaric Kamga Bopda | Associations ≠ causalité")