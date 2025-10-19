import os, re, json, time, io
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import requests
from bs4 import BeautifulSoup
import tldextract
import phonenumbers
from rapidfuzz import fuzz
import pandas as pd
from dotenv import load_dotenv

# Optionnel Google Sheets
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    HAS_GS = True
except Exception:
    HAS_GS = False

load_dotenv()

# =========================
# ENV (100% gratuit)
# =========================
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY")
GOOGLE_CSE_CX  = os.getenv("GOOGLE_CSE_CX")
PAPPERS_API_KEY = os.getenv("PAPPERS_API_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Prospects AgenceVille")

HEADERS = {"User-Agent": "Mozilla/5.0 (ProspectFinder/1.2)"}

# Footprints FR (élargis)
FOOTPRINTS = [
    "site réalisé par","site realisé par","site realise par",
    "propulsé par","propulse par",
    "créé par","cree par",
    "crédits","credits",
    "webmaster","agence web",
    "développé par","developpe par",
    "design :","design:",
    "site conçu par","site concu par",
    "créateur :","créateur:","createur :","createur:","Créateur :","Créateur:"
]

LEGAL_HINTS = ["crédits","credits","mentions légales","mentions legales","legal","impressum"]

RE_ADDRESS = re.compile(
    r"(\d{1,4}\s+(?:bis|ter|quater)?\s*[A-Za-zÀ-ÖØ-öø-ÿ'\-\. ]+"
    r"(?:rue|boulevard|bd|avenue|av|chemin|impasse|route|allée|allee|quai|cours|place|pl|esplanade|faubourg|fg|passage)"
    r"[A-Za-zÀ-ÖØ-öø-ÿ'\-\. ]*,?\s*\d{2}\s?\d{3}\s+[A-Za-zÀ-ÖØ-öø-ÿ'\-\. ]+)",
    re.IGNORECASE
)
RE_SIREN = re.compile(r"\b(\d{3}\s?\d{3}\s?\d{3})\b")
RE_SIRET = re.compile(r"\b(\d{3}\s?\d{3}\s?\d{3}\s?\d{5})\b")

# =========================
# UI
# =========================
st.set_page_config(page_title="Prospect Finder – Agence+Ville", page_icon="📇", layout="wide")
st.title("📇 Prospect Finder – Agence + Ville")
st.caption("Saisis un nom d'agence et une ville pour trouver les sites gérés par cette agence, avec coordonnées (100% gratuit : Google CSE + Pappers).")

with st.sidebar:
    st.header("⚙️ Options")
    max_pages = st.slider("Profondeur de recherche (pages CSE)", 1, 10, 3)
    threshold = st.slider("Seuil de correspondance (0-100)", 60, 95, 80)
    enable_fallback_no_city = st.toggle("Élargir sans la ville (fallback)", value=True)
    use_pappers = st.toggle("Enrichir avec Pappers", value=bool(PAPPERS_API_KEY))
    push_gsheet = st.toggle("Exporter vers Google Sheets", value=False)
    st.markdown("---")
    st.markdown("**Clés chargées :**")
    st.write({
        "Google CSE (gratuit)": bool(GOOGLE_CSE_KEY and GOOGLE_CSE_CX),
        "PAPPERS": bool(PAPPERS_API_KEY),
        "Google Sheets": bool(GOOGLE_SERVICE_ACCOUNT_JSON) and HAS_GS,
    })

col1, col2 = st.columns([2,1])
with col1:
    agency = st.text_input("Nom de l'agence concurrente", placeholder="Ex: Agence Web XYZ")
with col2:
    city = st.text_input("Ville", placeholder="Ex: Lyon")

run = st.button("🔍 Scanner", type="primary")

# Conserver le dernier résultat affiché (évite le “remonte en haut” après download)
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None

# =========================
# Utils
# =========================
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def fuzzy_contains(hay: str, needle: str, thr: int = 80) -> bool:
    hay = norm(hay).lower()
    needle = norm(needle).lower()
    return fuzz.partial_ratio(hay, needle) >= thr

def absolute_url(base, href):
    try:
        return requests.compat.urljoin(base, href)
    except Exception:
        return href

def extract_domain(url: str) -> str:
    try:
        e = tldextract.extract(url)
        return f"{e.domain}.{e.suffix}" if e.suffix else e.domain
    except Exception:
        return url

def format_phone_list(text: str):
    out = set()
    for match in phonenumbers.PhoneNumberMatcher(text, "FR"):
        out.add(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL))
    return list(out)

# Validation Luhn (INSEE)
def luhn_checksum(num: str) -> int:
    s = 0
    parity = len(num) % 2
    for i, ch in enumerate(num):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return s % 10

def is_valid_siren(s: str) -> bool:
    return s.isdigit() and len(s) == 9 and luhn_checksum(s) == 0

def is_valid_siret(s: str) -> bool:
    return s.isdigit() and len(s) == 14 and luhn_checksum(s) == 0

# =========================
# Fetch & parse
# =========================
def fetch(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type",""):
            return r.text
    except Exception:
        pass
    return ""

def find_legal_links(soup, base_url):
    links = []
    for a in soup.select("a[href]"):
        txt = norm(a.get_text(" ").lower())
        href = a.get("href", "")
        if any(h in txt for h in LEGAL_HINTS):
            links.append(absolute_url(base_url, href))
    return list(dict.fromkeys(links))

def contains_agency_proof(html: str, agency: str, thr: int):
    soup = BeautifulSoup(html, "lxml")
    texts = []
    for f in soup.select("footer"):
        texts.append(f.get_text(" "))
    for sel in ["a","small",".credits","#credits","#credit"]:
        for el in soup.select(sel):
            texts.append(el.get_text(" "))
    body = soup.get_text(" ")
    texts.append(body)
    joined = " \n ".join(texts)

    has_fp = any(fp in joined.lower() for fp in [f.lower() for f in FOOTPRINTS])
    if has_fp and fuzzy_contains(joined, agency, thr):
        idx = joined.lower().find(agency.lower())
        snippet = joined[max(0, idx-80): idx+120] if idx != -1 else joined[:200]
        return True, norm(snippet)

    # Essaye liens "Mentions / Crédits"
    for lnk in find_legal_links(soup, "#"):
        h2 = fetch(lnk)
        if not h2:
            continue
        if any(fp in h2.lower() for fp in [f.lower() for f in FOOTPRINTS]) and fuzzy_contains(h2, agency, thr):
            idx = h2.lower().find(agency.lower())
            snippet = h2[max(0, idx-80): idx+120] if idx != -1 else h2[:200]
            return True, norm(snippet)
    return False, ""

def parse_jsonld_org(html: str):
    soup = BeautifulSoup(html, "lxml")
    org = {}
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        stack = data if isinstance(data, list) else [data]
        for node in stack:
            if not isinstance(node, dict):
                continue
            t = node.get("@type")
            if isinstance(t, list):
                ok = any(isinstance(x,str) and ("Organization" in x or "LocalBusiness" in x) for x in t)
            else:
                ok = t in ("Organization","LocalBusiness","Corporation","ProfessionalService")
            if ok:
                org["name"] = org.get("name") or node.get("name")
                addr = node.get("address") or {}
                if isinstance(addr, dict):
                    org["street"] = addr.get("streetAddress")
                    org["postalCode"] = addr.get("postalCode")
                    org["city"] = addr.get("addressLocality")
                tel = node.get("telephone") or node.get("phone")
                if tel:
                    org["tel"] = tel
    return org

def extract_legal(text: str):
    siren = None; siret = None
    m = RE_SIRET.search(text)
    if m:
        siret_cand = re.sub(r"\s","", m.group(1))
        if is_valid_siret(siret_cand):
            siret = siret_cand
            siren = siret[:9]
    if not siren:
        m2 = RE_SIREN.search(text)
        if m2:
            siren_cand = re.sub(r"\s","", m2.group(1))
            if is_valid_siren(siren_cand):
                siren = siren_cand

    addr = None
    m3 = RE_ADDRESS.search(text)
    if m3:
        addr = norm(m3.group(1))

    manager = None
    mm = re.search(r"(Directeur(?:rice)? de publication|Gérant|Gerant|Président|President)\s*[:\-–]\s*([A-Za-zÀ-ÖØ-öø-ÿ' \-\.]+)", text, re.IGNORECASE)
    if mm:
        manager = norm(mm.group(2))
    return siren, siret, addr, manager

def pappers_enrich(siren: str):
    if not (PAPPERS_API_KEY and siren):
        return {}
    try:
        r = requests.get(
            "https://api.pappers.fr/v2/entreprise",
            params={"api_token": PAPPERS_API_KEY, "siren": siren},
            timeout=25
        )
        if r.status_code == 200:
            d = r.json()
            dirigeants = []
            for dirg in d.get("dirigeants", [])[:3]:
                nom = norm(f"{dirg.get('prenom','')} {dirg.get('nom','')}")
                qual = dirg.get("qualite")
                dirigeants.append(f"{nom} ({qual})" if qual else nom)
            addr = d.get("siege", {})
            siege = norm(" ".join(filter(None, [
                addr.get("numero_voie"), addr.get("type_voie"), addr.get("nom_voie"),
                addr.get("code_postal"), addr.get("ville")
            ])))
            return {
                "dirigeants": ", ".join(dirigeants) if dirigeants else None,
                "adresse_officielle": siege or None,
                "denomination": d.get("denomination"),
                "siret_siege": d.get("siret_siege")
            }
    except Exception:
        pass
    return {}

# =========================
# Search (CSE gratuit)
# =========================
@st.cache_data(show_spinner=False)
def cse_search(agency: str, city: str, max_pages: int = 3):
    results = []
    if not (GOOGLE_CSE_KEY and GOOGLE_CSE_CX):
        return results
    for fp in FOOTPRINTS:
        start = 1
        for _ in range(max_pages):
            params = {
                "key": GOOGLE_CSE_KEY,
                "cx": GOOGLE_CSE_CX,
                "q": f'"{fp}" "{agency}" "{city}"' if city else f'"{fp}" "{agency}"',
                "num": 10,
                "start": start,
                "hl": "fr"
            }
            try:
                r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=30)
                data = r.json()
                for it in data.get("items", []):
                    link = it.get("link")
                    if link:
                        results.append({"url": link, "query": params["q"], "source": "cse"})
                start += 10
            except Exception:
                pass
            time.sleep(0.7)
    # dédup url
    seen=set(); uniq=[]
    for r in results:
        if r["url"] and r["url"] not in seen:
            uniq.append(r); seen.add(r["url"])
    return uniq

@st.cache_data(show_spinner=False)
def search_candidates(agency: str, city: str, max_pages: int = 3, enable_fallback: bool=True):
    # cible: agence + ville
    res = cse_search(agency, city, max_pages)
    # fallback: agence sans ville (on filtrera ensuite)
    if enable_fallback and len(res) < 20:
        res += cse_search(agency, "", max_pages)
    # dédup final
    seen=set(); uniq=[]
    for r in res:
        if r["url"] and r["url"] not in seen:
            uniq.append(r); seen.add(r["url"])
    return uniq

# =========================
# Scan d'une URL
# =========================
def scan_once(url: str, agency: str, thr: int = 80, city_filter: str = ""):
    html = fetch(url)
    if not html:
        return None
    ok, proof = contains_agency_proof(html, agency, thr)
    if not ok:
        return None

    # Filtre ville (si fallback sans ville a ramené des candidats)
    if city_filter and (city_filter.lower() not in html.lower()):
        soup = BeautifulSoup(html, "lxml")
        keep = False
        for lnk in find_legal_links(soup, url):
            h2 = fetch(lnk)
            if h2 and city_filter.lower() in h2.lower():
                keep = True
                break
        if not keep:
            return None

    phones = format_phone_list(html)
    org = parse_jsonld_org(html)
    siren, siret, addr, manager = extract_legal(html)

    tel = org.get("tel") if org.get("tel") else (phones[0] if phones else None)
    addr_guess = addr or norm(" ".join(filter(None, [org.get("street"), org.get("postalCode"), org.get("city")])))
    pap = pappers_enrich(siren) if PAPPERS_API_KEY else {}
    dirigeants = pap.get("dirigeants") or manager
    final_addr = pap.get("adresse_officielle") or addr_guess
    legal_name = pap.get("denomination") or org.get("name")

    score = 0
    if proof: score += 40
    if tel: score += 20
    if final_addr: score += 15
    if (siren or siret): score += 15
    if dirigeants: score += 10

    domain = extract_domain(url)

    return {
        "site_url": url,
        "domaine": domain,
        "raison_sociale": legal_name,
        "telephone": tel,
        "adresse": final_addr,
        "siren": siren,
        "siret": siret or pap.get("siret_siege"),
        "dirigeant": dirigeants,
        "preuve": proof,
        "score": score
    }

# =========================
# Exécution (parallélisée)
# =========================
@st.cache_data(show_spinner=True)
def run_scan(agency: str, city: str, max_pages: int, thr: int, enable_fallback: bool):
    cands = search_candidates(agency, city, max_pages, enable_fallback)
    rows = []
    seen_domains = set()

    def work(item):
        try:
            return scan_once(item["url"], agency, thr, city_filter=city)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(work, it) for it in cands]
        for fut in as_completed(futures):
            rec = fut.result()
            if not rec:
                continue
            if rec["domaine"] in seen_domains:
                continue
            seen_domains.add(rec["domaine"])
            rows.append(rec)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["score","domaine"], ascending=[False, True])
    return df, len(cands)

def push_to_gsheet(df: pd.DataFrame, sheet_name: str):
    if not (HAS_GS and GOOGLE_SERVICE_ACCOUNT_JSON):
        st.error("Google Sheets non configuré.")
        return False
    try:
        sa_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
        gc = gspread.authorize(creds)
        try:
            sh = gc.open(sheet_name)
        except Exception:
            sh = gc.create(sheet_name)
        try:
            ws = sh.worksheet("Prospects")
        except Exception:
            ws = sh.add_worksheet(title="Prospects", rows="100", cols="20")
        ws.clear()
        ws.update([df.columns.tolist()] + df.astype(str).values.tolist())
        return True
    except Exception as e:
        st.error(f"Erreur Google Sheets: {e}")
        return False

# =========================
# Run button
# =========================
if run:
    if not (agency and city):
        st.warning("Saisis l'agence et la ville.")
        st.stop()
    if not (GOOGLE_CSE_KEY and GOOGLE_CSE_CX):
        st.error("Configure GOOGLE_CSE_KEY et GOOGLE_CSE_CX (Google Programmable Search).")
        st.stop()

    with st.spinner("Recherche en cours…"):
        df, nb_cands = run_scan(agency, city, max_pages, threshold, enable_fallback_no_city)

    if df is None or df.empty:
        st.info(f"Aucun site trouvé avec preuve suffisante. (Candidats CSE: {nb_cands})")
    else:
        st.success(f"{len(df)} site(s) trouvé(s) • {nb_cands} URL(s) candidates avant filtrage.")
        st.session_state["last_results"] = df

# =========================
# Affichage persistant (évite le scroll au top après download)
# =========================
df_to_show = st.session_state.get("last_results")
if df_to_show is not None and not df_to_show.empty:
    st.dataframe(df_to_show, use_container_width=True)

    # Téléchargements (pas de retour haut car on réaffiche df_to_show après rerun)
    csv_bytes = df_to_show.to_csv(index=False).encode("utf-8")
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        df_to_show.to_excel(writer, index=False, sheet_name="Prospects")

    st.download_button("⬇️ Télécharger CSV", data=csv_bytes, file_name="prospects.csv", mime="text/csv")
    st.download_button("⬇️ Télécharger Excel",
                       data=xlsx_buf.getvalue(),
                       file_name="prospects.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if push_gsheet:
        ok = push_to_gsheet(df_to_show, GOOGLE_SHEET_NAME)
        if ok: st.toast("Exporté vers Google Sheets ✅")
        else:  st.toast("Échec export Google Sheets ❌")
