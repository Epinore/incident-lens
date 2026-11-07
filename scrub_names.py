"""Mask company, contractor, rig/vessel and person names in the text the app shows.

The processed data keeps real operator names, and contractor and rig names are not masked at
all (see the dataset README). Incident Lens is about what happened, not who, so before the
artifacts are published we replace them with [company], [vessel] or [name].
Pattern- and list-based, so it is not perfect; run `python scrub_names.py --audit` to list leftovers.
Run after build.py / evaluate.py: rewrites artifacts/incidents.pkl and artifacts/oof.csv in place.
"""
import re
import sys
from pathlib import Path
import pandas as pd

ART = Path(__file__).parent / "artifacts"

OPERATORS = """Anadarko|Apache|Arena Offshore|Arena Energy|BOE Exploration|Beacon Offshore|Cantium|Chevron|Cox Operating|Cox Oil|Equinor|ExxonMobil|Exxon|
Fieldwood|GOM Shelf|Hess|High Point Gas Gathering|Kosmos|LLOG|Murphy|NuGulf|Renaissance Offshore|Sanare|Shell|Talos|W ?& ?T|Walter Oil|Whitney Oil|Ankor|ANKOR|
Energy XXI|EnVen|Byron Energy|Freeport-McMoRan|McMoRan|Marathon|Noble Energy|Petrobras|Stone Energy|Taylor Energy|Black Elk|Eni|ENI|Woodside|BHP|Occidental|
ConocoPhillips|Hilcorp|Contango|Helis|Kerr-McGee|Northstar|PetroQuest|Petroquest|QuarterNorth|Castex|Energy Resource Technology|EPL Oil|Energy Partners|Fairways Offshore|
Pisces Energy|Linder Oil|Sanchez|GulfSlope|Gulfslope|Peregrine Oil|White Fleet|Panaco|Dynamic Offshore|Ridgewood|Tana Exploration|Bennu|Monforte|Probe Resources|Beryl Oil|BP"""
CONTRACTORS = """Transocean|Tranocean|Noble|Diamond Offshore|Valaris|ENSCO|Ensco|Rowan|Pacific Drilling|Seadrill|Helix|Nabors|Helmerich ?& ?Payne|H&P|Halliburton|Schlumberger|SLB|
Baker Hughes|Weatherford|Oceaneering|Superior Energy|Island Offshore|Edison Chouest|Chouest|Hornbeck|Tidewater|Danos|Wood Group|Grand Isle Shipyard|Shamrock|Expro|TechnipFMC|
Technip|McDermott|Saipem|Allseas|Cudd|Gulf Island|C-Innovation|Maersk|Stena|Atwood|Hercules Offshore|Spartan Offshore|Enterprise Offshore|Blake International|Aries Marine|Montco|
Wild Well Control|Boots ?& ?Coots|Archer|Frank's International|TETRA Technologies|Tetra Technologies|Workstrings|Oil States|Dril-Quip|Cameron|National Oilwell Varco|NOV|Aker|
Subsea ?7|Kongsberg|Total Safety|Production Management Industries|Furmanite|Linear Controls|Moody[- ]Price|Partco|SEACOR|Seacor|Harvey Gulf|Bollinger|Laborde|Shelf Drilling|
Quality Production Management|Petroleum Helicopters|PHI|Era Helicopters|Bristow|Fugro|Ocean Rig|Vantage Drilling|Paragon Offshore|GlobalSantaFe|Pride International|
Greene's Energy|Greene’s Energy|Wood|Offshore Energy Services|Gulf Offshore Logistics|Hilcorp|Danos|Stric-Lan|Q[- ]?Max|Turnkey Offshore|Fluor|Kiewit|Jacobs|
Delmar|Crosby|Ocean Marine|Gulf Coast Marine|Shore Offshore|Alliance Offshore|ProServ|Sparrows|Seatrax|Liebherr|Applied Machinery"""
SUFFIX = r"(?:L\.?L\.?C\.?|Inc\.?|Incorporated|Corporation|Corp\.?|Company|Co\.|Ltd\.?|Limited|LP|L\.P\.|U\.?S\.?A\.?)"
VESSEL_KW = r"(?:drillship|drill ship|rig|semi-submersible|semisubmersible|jack-up|jackup|liftboat|lift boat|M/V|MV|DSV|OSV|tug(?:boat)?|barge|vessel|FPSO)"
STOP = {"The", "A", "An", "Captain", "Operator", "Crane", "Manager", "Supervisor", "Mechanic", "Operations", "Crew", "On", "In", "At", "While", "During",
        "Was", "Were", "And", "Of", "To", "Deck", "Floor", "Personnel", "Lead", "Driller", "Superintendent", "Foreman", "Person", "Injured", "IP", "PIC",
        "OIM", "BSEE", "When", "After", "Before", "As", "Then", "It", "This", "That", "He", "She", "They", "Leader", "Hand", "Roustabout", "Rigger"}

# Word boundary that also works around "&" and "-" inside names; no "g"/"i" flags on the names themselves.
NAMES_RX = re.compile(r"(?<![\w\[])(?:%s)(?:\s+(?:Offshore|Energy|Oil(?:\s+(?:and|&)\s+Gas)?|Exploration(?:\s*(?:&|and)\s*Production)?|Petroleum|Operating|Drilling|"
                      r"Gulf of Mexico|Deepwater|Marine|Services|International|Corp(?:oration)?\.?|Inc\.?|Company|LLC|L\.L\.C\.|USA|U\.S\.A\.|Group|Americas))*(?![\w])"
                      % "|".join(x.strip() for x in (OPERATORS + "|" + CONTRACTORS).replace("\n", "").split("|") if x.strip()))
SUFFIX_RX = re.compile(r"(?:\b[A-Z][\w&'’.\-]*(?:\s+(?:&|and)?\s*[A-Z][\w&'’.\-]*){0,4}),?\s+%s(?![\w])" % SUFFIX)
VESSEL_RX = re.compile(r"\b(%s)\s+(?:\[company\]\s+)?((?:[A-Z0-9][\w'’\-]*)(?:\s+[A-Z0-9][\w'’\-]*){0,3})" % VESSEL_KW)
PERSON_RX = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Capt|Captain)\.?\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?")
GENERIC = {"USA", "Gulf", "Deepwater", "Island", "Total", "Supreme", "HES", "Company", "Operator", "Contractor", "Owner"}
FIRST = ("James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew Anthony Mark Donald Steven Paul Andrew "
         "Joshua Kenneth Kevin Brian George Timothy Ronald Jason Edward Jeffrey Ryan Jacob Gary Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon "
         "Benjamin Samuel Gregory Frank Raymond Patrick Jack Dennis Jerry Tyler Aaron Jose Adam Henry Nathan Douglas Zachary Peter Kyle Walter Jeremy Harold "
         "Keith Christian Roger Gerald Carl Terry Sean Austin Arthur Lawrence Jesse Dylan Bryan Joe Jordan Billy Bruce Albert Willie Gabriel Logan Alan Juan "
         "Wayne Roy Ralph Randy Eugene Vincent Russell Louis Philip Bobby Johnny Bradley Mike Jim Bob Tom Bill Dave Steve Chris Mary Patricia Jennifer Linda "
         "Elizabeth Barbara Susan Jessica Sarah Karen Lisa Nancy Troy Danny Ricky Glenn Dale Chad Shane Travis Brent Darrell Lloyd Leroy Curtis Clyde Earl Fred Lars Ted Ed Don Ron Ken Rick Greg Jeff Matt Tony Dan Sam Ben Pete Ray Jay Marty Kirk Todd Craig Neil Tim Jon Rob Randall Allen Marvin Howard Leonard Herbert Hefren").split()
NOT_PERSON = {"District", "Regional", "Memorial", "Island", "Pugh", "St", "Parish", "Lake", "Field", "Block", "Area", "Platform", "Dome", "Basin"}
FIRST_RX = re.compile(r"(?<![\w\[])(?:%s)\s+([A-Z][a-zA-Z'\-]+)" % "|".join(FIRST))
ALIAS_RX = re.compile(r"\[company\](?:\s*[–\-,]\s*USA)?\s*\((?:“|\")?([A-Z][A-Za-z&\-]{1,15})(?:”|\")?\)")


def _vessel(m):
    words = m.group(2).split()
    if words[0].rstrip(".,") in STOP or (len(words) == 1 and len(words[0]) <= 2):
        return m.group(0)
    return f"{m.group(1)} [vessel]"


def scrub(t, aliases=()):
    if not isinstance(t, str) or not t:
        return t
    t = PERSON_RX.sub("[name]", t)
    t = FIRST_RX.sub(lambda m: m.group(0) if m.group(1) in NOT_PERSON else "[name]", t)
    t = NAMES_RX.sub("[company]", t)
    t = SUFFIX_RX.sub("[company]", t)
    t = VESSEL_RX.sub(_vessel, t)
    for a in aliases:
        t = re.sub(r"(?<![\w\[])%s(?![\w])" % re.escape(a), "[company]", t)
    t = re.sub(r"\[company\](?:\s*[–\-,]\s*USA)?\s*\(\[company\]\)", "[company]", t)
    t = re.sub(r"(\[company\][\s,&/]*){2,}", "[company] ", t).replace("[company] .", "[company].").replace("[company] ,", "[company],")
    return t


def main():
    inc = pd.read_pickle(ART / "incidents.pkl")
    text_cols = ["what", "outcome"]
    first = inc["what"].map(lambda s: scrub(s))
    aliases = sorted({a for s in first.dropna() for a in ALIAS_RX.findall(s) if a not in STOP and a not in GENERIC and len(a) >= 3})
    print("aliases found:", aliases)
    for c in text_cols:
        inc[c] = inc[c].map(lambda s: scrub(s, aliases))
    for c in ["causes", "recs"]:
        inc[c] = inc[c].map(lambda xs: [scrub(x, aliases) for x in xs] if isinstance(xs, list) else xs)
    inc.to_pickle(ART / "incidents.pkl")
    oof = pd.read_csv(ART / "oof.csv", dtype={"y": str})
    oof["src_text_stripped"] = oof["src_text_stripped"].map(lambda s: scrub(s, aliases))
    oof.to_csv(ART / "oof.csv", index=False)
    print("rewrote incidents.pkl and oof.csv")


def audit():
    inc = pd.read_pickle(ART / "incidents.pkl")
    blob = " ".join(inc.what.dropna()) + " " + " ".join(x for xs in inc.causes for x in xs) + " " + " ".join(x for xs in inc.recs for x in xs)
    names = [x.strip() for x in (OPERATORS + "|" + CONTRACTORS).replace("\n", "").split("|") if x.strip()]
    left = {n: len(re.findall(r"(?<![\w\[])%s(?![\w])" % n, blob)) for n in names}
    print("leftover list names:", {k: v for k, v in left.items() if v})
    print("[company]", blob.count("[company]"), "[vessel]", blob.count("[vessel]"), "[name]", blob.count("[name]"))


if __name__ == "__main__":
    audit() if "--audit" in sys.argv else main()
