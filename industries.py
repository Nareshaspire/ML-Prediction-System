"""Reference table of every industry the scanner covers.

An "industry" here is an HS 2-digit chapter (96 of them; chapter 77 is reserved and unused).
Chapters are grouped into the 21 HS sections ("sectors"), tagged with themes such as
critical minerals, and mapped to a listed-equity proxy (a sector ETF) for market signals.

Caveats (read before relying on the critical-mineral flag)
  * HS-2 is deliberately coarse: it lets one model scan the whole economy. Individual critical
    minerals (lithium, cobalt, graphite, rare earths...) are defined at HS-6 inside the chapters,
    so the flag is a SCREEN to find chapters worth drilling into, not a mineral-level answer.
  * A flagged chapter also holds non-critical goods (chapter 26 includes iron ore, chapter 28
    includes bulk industrial chemicals, chapter 76 is mostly ordinary aluminium).
  * Not flagged although they contain some critical material: chapter 71 (platinum-group metals,
    dominated by gold, diamonds and jewellery) and chapter 72 (ferroalloys, dominated by steel).
"""
from __future__ import annotations

import pandas as pd

CHAPTER_NAMES = {
    1: "Live animals", 2: "Meat", 3: "Fish and seafood", 4: "Dairy, eggs, honey",
    5: "Other animal products", 6: "Live plants, cut flowers", 7: "Edible vegetables",
    8: "Edible fruit and nuts", 9: "Coffee, tea, spices", 10: "Cereals",
    11: "Milling products, malt", 12: "Oil seeds and oleaginous fruits",
    13: "Lac, gums, resins", 14: "Vegetable plaiting materials", 15: "Animal and vegetable fats, oils",
    16: "Meat and fish preparations", 17: "Sugars", 18: "Cocoa", 19: "Cereal and flour preparations",
    20: "Vegetable and fruit preparations", 21: "Miscellaneous edible preparations",
    22: "Beverages, spirits, vinegar", 23: "Food industry residues, animal feed", 24: "Tobacco",
    25: "Salt, sulphur, stone, cement", 26: "Ores, slag and ash", 27: "Mineral fuels, oils",
    28: "Inorganic chemicals", 29: "Organic chemicals", 30: "Pharmaceutical products",
    31: "Fertilizers", 32: "Tanning, dyeing extracts, paints", 33: "Essential oils, cosmetics",
    34: "Soap, waxes, candles", 35: "Albuminoidal substances, glues", 36: "Explosives, pyrotechnics",
    37: "Photographic goods", 38: "Miscellaneous chemical products", 39: "Plastics",
    40: "Rubber", 41: "Raw hides and skins", 42: "Leather articles", 43: "Furskins",
    44: "Wood and wood articles", 45: "Cork", 46: "Straw and basketware", 47: "Pulp of wood",
    48: "Paper and paperboard", 49: "Printed books, newspapers", 50: "Silk", 51: "Wool",
    52: "Cotton", 53: "Other vegetable textile fibres", 54: "Man-made filaments",
    55: "Man-made staple fibres", 56: "Wadding, felt, nonwovens", 57: "Carpets",
    58: "Special woven fabrics", 59: "Coated textile fabrics", 60: "Knitted fabrics",
    61: "Apparel, knitted", 62: "Apparel, not knitted", 63: "Other made-up textile articles",
    64: "Footwear", 65: "Headgear", 66: "Umbrellas, walking sticks", 67: "Feathers, artificial flowers",
    68: "Stone, plaster, cement articles", 69: "Ceramic products", 70: "Glass and glassware",
    71: "Pearls, precious stones and metals", 72: "Iron and steel", 73: "Articles of iron or steel",
    74: "Copper", 75: "Nickel", 76: "Aluminium", 78: "Lead", 79: "Zinc", 80: "Tin",
    81: "Other base metals (cobalt, tungsten, titanium...)", 82: "Tools, cutlery",
    83: "Miscellaneous base metal articles", 84: "Machinery and mechanical appliances",
    85: "Electrical machinery and electronics", 86: "Railway equipment", 87: "Vehicles",
    88: "Aircraft and spacecraft", 89: "Ships and boats", 90: "Optical, medical instruments",
    91: "Clocks and watches", 92: "Musical instruments", 93: "Arms and ammunition",
    94: "Furniture, bedding, lighting", 95: "Toys, games, sports equipment",
    96: "Miscellaneous manufactured articles", 97: "Works of art, antiques",
}

# (first chapter, last chapter, section name)
SECTIONS = [
    (1, 5, "Animal products"), (6, 14, "Vegetable products"), (15, 15, "Fats and oils"),
    (16, 24, "Prepared foodstuffs"), (25, 27, "Mineral products"), (28, 38, "Chemicals"),
    (39, 40, "Plastics and rubber"), (41, 43, "Hides and leather"), (44, 46, "Wood products"),
    (47, 49, "Pulp and paper"), (50, 63, "Textiles"), (64, 67, "Footwear and headgear"),
    (68, 70, "Stone, ceramics, glass"), (71, 71, "Precious stones and metals"),
    (72, 83, "Base metals"), (84, 85, "Machinery and electrical"), (86, 89, "Transport equipment"),
    (90, 92, "Instruments"), (93, 93, "Arms"), (94, 96, "Miscellaneous manufactures"),
    (97, 97, "Art and antiques"),
]

# Chapters that contain critical-mineral value chains: ores and concentrates, graphite and
# fluorspar, rare-earth / lithium compounds, refined and semi-processed metals.
CRITICAL_MINERAL_CHAPTERS = {
    "25": "graphite, fluorspar, quartz/silica, magnesium compounds",
    "26": "ores and concentrates: cobalt, manganese, chromium, tungsten, niobium, tantalum, titanium, zirconium",
    "28": "lithium, rare-earth and cobalt compounds; high-purity silicon",
    "74": "copper",
    "75": "nickel",
    "76": "aluminium",
    "79": "zinc",
    "80": "tin",
    "81": "cobalt, tungsten, titanium, magnesium, antimony, germanium, gallium, vanadium, niobium",
}

# Listed-equity proxy per chapter range: US sector ETFs (global, liquid, free via yfinance).
EQUITY_PROXIES = [
    (1, 24, "XLP"), (25, 26, "XME"), (27, 27, "XLE"), (28, 29, "XLB"), (30, 30, "XLV"),
    (31, 38, "XLB"), (39, 40, "XLB"), (41, 43, "XLY"), (44, 49, "XLB"), (50, 67, "XLY"),
    (68, 70, "XLB"), (71, 71, "GDX"), (72, 81, "XME"), (82, 84, "XLI"), (85, 85, "XLK"),
    (86, 86, "XLI"), (87, 87, "XLY"), (88, 89, "XLI"), (90, 90, "XLV"), (91, 92, "XLY"),
    (93, 93, "ITA"), (94, 97, "XLY"),
]

# Market-wide series used as macro features: ticker -> column name.
MACRO_TICKERS = {
    "CADINR=X": "cad_inr",   # exchange rate
    "CL=F": "crude_oil",
    "HG=F": "copper",
    "GC=F": "gold",
    "^GSPTSE": "tsx",        # Canadian equities
    "^NSEI": "nifty50",      # Indian equities
}


HS2_NAMES = {f"{c:02d}": name for c, name in CHAPTER_NAMES.items()}


def _lookup(ranges, chapter):
    for lo, hi, value in ranges:
        if lo <= chapter <= hi:
            return value
    return None


def sector_of(code) -> str:
    """HS section name for a chapter code such as '27' or 27."""
    return _lookup(SECTIONS, int(code))


def reference_table() -> pd.DataFrame:
    """One row per industry (HS-2 chapter): name, sector, themes and equity proxy."""
    rows = []
    for code, name in HS2_NAMES.items():
        chapter = int(code)
        rows.append(
            {
                "hs2": code,
                "industry": name,
                "sector": sector_of(chapter),
                "critical_mineral": code in CRITICAL_MINERAL_CHAPTERS,
                "energy": chapter == 27,
                "agri_food": chapter <= 24,
                "equity_proxy": _lookup(EQUITY_PROXIES, chapter),
            }
        )
    return pd.DataFrame(rows)


chapter_table = reference_table  # older name


def all_hs2() -> list:
    return list(HS2_NAMES)
