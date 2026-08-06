"""Single source of truth for the `geo_group` fairness label.

Both the notebook Phase 3 cell and `regenerate_geo_artifacts.py` import this so
the mapping never diverges. See `data/fairness_labeling_rules.md` for the write-up.

`geo_group` is a World Bank income proxy derived from `country_code`:
  - high_resource : World Bank high-income economy
  - emerging      : any other resolved country
  - unknown       : no country code
"""

# ISO-3166-1 alpha-2 codes for World Bank high-income economies (FY2024-25),
# restricted to those that actually appear (or plausibly appear) in the corpus.
HIGH_INCOME_ISO2 = {
    # North America
    "US", "CA",
    # Europe
    "GB", "IE", "FR", "DE", "NL", "BE", "LU", "CH", "AT", "IT", "ES", "PT",
    "SE", "NO", "DK", "FI", "IS", "CZ", "SK", "SI", "EE", "LV", "LT", "PL",
    "HR", "HU", "RO", "BG", "GR", "MT", "CY", "MC", "LI", "AD", "SM", "VA",
    # Eurasia reclassified high income
    "RU",
    # Asia / Middle East
    "JP", "KR", "SG", "HK", "MO", "TW", "IL", "AE", "QA", "KW", "BH", "SA",
    "OM", "BN",
    # Oceania
    "AU", "NZ",
    # Latin America & Caribbean
    "CL", "UY", "PA", "TT", "BS", "BB", "KN", "AG", "PR",
}

_UNKNOWN_TOKENS = {"", "UNK", "UNKNOWN", "NAN", "NONE", "NULL"}


def geo_group(country_code) -> str:
    """Map a country code to {high_resource, emerging, unknown}."""
    cc = str(country_code).strip().upper() if country_code is not None else ""
    if cc in _UNKNOWN_TOKENS:
        return "unknown"
    return "high_resource" if cc in HIGH_INCOME_ISO2 else "emerging"


GEO_GROUP_ORDER = ["high_resource", "emerging", "unknown"]
