# Fairness Labeling Rules (dual-label scheme)

FairSearch-arXiv audits retrieval fairness along **two independent group
dimensions**. Both are attached to every document in
`final_50k_labeled.parquet` and are joined onto every downstream retrieval /
synthesis result by `document_id`.

## 1. `privilege_label` — institution tier (unchanged)

Derived from the QS Top-50 CS institution list (Phase 3):

- `privileged` = primary institution canonicalizes to a QS Top-50 CS university
  (`is_top50_institution == True`).
- `underrepresented` = any other resolved institution.
- `unknown` = no resolvable affiliation (~0 here; Phase 1 gates on institution
  presence).

Corpus share: privileged 17.1% / underrepresented 82.9% / unknown 0%.

## 2. `geo_group` — resource/income tier (new, data-supported)

Derived from `country_code` (OpenAlex country, curated override for QS Top-50
rows). We use a **World Bank income proxy** rather than raw region because Asia
mixes high-income (JP/KR/SG/HK/IL) with non-high-income (CN/IN) economies, so a
pure region split would conflate the two.

- `high_resource` = country is a **World Bank high-income economy** (see
  `HIGH_INCOME_ISO2` below).
- `emerging` = any other resolved country (includes CN upper-middle-income,
  IN lower-middle-income, BR, TR, etc.).
- `unknown` = no country code (`UNK` / empty).

### `HIGH_INCOME_ISO2` (ISO-3166-1 alpha-2, World Bank FY2024–25 high income)

North America: `US, CA`
Europe: `GB, IE, FR, DE, NL, BE, LU, CH, AT, IT, ES, PT, SE, NO, DK, FI, IS, CZ, SK, SI, EE, LV, LT, PL, HR, HU, RO, BG, GR, MT, CY, MC, LI, AD, SM, VA`
Europe/Eurasia reclassified high income: `RU`
Asia / Middle East: `JP, KR, SG, HK, MO, TW, IL, AE, QA, KW, BH, SA, OM, BN`
Oceania: `AU, NZ`
Latin America & Caribbean: `CL, UY, PA, TT, BS, BB, KN, AG, PR`

Everything else with a resolved country code maps to `emerging`; `UNK`/empty
maps to `unknown`.

## Rationale for the dual label

- The institution dimension is already near corpus-balanced under retrieval
  (`SPD_privileged ≈ -0.006`, `SRR_privileged ≈ 0.96`) — itself a reportable
  finding, not a bias to "fix".
- The data-supported bias signal lives in the geographic/income dimension:
  Country Gini 0.865 > Institution Gini 0.752, and retrieval shifts Europe up
  / Asia down.
- `geo_group` (income proxy) and `region` are both retained: `region` for the
  descriptive North/South picture, `geo_group` for the cleaner income-tier
  fairness audit.

## Alternative (recorded, not used)

Region-based Global-North vs Global-South (`NA + Europe + Oceania` vs rest).
Rejected as the primary geo label because it misclassifies high-income Asian
economies as "South". Kept available via the existing `region` column.

The authoritative mapping lives in `geo_labels.py` (`HIGH_INCOME_ISO2`,
`geo_group()`), imported by both the notebook Phase 3 cell and
`regenerate_geo_artifacts.py` so the two never diverge.
