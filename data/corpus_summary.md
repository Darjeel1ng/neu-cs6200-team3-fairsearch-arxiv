# Corpus Summary (Phase 4)

- Documents: 50000
- Distinct institutions: 5937
- Distinct countries: 137
- Year range: 2024-2026

## Fairness group priors (privilege_label)
- privileged: 0.1711 (8553 papers)
- underrepresented: 0.8289 (41447 papers)
- unknown: 0.0000 (0 papers)
- privileged : underrepresented ratio: 0.206

## Region distribution
- Europe: 0.4306
- North America: 0.2701
- Asia: 0.2527
- Oceania: 0.0231
- South America: 0.0149
- Africa: 0.0052
- Unknown: 0.0034

## Concentration / inequality
- Top-10 institution share: 0.0643
- Top-50 institution share: 0.2078
- Institution Gini: 0.7525
- Country Gini: 0.8654

## Top primary categories
- cs.NA: 8226
- cs.LG: 6952
- cs.CV: 5662
- cs.SY: 3868
- cs.RO: 2822
- cs.HC: 2423
- cs.AI: 2373
- cs.CL: 1996
- cs.CR: 1776
- cs.IT: 1390

## Top countries
- US: 11852
- CN: 5672
- DE: 4215
- FR: 3631
- GB: 2579
- IT: 2139
- CA: 1519
- JP: 1479
- ES: 1216
- IN: 1074

## Top institutions
- Centre National de la Recherche Scientifique: 660
- Technical University of Munich: 353
- Tsinghua University: 316
- ETH Zurich: 312
- Carnegie Mellon University: 300
- University of Illinois Urbana-Champaign: 265
- KTH Royal Institute of Technology: 255
- Delft University of Technology: 255
- Georgia Institute of Technology: 250
- Massachusetts Institute of Technology: 249

## Figures
- figures\privilege_label.png
- figures\region.png
- figures\primary_category.png
- figures\year.png
- figures\privilege_by_region.png
- figures\institution_lorenz.png

## Note
Retrieval fairness (TF-IDF baseline + curated CS topic queries, compared against the Phase 5 dense retriever) is deferred to after index construction. The priors above are the reference distribution for that analysis.