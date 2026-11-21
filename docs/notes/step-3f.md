# Step 3f — real data (NHTSA recalls, AI4I 2020): preparation

Status (2026-11-21): **prepared, nothing downloaded.** Parser, sheet builder, label
checker and the NHTSA section of `eval_sets/diagnosis/LABELLING_GUIDE.md` are done and
tested on synthetic files. The downloads wait for approval.

## Sources checked (no data downloaded)

Sizes come from the NHTSA S3 bucket listing (exact bytes) and from the UCI dataset page
(rounded; UCI sends no `Content-Length`). Licences come from the data.gov catalogue
entry and the UCI page. The nhtsa.gov datasets page refused automated access (HTTP
403), so it was not read.

| File | URL | Size | Last modified | Licence |
| --- | --- | --- | --- | --- |
| Recalls since 2010 (**needed**) | `https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip` | 15,075,999 B (15.1 MB zip) | 2026-11-20 (updated often: the SHA-256 is recorded at download) | U.S. government work; data.gov "accessLevel: public", no licence field → treated as public domain (17 U.S.C. § 105, an interpretation) |
| Recalls before 2010 (not needed) | `.../rcl/FLAT_RCL_PRE_2010.zip` | 7,447,141 B | 2026-11-20 | as above |
| Recalls field description | `.../rcl/RCL.txt` | 3,053 B | 2026-11-20 | read for the parser, not saved |
| Complaints, all years (not needed now) | `.../cmpl/FLAT_CMPL.zip` | 377,184,005 B (377 MB zip) | 2026-11-20 | as above |
| Complaints 2025–2026 only (optional) | `.../cmpl/COMPLAINTS_RECEIVED_2025-2026.zip` | 44,679,686 B | 2026-11-20 | as above |
| AI4I 2020 (**needed**) | `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip` | about 510 KB (page: "509.9 KB"; `ai4i2020.csv` 509.8 KB; 10,000 rows) | page: last updated 2024-02-14 | **CC BY 4.0**. Cite the dataset DOI 10.24432/C5HS5C and S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", AI4I 2020, DOI 10.1109/AI4I49448.2020.00023 |

- The old data.gov link `.../FLAT_RCL.zip` returns 404. NHTSA now splits the recalls
  file into PRE_2010 and POST_2010 parts.
- The unzipped recalls size is not known yet; it will be measured after unzipping.

## What was built

- `iqs_diagnosis/ingestion/nhtsa.py`
  - Reads the 29 tab-separated fields of `RCL.txt` and merges rows into one record per
    campaign.
  - Short (older) rows are padded. Rows with more than 29 fields are counted and
    skipped, and undecodable bytes are counted.
- `iqs_diagnosis/evaluation/nhtsa_labels.py`
  - Draws a stratified, seeded sample per component group: a "term" stratum matching a
    manufacturing word list, plus a random "control" stratum.
  - Writes the CSV with shuffled rows and the stratum hidden. The manifest records the
    source file's SHA-256 and the strata. An existing sheet is never overwritten.
  - The checker enforces the guide's rules:
    - valid code;
    - evidence must be a verbatim quote;
    - confidence must be high or medium;
    - status and cause columns must be consistent;
    - text, ids and the header must be unchanged.
- `scripts/prepare_nhtsa_labelling.py` (`--list-groups`, then `--groups ...`) and
  `scripts/check_nhtsa_labels.py` (progress, yield by stratum, errors).
- `packages/diagnosis/tests/test_nhtsa_labelling.py` (synthetic flat file).

## Needs your approval / decision

1. **Download** `FLAT_RCL_POST_2010.zip` (15.1 MB) to `data/raw/nhtsa/` and unzip it.
2. **Download** AI4I 2020 (about 0.5 MB) to `data/raw/ai4i/`.
3. **Complaints: recommend not now.** They carry no cause, so they are not needed for
   labelling or for Step 4 gate 1. If you want them later, use the 2025–2026 file
   (44.7 MB) rather than the full 377 MB.
4. **Component groups (3–5).** Run `--list-groups` after the download and pick from the
   real names and counts. My prior is the mechanical, production-heavy groups: service
   brakes, steering, suspension, power train and structure.
5. **Sheet size.** The default of 60 term + 15 control per group gives 375 rows for 5
   groups. At roughly 1–1.5 min per row that is 6–9 h, well above the 2–3 h estimated
   earlier.
   - I recommend 30 term + 8 control per group (about 190 rows, 3–5 h).
   - Label the 50-row pilot first, then decide.
6. **Committing the sheet.** The CSV holds NHTSA recall text, which is public domain.
   The manifest holds a SHA-256, which the secret scan will flag. This needs an exclude
   for `eval_sets/diagnosis/nhtsa/[a-z0-9_]+\.manifest\.json`, the same kind of exclude
   as the approved `sim/` one. Recall text containing part numbers may also trigger the
   scan; I'll check that at the first commit and ask.
7. **Mapping interpretations to review in the guide:**
   - a supplier lot with no stated mechanism → `MATERIAL.BAD_BATCH` (medium);
   - "improperly torqued/welded" → `METHOD.WRONG_PROCESS_PARAMETERS` (medium);
   - plant inspection equipment out of calibration → `MEASUREMENT.SENSOR_DRIFT`
     (medium).

## Risks to know before labelling

- **Few causes and small counts.** Recall texts can support only about 5 of the 10
  causes, and most rows will be `not_determinable`. Step 4 gate 1 (coverage on a real
  test split) may therefore end up with a small test split and wide Clopper–Pearson
  intervals. The pilot will show the yield.
- **The agent cannot read text cases yet.** Today's agent works on simulator symptom
  profiles. To run labelled NHTSA cases it needs a text path: symptom extraction from
  text through the LLM gateway (3e, Ollama only, not approved yet), or at least a
  text-only BM25 baseline. Labelling can go ahead in parallel.
