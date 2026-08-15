# Author release checklist

- [ ] Confirm all five authors approve the public repository, author order and `CITATION.cff` names.
- [ ] Add ORCID identifiers to `CITATION.cff` if available.
- [ ] Replace `OWNER` in the GitHub URLs in `CITATION.cff`.
- [ ] Select and approve a code/text license; replace `LICENSE_PENDING.md` with `LICENSE`.
- [ ] Confirm the actual downloader's IHME agreement permits all included figure-level/aggregate tables.
- [ ] Recheck that `data_raw/`, analytic panels, sample predictions and model binaries are absent.
- [ ] Create the GitHub repository and upload the **contents** of this directory, not the outer ZIP.
- [ ] Make an immutable GitHub release (recommended tag: `v1.0.0`).
- [ ] Archive the release in Zenodo and obtain a DOI.
- [ ] Insert the final GitHub URL and DOI into `CITATION.cff`, the manuscript, supplement and cover letter.
- [ ] Re-run `verify_release.py`; require PASS before public release.
