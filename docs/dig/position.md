# Dig tunnel 11: position-based intruders at EDDM / EHAM / LSZH / LEMD / LEBL — no GO, closed (3 October 2026)

Question: the intruder stand bound (`configs/intruder_bound_v1.json`) fails at EDDM, EHAM, LSZH, LEMD and LEBL because a stand code
covers several positions or aircraft are towed. Can ADS-B parking positions replace stand codes there? 2025 only; nothing declared,
built or uploaded; no official score. Scratch: `runs/dig/position/` (`build_pos.py`, `pos_lib.py`, `validity*.py`, `diag*.py`).

## Method

- **Identity** (as tunnel 10): every arrival and departure at the eight ADS-B airports gets an ICAO address from the adsb.lol day files
  (nearest point of the same callsign at the same airport within 600 s of landing / take-off; callsign = `CALLSIGN_flt`, else the
  learned flight-number key). Identified: 99% of arrivals and departures at EDDM, EHAM, LSZH, LEBL every month; LEMD 40-50% (dark in
  January 2025); LFPG 37-89%.
- **Parking position of an arrival**: median of its stationary ground points (gs <= 3 kt, 30-s bins) in [in-block - 300, in-block + 900] s
  (fallback: last slow point near in-block). Coverage: EDDM 32-90%, EHAM 49-85%, LSZH 53-73%, LEBL 42-66%, LEMD 17-50% of arrivals
  (ranking similar).
- **Rule**: our departure (address X) parked where our inbound (latest arrival of X before take-off) parked; an arrival Y != X parking
  within R m of that position after our inbound and before our predicted off-block (gap <= 3,600 s) -> label > take-off - Y's in-block.
  Validity = share of candidates with label >= B - 60 s, all twelve 2025 months, twelve-month table predictor.

## Results

| Variant | Candidates per year (5 target airports) | Validity |
|---|---:|---:|
| all fixes, R 15-40 m, first or last intruder | 6,000-50,000 | 1.5-3.3% at every airport |
| ground time <= 1.5-4 h, R 10-20 m | 400-1,500 | 0-9% |
| **control EDDF / EGLL**, same rule | 1,100-4,000 | 2-5% (the stand-code rule holds 97-100% there) |
| long stationary fixes only (>= 11 bins), ground <= 3 h | **3** | 1 of 2 / 1 of 1 |
| fixes >= 4 bins, ground <= 3-6 h, R 15-25 m | 100-230 | 0-14% |

The control decides it: at EDDF and EGLL, where the stand-code bound is valid, the position version fails just as badly, so the failure
is in the position measurement, not the airports.

- **Why it fails.** (1) The identified "inbound" is often hours or days old (median ground time 5.3 h, a quarter above 22 h): those
  aircraft are towed off long before departure. (2) Short stationary runs are mostly where ground coverage ends (apron entry stop
  bars) rather than stands: fixes with 0-6 stationary bins sit a median 30-110 m from the learned stand; with 11 or more bins 87-99%
  sit within 25 m. But only 0.2-3% of identified departures have such a long-fix inbound within 3 h, because transponders go silent
  at the stand.
- **Ranking**: no configuration is valid, so there is nothing to size.

**Closed.** A same-position intruder needs a reliable parking fix for both aircraft; adsb.lol coverage at the stands of these airports
does not provide it often enough. The stand-code rule (v58) and its identity extension (`configs/intruder_identity_v1.json`) remain the
usable forms.
