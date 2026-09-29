# Stack weights by state and airport (29 September 2026) — closed

Nothing is declared, selected,
exported or uploaded; no ranking row and no official score is read. Tool: `tools/screen_segment_weights.py`; report:
[`segment_weights.json`](segment_weights.json).

## Question

The stack mixes three parts on non-LIRF rows: the v9 side (v9 plus the residual stage), the tree committee and the
context networks. The shipped weights are one outer weight per state (LTFM 1.0, matched 0.6, unmatched 0.7) and one network
share (0.85). Would weights per airport, or per airport and state, do better?

## Method

Rows: the v46 development stack, 622,504 non-LIRF rows. The gates, the stretch and the LIRF layers are left out on both
sides. For each held month the two weights are fitted by least squares on the other three months, per cell, shrunk toward
the global solution, and read on the held month against the shipped weights.

## Reading

Change of the held month's RMSE against the shipped weights (positive = worse):

| weights fitted per | shrinkage | January | July | November | December | mean, equal-footing months |
|---|---:|---:|---:|---:|---:|---:|
| one cell (global) | none | +0.07% | +0.44% | +0.61% | +0.40% | +0.30% |
| state (LTFM / matched / unmatched) | none | −0.02% | +0.01% | −0.15% | +0.04% | +0.01% |
| state | 0.3 | −0.02% | +0.03% | −0.04% | −0.04% | −0.01% |
| airport | 0.3 | +0.04% | +0.07% | +0.08% | +0.09% | +0.07% |
| airport and state | 0.3 | +0.09% | +0.17% | −0.27% | +0.06% | +0.11% |

- **The shipped weights sit at the cross-validated optimum.** Refitting the three state weights moves the held month by
  ±0.04% at most, with no consistent sign.
- **Finer weights are worse.** Per airport, and per airport and state, the fitted weights lose on three or four of the four
  held months: they follow a few extreme rows of the fitting months.
- In-sample optima by month, for the record: matched rows 0.39–0.51 on the outer weight, unmatched rows 0.70–0.95, LTFM
  0.87–1.19. The spread between months is as large as the distance to the shipped values.

**Closed.** No weight scheme of this kind is a candidate. Three development months cannot support more than three weights.
