# LIRF label-mechanism screen (29 September 2026) — closed

Nothing is declared, selected,
exported or uploaded; no ranking row and no official score is read. Tool: `tools/screen_lirf_mechanism.py`; report:
[`lirf_mechanism.json`](lirf_mechanism.json). Rows: the v46 development stack
(`runs/clean_line_screens/hour/dev_stack.parquet`), LIRF departures with an NM off-block clock, outside the fill population,
labels below 10,800 s: 49,385 rows, about a fifth of the ordinary development squared error.

## The classes

By the airport block time, which is known on labelled months only and never enters an input:

| class | definition | rows | stack RMSE (s) | stack bias (s) | share of these rows' squared error |
|---|---|---:|---:|---:|---:|
| agree | block within 300 s of the NM clock | 56.8% | 199 | −76 | 22.1% |
| early | block more than 300 s before the NM clock (median 538 s) | 21.3% | 412 | +165 | 35.5% |
| fill | block within 5 s of the schedule time | 16.5% | 483 | +151 | 37.8% |
| late | block more than 300 s after the NM clock | 5.4% | 293 | −207 | 4.6% |

The NM proxy has the same median (about 900 s) in every class, and 91% of the proxies are whole minutes. The class
shares are stable over the months, except July (early 31%, fill 13%).

If the class of every row were known, the error on these rows would fall by roughly 60%. The question is whether any
label-free input tells the classes apart better than the stack already does. Fill separation is on record as closed
(`docs/error_map/README.md`); this screen reads **early against agree**.

## Reading

| score for "early" | January | July | November | December | all |
|---|---:|---:|---:|---:|---:|
| the stack's own: prediction minus NM proxy | 0.872 | 0.889 | 0.873 | 0.887 | 0.885 |
| classifier, label-free inputs only | 0.837 | 0.867 | 0.859 | 0.871 | 0.864 |
| classifier, the stack's outputs only | 0.871 | 0.890 | 0.873 | 0.890 | 0.887 |
| classifier, label-free inputs and the stack's outputs | 0.870 | 0.892 | 0.878 | 0.893 | 0.888 |

(AUC; month-blocked LightGBM, fixed settings; inputs: operator, stand, aircraft type, destination, market segment, runway,
wake category, schedule and clock intervals, the seconds of each clock, and the ADS-B match columns.)

- **A dedicated classifier adds 0.003 of AUC to the stack's own separation.** On label-free inputs alone it is worse than
  the stack.
- **The stack is calibrated along the classifier's probability.** Its bias is between −32 s and +12 s in every band, and its
  mean lift above the proxy follows the labels' (from −60 s against −48 s in the lowest band to +807 s against +792 s in
  the highest).
- The class biases in the table above are therefore the cost of an honest hedge, not a calibration error: the stack
  cannot know the class, so it predicts between them.

**Closed.** At LIRF the stack already holds what the inputs say about the label mechanism. The remaining error there is the
class itself, which is decided by something the data does not record.

## Follow-up: arrival-side reporting signals (29 September, after the arrival second task) — closed

Tool: `tools/screen_lirf_arrival_side.py`. Same rows and classes. Added label-free inputs from arrival rows, whose block times
are supplied in every month: the inbound leg (the last arrival at the same stand before the departure's NM clock, found for
95% of the rows) with its fill flag, block second, taxi-in and gaps; and the arrival fill share of the same operator that
day, of the airport in the hour and on the day (24 inputs).

| reading (AUC, month-blocked, stack outputs included) | without | with arrival-side inputs | arrival-side inputs alone |
|---|---:|---:|---:|
| early against agree | 0.888 | 0.888 | 0.722 |
| fill against the rest | 0.865 | 0.869 | 0.750 |

- The inbound leg is a fill for 17% of the fill rows against 7% of the others, and the operator's arrival fill share that day
  is 16% against 7%. The signal is real and the stack's inputs already carry nearly all of it.
- A month-blocked correction of label minus v46 on these rows, at half weight: +0.40% without and +0.04% with the
  arrival-side inputs; by month with them January +0.35%, July +0.54%, November −0.44%, December −1.09%. Two of the three
  equal-footing months worsen.

**Closed.** Arrival-side reporting behaviour does not separate the label classes of LIRF clock rows beyond what the stack
holds. The population classifier, closed by the user after v43, is not touched by this screen.
