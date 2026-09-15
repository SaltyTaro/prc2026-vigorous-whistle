# Unresolved competition details

As of **13 September 2026**.
No message has been sent to the organizers.

**14 September follow-up:** The user does not know of an organizer announcement
about `AOBT_3_flt` or `LOBT_flt`; that clarification request is answered. A fresh
check of the official data, rationale, and ranking pages plus the EUROCONTROL
data mirror confirms that these fields are documented and the task is described
as post-operations analysis. No field-specific ban was found in those pages.
The earlier exclusion was a temporary project decision, not a verified organizer
restriction. The user subsequently approved a separate NM-clock experiment after
this uncertainty was explained. Its new contract uses these two fields; previous
packages retain their original declarations. Sources and hashes:
[field-use review](field_use_review_20260914.json).
The [new campaign review](../nmclock/field_use_review.json) records another
official-page retrieval at 03:41 UTC and the user's authorization.

| Point | Evidence available | What remains to establish |
| --- | --- | --- |
| 10 or 11 airports | Actual training/ranking files contain ten, matching the detailed official table; some overview prose says eleven | Operationally resolved to ten for the current data; monitor any future dataset update |
| Evaluation file version | Current ranking has 689,534 movements and template has 344,841 departures; both objects were modified 4 September | Any organizer change log explaining the earlier smaller file sizes and evaluation set |
| Closing timezone | Published deadline is `2026-10-11 23:59:59 CET` | Whether CET means fixed UTC+1 or local Central European summer time, UTC+2 on that date |
| Daily submission reset | Published limit is five/day | Reset timezone, calendar day versus rolling 24 hours, and whether rejected/failed uploads count |
| Prize amounts and payment | €5,000 combined for first three | Individual shares, payment requirements, and any tax/administrative process |
| Final ranking process | Best submitted RMSE determines team order in the published rules | Tie-break, any final unseen evaluation, reproducibility review timeline, and how a winning version is selected for reproduction |
| Prediction numeric type | Template target is `int32` and movement ID is `double`; phase-2 v1 preserved those types and all 344,841 pairs were scored | Whether non-integer floating-point predictions are accepted without coercion remains unverified; the working baseline already preserves the accepted template types |
| Source/paper deliverables | Public GPLv3 code, reproducibility, and open external data are award conditions; JOAS publication is strongly encouraged | Exact submission/review dates for code, documentation, and any paper requirement beyond the published encouragement |
| Dataset redistribution | Challenge data is supplied through approved-team credentials | A specific redistribution license for the raw competition files; code publication alone does not establish this |

For the closing-time ambiguity, literal CET would mean **22:59:59 UTC on
11 October**, or **04:29:59 IST on 12 October**. If CEST was intended, those
times would be **21:59:59 UTC** and **03:29:59 IST**. This is a one-hour
uncertainty; plan the final validated submission well before either cutoff.

One public participant README attributes an explicit warning about
`AOBT_3_flt`/`LOBT_flt` to "the brief." That quoted field-specific sentence does
not appear in the retrieved official Netlify pages or their archived Quarto
sources. Other public participants openly describe using those fields. This
establishes a disagreement between secondary sources, not either permission or
a prohibition from the organizers. Keep the distinction explicit and consult
the original organizer announcement before relying on a particular interpretation.

The relevant communication sources are the
[`#prc-data-competition` channel]
and `challenge@opensky-network.org`. The user must explicitly authorize any
outgoing administrator message; none has been requested or sent. Do not repeat
the already answered request for an announcement. The authorized NM-clock
experiment proceeds under its documented source interpretation.
