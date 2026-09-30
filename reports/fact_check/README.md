# Manual check of the fact-token rule

Purpose: a plausibility check of the surface rule that defines fact tokens (the basis of the Fact Recovery Rate),
not a validation. One annotator (Simon), 2026-09-30, no second annotator.

Sample: 50 fact slots drawn with the project seed from all 624 fact slots of the 150 test sentences at level 75
(`scripts/fact_check_sample.py`; the 75% span holds every fact slot of a sentence). `fact_sample.csv` holds each
token with 12 words of context on each side and the mark in `is_fact`.

Guiding question: would a fact be wrong if a repair got this word wrong (who, where, when, how much, which
institution)? Yes = (part of) a proper name (person, place, street, institution, ship, newspaper, including a title
directly before a name), a number, a date or a sum of money. No = an ordinary word that is only capitalised
(position after a quote or in a headline, generic role nouns such as "the Government"). Unsure = a real doubt.

Result: 42 yes, 6 no, 2 unsure (reported in the Limitations). The six No cases: "Authorities" (Home
Authorities), "Co." (Cox and Co.), "POWERFULLY-BUILT" and "MAN" (all-caps opening words of an article, which the
rule does not exclude because it only skips the first word), "Heard" (capitalised after a comma), "Honourable"
(before a name). Unsure: "Common" (Court of Common Pleas) and "K" (police division in "No. 332 K"). Under the
written rules, "Co.", "Honourable" and "Common" would count as yes, so up to 46 of 50; the annotator's marks are
reported unchanged, as the conservative number.
