# Manual check of the fact-token rule

Purpose: a plausibility check of the surface rule that defines fact tokens (the basis of the Fact Recovery Rate),
not a validation. One annotator from the team, 2026-09-30, no second annotator.

Sample: 50 fact slots drawn with the project seed from all 624 fact slots of the 150 test sentences at level 75
(`scripts/fact_check_sample.py`; the 75% span holds every fact slot of a sentence). `fact_sample.csv` holds each
token with 12 words of context on each side and the mark in `is_fact`.

Guiding question: would a fact be wrong if a repair got this word wrong (who, where, when, how much, which
institution)? Yes = (part of) a proper name (person, place, street, institution, ship, newspaper, including a title
directly before a name), a number, a date or a sum of money. No = an ordinary word that is only capitalised
(position after a quote or in a headline, generic role nouns such as "the Government"). Unsure = a real doubt.

Result, first pass: 42 yes, 6 no, 2 unsure. Reviewed against the written rules above (same day, same annotator),
four marks were corrected because they contradicted those rules: "Co." (Cox and Co., part of a firm's name),
"Honourable" (a title directly before a name), "Common" (Court of Common Pleas, a named court) and "K" (part of the
constable number "No. 332 K"). **Reported result: 46 of 50 are true facts, 4 are not** (Limitations). Both columns
are kept in `fact_sample.csv` (`is_fact_first_pass`, `is_fact`). The four No cases: "Authorities" (Home
Authorities, a generic noun), "POWERFULLY-BUILT" and "MAN" (all-caps opening words of an article, which the rule
counts because it only skips the first word), "Heard" (capitalised after a comma).

What the number means: it estimates the precision of the fact rule (how many counted "facts" really are facts),
about 92%, roughly 81 to 97% with 50 tokens (Wilson interval). It does not measure recall (facts the rule misses,
such as lower-cased or written-out numbers). All methods are scored on the same tokens, so the non-facts do not
change comparisons between methods; they make the absolute Fact Recovery Rate a slight mix of facts and ordinary
words.
