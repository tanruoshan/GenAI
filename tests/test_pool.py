import pysbd

from blnrepair.data import band_of, split_sentences
from blnrepair.facts import fact_indices

SEG = pysbd.Segmenter(language="en", clean=False)
BANDS = [[20, 29], [30, 44], [45, 60]]


def test_title_with_attached_dash_is_not_a_sentence_end():
    text = "The prisoner was fined. —Mr. Smith appeared for the defence."
    assert split_sentences(text, SEG) == ["The prisoner was fined.", "Mr. Smith appeared for the defence."]


def test_amount_followed_by_lowercase_is_not_a_sentence_end():
    text = "He was fined 6d. and costs were paid by the prisoner."
    assert split_sentences(text, SEG) == [text]


def test_esq_can_still_end_a_sentence():
    text = "The wife of R. W. Weston, Esq. The child was named John."
    assert len(split_sentences(text, SEG)) == 2


def test_facts_numbers_money_and_names():
    tokens = "On March 23 he paid £5 to Mr. Jones, of the Old Bailey.".split()
    assert [tokens[i] for i in fact_indices(tokens)] == ["March", "23", "£5", "Jones,", "Old", "Bailey."]


def test_first_word_is_not_a_fact_but_amount_is():
    tokens = "Robert paid 5s. for it".split()
    assert fact_indices(tokens) == [2]


def test_quote_before_capital_counts():
    assert fact_indices('he said “Hello” loudly'.split()) == [2]


def test_band_edges():
    assert [band_of(n, BANDS) for n in (19, 20, 29, 30, 44, 45, 60, 61)] == [
        None, "20-29", "20-29", "30-44", "30-44", "45-60", "45-60", None]


def test_amount_split_across_digits_is_rejoined():
    text = "The officer saw money passed, and when at the station £2 5s. 10 ½d. was found upon them A warder proved convictions."
    assert split_sentences(text, SEG) == [text]


def test_amount_at_sentence_end_still_ends_it():
    text = "He was fined 5s. The prisoner paid at once."
    assert split_sentences(text, SEG) == ["He was fined 5s.", "The prisoner paid at once."]


def test_leading_dash_token_is_removed():
    assert split_sentences("– Mr. Litherea, the well-known rider, was summoned.", SEG) == [
        "Mr. Litherea, the well-known rider, was summoned."]
    assert split_sentences("--- SEVERE SENTENCES.", SEG) == ["SEVERE SENTENCES."]


def test_dash_attached_to_the_first_word_is_removed():
    assert split_sentences("—Jules de Bodt was indicted for keeping a house.", SEG) == [
        "Jules de Bodt was indicted for keeping a house."]
    assert split_sentences("–Mr. Litherea, the well-known rider, was summoned.", SEG) == [
        "Mr. Litherea, the well-known rider, was summoned."]
    assert split_sentences("---Edwin Boosey was charged.", SEG) == ["Edwin Boosey was charged."]


def test_dash_is_only_removed_at_the_start_of_a_sentence():
    text = "He said—Jules was there – and Mr. Smith too, a well-known man, on 5-10 days."
    assert split_sentences(text, SEG) == [text]


def test_second_sentence_also_loses_its_dash():
    assert split_sentences("He was fined 5s. —Mr. Jones appeared for him.", SEG) == [
        "He was fined 5s.", "Mr. Jones appeared for him."]


def test_a_lone_dash_is_not_a_sentence():
    assert split_sentences("He left. —", SEG) == ["He left."]


def test_i_is_not_a_fact():
    tokens = 'and saying, "I would treat them well'.split()
    assert [tokens[i] for i in fact_indices(tokens)] == []


def test_titles_are_not_facts_but_names_are():
    tokens = "He asked Mrs. Davis and Dr. Hall and ST. Mary and Mr. Cunliffe".split()
    assert [tokens[i] for i in fact_indices(tokens)] == ["Davis", "Hall", "Mary", "Cunliffe"]


def test_word_after_a_colon_is_not_a_fact():
    tokens = "Mr. Phillips, jeweller, Whitechapel: On the day he came".split()
    assert [tokens[i] for i in fact_indices(tokens)] == ["Phillips,", "Whitechapel:"]


def test_numbers_and_money_are_still_facts_after_a_colon_or_title():
    tokens = "He said: £100 was paid, 5s. more, and 37 came to Cunliffe".split()
    assert [tokens[i] for i in fact_indices(tokens)] == ["£100", "5s.", "37", "Cunliffe"]


def test_the_miss_sir_rev_messrs_are_not_facts():
    tokens = "He met Miss Ada, Sir John, the Rev. Smith, THE Messrs. Brown and The Lord".split()
    assert [tokens[i] for i in fact_indices(tokens)] == ["Ada,", "John,", "Smith,", "Brown", "Lord"]


def test_days_lord_court_and_police_court_are_still_facts():
    tokens = "They came on Monday to the Police-court before Lord Justice at Court".split()
    assert [tokens[i] for i in fact_indices(tokens)] == ["Monday", "Police-court", "Lord", "Justice", "Court"]


def test_i_is_compared_exactly():
    assert fact_indices('he said I would and "i" did'.split()) == []
