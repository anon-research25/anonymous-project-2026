"""LEAD-3 and explicitly reference-using extractive ORACLE explanation baselines."""
import re

from rouge_score import rouge_scorer


def sentence_split(text):
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]


def lead3(evidence):
    sentences = sentence_split("\n".join(item.text for item in evidence if item.modality == "text"))
    return " ".join(sentences[:3])


def greedy_rouge_oracle(evidence, reference, max_sentences=3):
    """Greedily optimize mean ROUGE-1/2/L F1 and preserve source sentence order."""
    sentences = sentence_split("\n".join(item.text for item in evidence if item.modality == "text"))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    selected, best_score = [], -1.0
    for _ in range(min(max_sentences, len(sentences))):
        candidate, candidate_score = None, best_score
        for index in range(len(sentences)):
            if index in selected:
                continue
            indices = sorted(selected + [index])
            scores = scorer.score(reference, " ".join(sentences[i] for i in indices))
            score = sum(value.fmeasure for value in scores.values()) / len(scores)
            if score > candidate_score:
                candidate, candidate_score = index, score
        if candidate is None:
            break
        selected.append(candidate)
        best_score = candidate_score
    return " ".join(sentences[index] for index in sorted(selected))
