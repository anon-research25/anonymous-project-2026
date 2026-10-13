from __future__ import annotations

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def explanation_metrics(hypotheses, references, rouge_stemmer=True, bleu_tokenizer="13a"):
    """ROUGE F1 means and corpus BLEU-1..4, with signatures and scale 0..100."""
    if len(hypotheses) != len(references) or not references:
        raise ValueError("Explanation evaluation needs paired, nonempty reference records")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=rouge_stemmer)
    values = [scorer.score(reference, hypothesis) for hypothesis, reference in zip(hypotheses, references)]
    metrics = {key: float(np.mean([value[key].fmeasure for value in values]) * 100)
               for key in ("rouge1", "rouge2", "rougeL")}
    signatures = {}
    for order in range(1, 5):
        bleu = BLEU(max_ngram_order=order, tokenize=bleu_tokenizer, smooth_method="exp", effective_order=False)
        metrics[f"bleu{order}"] = bleu.corpus_score(hypotheses, [references]).score
        signatures[f"bleu{order}"] = str(bleu.get_signature())
    return {**metrics, "n": len(references), "scale": "0..100",
            "rouge_variant": "mean_per_example_fmeasure", "rouge_stemmer": rouge_stemmer,
            "bleu_signatures": signatures,
            "interpretation": "Lexical/sequence similarity only; not factual faithfulness."}
