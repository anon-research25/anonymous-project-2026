# Trained model artifacts

Graph training writes `mocheg/graph/best.pt` and `last.pt`, with analogous COSMOS paths. Checkpoints contain model, optimizer, scheduler, scaler, epoch/step, per-rank RNG state, configuration, and feature/CLIP provenance. Sidecar JSON files contain selection metrics and provenance. Resume loads trusted local training artifacts.

Router training writes `mocheg/router/best/`, containing the Hugging Face sequence classifier, tokenizer, and S/M/C mapping. Trainer epoch checkpoints support resume. Pretrained CLIP, BGE, and LLaVA assets are loaded through configured model identifiers or local snapshot directories.

See the root README for the complete training sequence and checkpoint loading options. Training artifacts and pretrained weights are generated or downloaded through those commands and are excluded from version control.
