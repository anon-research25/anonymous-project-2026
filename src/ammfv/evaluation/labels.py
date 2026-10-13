from ..schema import VERDICTS


class TaskLabelMapping:
    """A canonical NEI state is never silently forced into a binary benchmark class."""

    def __init__(self, config):
        self.mode = config["label_space"]
        self.mapping = config.get("output_mapping", {})
        self.classes = config.get("classes") or list(VERDICTS)
        self.nei_policy = config.get("nei_policy", "error")
        if self.mode not in {"canonical", "benchmark"}:
            raise ValueError("label_space must be canonical or benchmark")
        if self.mode == "benchmark" and (not self.mapping or not config.get("classes")):
            raise ValueError("Benchmark evaluation needs explicit classes and output_mapping")

    def gold(self, claim):
        value = claim.label if self.mode == "canonical" else claim.benchmark_label
        if value not in self.classes:
            raise ValueError(f"Gold label {value!r} outside configured evaluation classes")
        return value

    def prediction(self, verdict):
        if verdict is None:
            return "__FAILED__"
        if verdict not in VERDICTS:
            raise ValueError(f"Unknown prediction {verdict}")
        if self.mode == "canonical":
            return verdict
        mapped = self.mapping.get(verdict)
        if mapped in self.classes:
            return mapped
        if verdict == "NOT_ENOUGH_INFO" and self.nei_policy == "abstain_as_error":
            return "__ABSTAIN__"
        raise ValueError(f"No explicit benchmark mapping for {verdict}; set an NEI policy")
