"""
WER (Word Error Rate) evaluation harness.

WER is the primary metric for ASR quality. Definition:
    WER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions, N = words in reference

Perfect WER = 0%. Industry baselines:
    Whisper large-v3 on LibriSpeech clean:  ~2% WER
    Whisper base on LibriSpeech clean:      ~5-6% WER
    Whisper base on real conversational:    ~15-25% WER (depends heavily on domain)
    Human transcription:                    ~5% WER (due to disfluencies)

WER alone is misleading for domain-specific use cases. Supplement with:
- CER (Character Error Rate): more granular, better for short texts
- MER (Match Error Rate): penalizes false insertions more heavily
- WIL (Word Information Lost): information-theoretic variant

We implement WER via dynamic programming (edit distance), same algorithm
used by jiwer and most ASR evaluation frameworks.
"""

from __future__ import annotations

import re
import json
import string
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Normalize for fair WER comparison.
    Both hypothesis and reference must be normalized identically.

    Standard normalization:
    - Lowercase
    - Remove punctuation (ASR models don't always output it consistently)
    - Collapse multiple spaces
    - Strip leading/trailing whitespace

    Do NOT remove numbers — "3 units" vs "three units" is a real error.
    """
    text = text.lower()
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[int, int, int]:
    """
    Dynamic programming edit distance between two token sequences.
    Returns (substitutions, deletions, insertions).

    Standard Levenshtein over words, not characters.
    Time: O(|ref| * |hyp|), Space: O(|ref|) with rolling array.
    """
    r, h = len(ref_tokens), len(hyp_tokens)

    # dp[i][j] = min edits to align ref[:i] with hyp[:j]
    # Using 2-row rolling to save memory
    prev = list(range(h + 1))
    curr = [0] * (h + 1)

    for i in range(1, r + 1):
        curr[0] = i
        for j in range(1, h + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1]  # match, no cost
            else:
                curr[j] = 1 + min(
                    prev[j],        # deletion from ref
                    curr[j - 1],    # insertion into ref
                    prev[j - 1],    # substitution
                )
        prev, curr = curr, [0] * (h + 1)

    # Total edits = prev[h] (last row, last column)
    total_edits = prev[h]

    # Backtrack to count S/D/I separately (standard for detailed analysis)
    # Full matrix backtrack — rebuild for small sequences only
    if r * h <= 10000:
        s, d, ins = _backtrack_sdi(ref_tokens, hyp_tokens)
    else:
        # Approximate: assume uniform edit types
        s = min(r, h, total_edits)
        d = max(0, r - h - s // 2)
        ins = max(0, total_edits - s - d)

    return s, d, ins


def _backtrack_sdi(ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack
    i, j = r, h
    s = d = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1; i -= 1; j -= 1  # substitution
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1; i -= 1          # deletion
        else:
            ins += 1; j -= 1        # insertion

    return s, d, ins


@dataclass
class WERScore:
    reference: str
    hypothesis: str
    substitutions: int
    deletions: int
    insertions: int
    ref_word_count: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.ref_word_count == 0:
            return 0.0
        return self.total_errors / self.ref_word_count

    @property
    def cer(self) -> float:
        """Character error rate."""
        ref_chars = list(self.reference.replace(" ", ""))
        hyp_chars = list(self.hypothesis.replace(" ", ""))
        if not ref_chars:
            return 0.0
        s, d, ins = _backtrack_sdi(ref_chars, hyp_chars)
        return (s + d + ins) / len(ref_chars)


@dataclass
class WEREvalReport:
    total_words: int = 0
    total_substitutions: int = 0
    total_deletions: int = 0
    total_insertions: int = 0
    per_file: List[dict] = field(default_factory=list)

    @property
    def overall_wer(self) -> float:
        if self.total_words == 0:
            return 0.0
        return (self.total_substitutions + self.total_deletions + self.total_insertions) / self.total_words

    @property
    def overall_cer(self) -> float:
        """Approximate CER from WER components — compute properly in full eval."""
        return self.overall_wer * 0.7  # heuristic; replace with per-char computation

    def print_summary(self, title: str = "WER Evaluation"):
        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")
        print(f"  Total reference words:  {self.total_words}")
        print(f"  Substitutions:          {self.total_substitutions}")
        print(f"  Deletions:              {self.total_deletions}")
        print(f"  Insertions:             {self.total_insertions}")
        print(f"  Total errors:           {self.total_substitutions + self.total_deletions + self.total_insertions}")
        print(f"  WER:                    {self.overall_wer:.1%}")
        print(f"  Relative WER:           {self._relative_label()}")

    def _relative_label(self) -> str:
        wer = self.overall_wer
        if wer < 0.05: return "Excellent (< 5%)"
        if wer < 0.10: return "Good (< 10%)"
        if wer < 0.20: return "Acceptable (< 20%)"
        return f"Needs improvement ({wer:.0%})"

    def to_json(self, path: Optional[str] = None) -> str:
        data = {
            "overall_wer": self.overall_wer,
            "total_words": self.total_words,
            "substitutions": self.total_substitutions,
            "deletions": self.total_deletions,
            "insertions": self.total_insertions,
            "per_file": self.per_file,
        }
        js = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(js)
        return js


def compute_wer(reference: str, hypothesis: str) -> WERScore:
    """Single-pair WER computation. Both strings pre-normalized."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    s, d, ins = edit_distance(ref_tokens, hyp_tokens)
    return WERScore(
        reference=reference,
        hypothesis=hypothesis,
        substitutions=s,
        deletions=d,
        insertions=ins,
        ref_word_count=len(ref_tokens),
    )


class WEREvaluator:
    """
    Evaluate WER across a dataset of (audio_file, reference_transcript) pairs.

    Input format: directory with:
        audio.wav        → audio
        audio.txt        → reference transcript (plain text, one sentence per file)

    Or a JSON file:
        [{"audio": "path/to/file.wav", "text": "reference transcript"}, ...]
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def evaluate_from_json(
        self,
        manifest_path: str,
        transcripts: Dict[str, str],    # {audio_path: hypothesis_text}
    ) -> WEREvalReport:
        """
        manifest_path: JSON file with [{"audio": ..., "text": ...}]
        transcripts: dict mapping audio path → hypothesis from your ASR system
        """
        with open(manifest_path) as f:
            manifest = json.load(f)

        report = WEREvalReport()
        for item in manifest:
            audio_path = item["audio"]
            reference = item["text"]
            hypothesis = transcripts.get(audio_path, "")

            if self.normalize:
                reference = normalize_text(reference)
                hypothesis = normalize_text(hypothesis)

            score = compute_wer(reference, hypothesis)
            report.total_words += score.ref_word_count
            report.total_substitutions += score.substitutions
            report.total_deletions += score.deletions
            report.total_insertions += score.insertions
            report.per_file.append({
                "audio": audio_path,
                "wer": score.wer,
                "cer": score.cer,
                "ref_words": score.ref_word_count,
                "errors": score.total_errors,
            })

        return report

    def evaluate_from_directory(
        self,
        audio_dir: str,
        transcripts: Dict[str, str],
    ) -> WEREvalReport:
        """
        Audio dir with .wav + .txt pairs.
        """
        audio_dir = Path(audio_dir)
        report = WEREvalReport()

        for txt_path in sorted(audio_dir.glob("*.txt")):
            audio_path = txt_path.with_suffix(".wav")
            if not audio_path.exists():
                continue

            reference = txt_path.read_text().strip()
            hypothesis = transcripts.get(str(audio_path), "")

            if self.normalize:
                reference = normalize_text(reference)
                hypothesis = normalize_text(hypothesis)

            score = compute_wer(reference, hypothesis)
            report.total_words += score.ref_word_count
            report.total_substitutions += score.substitutions
            report.total_deletions += score.deletions
            report.total_insertions += score.insertions
            report.per_file.append({
                "audio": str(audio_path),
                "wer": round(score.wer, 4),
                "cer": round(score.cer, 4),
                "ref_words": score.ref_word_count,
            })

        return report
