"""
EEGPipeline — the per-window processing engine.

Runs EEGStep instances sequentially, validates after each step,
logs statistics, and snapshots results for frontend export.

Two factory methods:
  from_config()        — full window pipeline (SQI + filters + features)
                         used in real-time inference
  window_only_config() — SQI + features only (no filters)
                         used after recording_filter() in batch preprocessing
"""

from __future__ import annotations
from typing import List, Set
import numpy as np

from .context import EEGContext
from .base import EEGStep


class EEGPipeline:
    def __init__(self, validate: bool = True):
        self._steps: List[EEGStep] = []
        self._export_set: Set[str] = set()
        self.validate = validate

    def add(self, step: EEGStep, export: bool = False) -> "EEGPipeline":
        self._steps.append(step)
        if export:
            self._export_set.add(type(step).__name__)
        return self

    def run(self, ctx: EEGContext) -> EEGContext:
        for step in self._steps:
            name = type(step).__name__
            ctx = step.process(ctx)

            if self.validate:
                step.validate(ctx)

            ctx.log(name)

            if name in self._export_set:
                ctx.artifacts[name] = ctx.clone_data()

        return ctx

    def describe(self) -> List[str]:
        return [type(s).__name__ for s in self._steps]

    # ------------------------------------------------------------------ #
    # Factory: SQI + features only (post recording_filter)               #
    # ------------------------------------------------------------------ #
    @classmethod
    def window_only(cls, cfg: dict) -> "EEGPipeline":
        """
        Minimal per-window pipeline: SQI scoring + feature extraction.
        Used in batch preprocessing (after recording_filter has already run).
        """
        from .steps.sqi import SQIComputer
        from .steps.features import FeatureExtractor

        pipeline = cls(validate=True)
        pipeline.add(SQIComputer(cfg["sqi"]), export=True)
        pipeline.add(FeatureExtractor(cfg["features"], fs=cfg["eeg"]["srate"]),
                     export=True)
        return pipeline

    # ------------------------------------------------------------------ #
    # Factory: full window pipeline (for real-time / inference)           #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: dict) -> "EEGPipeline":
        """
        Full per-window pipeline including filters.
        Used in real-time inference where we process one window at a time.
        Note: uses scipy SOS filters (MNE FIR requires long signals).
        """
        from .steps.filters import (
            MedianSpikeRemoval, HighpassFilter, LowpassFilter,
            NotchFilter, WaveletDenoiser,
        )
        from .steps.sqi import SQIComputer
        from .steps.features import FeatureExtractor

        f = cfg["filters"]
        w = cfg["wavelet"]
        p = cfg["eeg"]

        pipeline = cls(validate=True)
        pipeline.add(MedianSpikeRemoval(kernel_ms=f["median_kernel_ms"], fs=p["srate"]))
        pipeline.add(HighpassFilter(cutoff=f["highpass_hz"], fs=p["srate"]))
        pipeline.add(WaveletDenoiser(wavelet=w["wavelet"], level=w["level"],
                                     esu_thresh=w["esu_energy_thresh"]), export=True)
        for hz in f["notch_hz"]:
            pipeline.add(NotchFilter(freq=hz, q=f["notch_q"], fs=p["srate"]))
        pipeline.add(LowpassFilter(cutoff=f["lowpass_hz"], fs=p["srate"]))
        pipeline.add(SQIComputer(cfg["sqi"]), export=True)
        pipeline.add(FeatureExtractor(cfg["features"], fs=p["srate"]), export=True)
        return pipeline
