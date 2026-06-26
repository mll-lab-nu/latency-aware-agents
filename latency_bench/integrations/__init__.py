"""Integrations bridging latency_bench into external training loops.

These modules import the trainer/runtime they integrate with (e.g. starVLA),
so they are intentionally NOT imported from ``latency_bench.__init__`` — import
them explicitly from the integrating process, where those deps are present.
"""
