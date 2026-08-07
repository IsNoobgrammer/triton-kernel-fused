"""Megakernel family for sm120 (RTX PRO 6000 Blackwell).

Fused, custom-autograd replacements for whole blocks rather than single ops, so the backward
chooses what crosses the forward/backward boundary instead of autograd saving every intermediate.
"""
