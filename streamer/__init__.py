"""TPU-like streaming deployment (T2.4) -- PAPER FACT 3.11.

Reimplemented from the paper. The author repository's client/server are a useful
STRUCTURAL reference but cannot run as shipped (their `utils` module is missing
upstream, and `names = ['face']` is left over from LPYOLO), so nothing is vendored.
"""
