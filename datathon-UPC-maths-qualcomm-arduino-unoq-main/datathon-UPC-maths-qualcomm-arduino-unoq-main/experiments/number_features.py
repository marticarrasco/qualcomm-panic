"""Weak arithmetic evidence for a multiple-choice option, derived from the question."""
import math
import re
from fractions import Fraction
from itertools import combinations

_NUMBER = re.compile(r"(?<!\w)[+-]?\d{1,7}(?:[.,]\d{1,5})?(?!\w|\d)")


def evidence(text, choices):
    numbers = []
    for match in _NUMBER.finditer(text):
        raw = match[0].replace(",", ".")
        try:
            numbers.append(Fraction(raw))
        except ValueError:
            pass
        if len(numbers) >= 20:
            break
    if not numbers:
        return [{} for _ in choices]
    unary, binary, related, small = set(), set(), set(), set()
    for n in set(numbers):
        unary.update((2*n, n/2, n*n, n+1, n-1))
        if n >= 0 and n.denominator == 1:
            root = math.isqrt(int(n))
            if root * root == n:
                unary.add(Fraction(root))
    for a, b in combinations(numbers, 2):
        binary.update((a+b, abs(a-b), a*b))
        if a:
            binary.add(b/a)
        if b:
            binary.add(a/b)
        if a.denominator == b.denominator == 1:
            ia, ib = abs(int(a)), abs(int(b))
            if ia and ib:
                related.update((math.gcd(ia, ib), math.lcm(ia, ib), ia % ib, ib % ia))
            if max(ia, ib) <= 25:
                related.add(math.comb(max(ia, ib), min(ia, ib)))
    for a, b, c in combinations(numbers[:10], 3):
        small.update((a+b+c, a*b*c, abs(a+b-c), abs(a+c-b), abs(b+c-a),
                      abs(a*b-c), abs(a*c-b), abs(b*c-a), a*b+c, a*c+b, b*c+a))
    hits = {"unary": unary, "binary": binary, "integer_pair": related, "ternary": small}
    out = []
    for value in choices:
        out.append({
            f"calc_{name}": float(value in pool) - sum(x in pool for x in choices) / 5
            for name, pool in hits.items()
        })
    return out
