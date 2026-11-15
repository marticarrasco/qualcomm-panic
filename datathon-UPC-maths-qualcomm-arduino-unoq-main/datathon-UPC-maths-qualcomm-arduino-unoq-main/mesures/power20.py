"""How much can a 20-problem test tell teams apart?"""
import math


def pmf(n, p):
    return [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]


n = 20
base = pmf(n, 0.20)
print("n = 20, one team at 20% (no signal) against a team with true accuracy p:")
for p in (0.25, 0.30, 0.35, 0.40, 0.50):
    other = pmf(n, p)
    win = sum(other[j] * sum(base[:j]) for j in range(n + 1))
    tie = sum(other[j] * base[j] for j in range(n + 1))
    sd = math.sqrt(p * (1 - p) / n)
    print(f"  p={p:.2f}: mean {n * p:4.1f}/20, sd {100 * sd:4.1f} pp;  P(strictly more right) = {100 * win:4.1f}%, "
          f"tie {100 * tie:4.1f}%, fewer {100 * (1 - win - tie):4.1f}%")
print(f"no-signal guesser, answers spread over the letters: P(>=5/20) = {100 * sum(base[5:]):.1f}%, "
      f"P(>=6/20) = {100 * sum(base[6:]):.1f}%, P(>=7/20) = {100 * sum(base[7:]):.1f}%, P(<=2/20) = {100 * sum(base[:3]):.1f}%")
cdf = [sum(base[:k + 1]) for k in range(n + 1)]
for teams in (10, 20, 30):
    med = next(k for k in range(n + 1) if cdf[k] ** teams >= 0.5)
    print(f"  best score among {teams} no-signal teams: median {med}/20 = {100 * med / n:.0f}%; "
          f"P(best >= 8/20) = {100 * (1 - cdf[7] ** teams):.0f}%")
print(f"sd of a no-signal guesser on 20 problems: {100 * math.sqrt(0.2 * 0.8 / 20):.1f} points")
