"""What does llama-server reuse between requests with this model? Reads timings.cache_n / prompt_n.
Usage: python -I cache_test.py [--url http://127.0.0.1:8091]"""
import argparse
import json
import urllib.request

SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")
# two made-up problems, so nothing from the datasets is needed here
P1 = ("7. Anna has 3 boxes with 12 pencils each. She gives 8 pencils to her brother and then shares the rest "
      "equally among 4 friends. How many pencils does each friend get?\nA) 5\nB) 6\nC) 7\nD) 8\nE) 9")
P2 = ("4. A train leaves at 9:40 and arrives at 11:15. The return trip takes 20 minutes longer. "
      "How many minutes does the return trip take?\nA) 95\nB) 105\nC) 115\nD) 120\nE) 125")
P1_TEXT_ONLY = P1.split("\nA)")[0]


def post(url, payload, timeout=120):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def ask(url, user, max_tokens=1, grammar="root ::= [A-E]", label=""):
    payload = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
               "max_tokens": max_tokens, "temperature": 0, "seed": 42, "cache_prompt": True}
    if grammar:
        payload["grammar"] = grammar
    r = post(url + "/v1/chat/completions", payload)
    t = r.get("timings", {})
    total = (t.get("cache_n") or 0) + (t.get("prompt_n") or 0)
    print(f"  {label:58} prompt tokens {total:4}: reused {t.get('cache_n')!s:>4}, read {t.get('prompt_n')!s:>4}, "
          f"wrote {t.get('predicted_n')}  -> {r['choices'][0]['message']['content']!r:.40}")
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8091")
    args = p.parse_args()
    u = args.url
    print("1. A new problem after a different one (only SYSTEM is shared)")
    ask(u, P2 + "\nAnswer:", label="P2, letter")
    ask(u, P1 + "\nAnswer:", label="P1, letter")
    print("2. The same request again")
    ask(u, P1 + "\nAnswer:", label="P1, letter, repeated")
    print("3. Warm-up, then the same problem with a different ending (the 'speculative prefill' idea)")
    ask(u, P2 + "\nAnswer:", label="(P2, to clear the slot)")
    ask(u, P1 + "\nAnswer:", label="warm-up: P1 + 'Answer:'")
    ask(u, P1 + "\nGive one short hint for the first step, without the answer.", max_tokens=24, grammar=None,
        label="then: P1 + hint request")
    ask(u, P1 + "\nWhich topic is this problem about? One word.", max_tokens=6, grammar=None,
        label="then: P1 + another request")
    print("4. The investigation's order: expression first (text only), then the letter (text + options)")
    ask(u, P2 + "\nAnswer:", label="(P2, to clear the slot)")
    ask(u, P1_TEXT_ONLY + "\nWrite ONLY the arithmetic expression (using numbers and +, -, *, /, parentheses) "
        "that computes the answer:\nExpression: ", max_tokens=16, grammar='root ::= [0-9+\\-*/(). ]+ "\\n"',
        label="micro-PoT: text + instruction")
    ask(u, P1 + "\nAnswer:", label="then the letter: text + options + 'Answer:'")
    print("5. Options in another order after the same statement (permutation debiasing)")
    perm = P1_TEXT_ONLY + "\nA) 9\nB) 5\nC) 6\nD) 7\nE) 8"
    ask(u, perm + "\nAnswer:", label="P1 with the options rotated")


if __name__ == "__main__":
    main()
