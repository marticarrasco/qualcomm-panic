"""Does a warm-up get reused when the later request continues the conversation instead of
changing the end of the same user message? Usage: python -I cache_test2.py [--url ...]"""
import argparse
import json
import urllib.request

SYSTEM = "You are a maths tutor for school students. You guide with short hints and never give the answer."
P1 = ("7. Anna has 3 boxes with 12 pencils each. She gives 8 pencils to her brother and then shares the rest "
      "equally among 4 friends. How many pencils does each friend get?\nA) 5\nB) 6\nC) 7\nD) 8\nE) 9")
P2 = ("4. A train leaves at 9:40 and arrives at 11:15. The return trip takes 20 minutes longer. "
      "How many minutes does the return trip take?\nA) 95\nB) 105\nC) 115\nD) 120\nE) 125")


def post(url, payload, timeout=120):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def ask(url, messages, max_tokens, label, grammar=None):
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 42, "cache_prompt": True}
    if grammar:
        payload["grammar"] = grammar
    r = post(url + "/v1/chat/completions", payload)
    t = r.get("timings", {})
    text = r["choices"][0]["message"]["content"]
    total = (t.get("cache_n") or 0) + (t.get("prompt_n") or 0)
    print(f"  {label:62} prompt {total:4}: reused {t.get('cache_n')!s:>4}, read {t.get('prompt_n')!s:>4}, "
          f"wrote {t.get('predicted_n')}  -> {text!r:.60}")
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8091")
    u = p.parse_args().url
    sys_msg = {"role": "system", "content": SYSTEM}

    print("A. Warm-up as a full turn, then the hint as the next turn of the same conversation")
    ask(u, [sys_msg, {"role": "user", "content": P2}], 1, "(another problem, to clear the slot)")
    ack = ask(u, [sys_msg, {"role": "user", "content": P1}], 1, "warm-up: the problem alone, 1 token")
    ask(u, [sys_msg, {"role": "user", "content": P1}, {"role": "assistant", "content": ack},
            {"role": "user", "content": "Give me one short hint for the first step. Do not say the answer."}],
        24, "hint: same conversation, next turn")
    ask(u, [sys_msg, {"role": "user", "content": P1}, {"role": "assistant", "content": ack},
            {"role": "user", "content": "Which topic is this problem about? One or two words."}],
        6, "another request on the same warm-up")

    print("B. The same, with a fixed acknowledgement forced by a grammar (so the turn is always the same)")
    ask(u, [sys_msg, {"role": "user", "content": P2}], 1, "(another problem, to clear the slot)")
    ack = ask(u, [sys_msg, {"role": "user", "content": P1 + "\nSay OK when you have read it."}], 2,
              "warm-up: problem + 'say OK'", grammar='root ::= "OK"')
    ask(u, [sys_msg, {"role": "user", "content": P1 + "\nSay OK when you have read it."},
            {"role": "assistant", "content": ack},
            {"role": "user", "content": "Give me one short hint for the first step. Do not say the answer."}],
        24, "hint: next turn after 'OK'")

    print("C. Control: no warm-up, the hint request in one message")
    ask(u, [sys_msg, {"role": "user", "content": P2}], 1, "(another problem, to clear the slot)")
    ask(u, [sys_msg, {"role": "user", "content": P1 + "\nGive me one short hint for the first step. Do not say the answer."}],
        24, "hint, cold")


if __name__ == "__main__":
    main()
