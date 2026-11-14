"""Budgeted text reasoning, with optional exact arithmetic."""
from contextlib import closing
import re
import time

from . import reasoner
from .arithmetic import calculate, match_option, numeric_options
from .config import CONFIG
from .llm import URL, format_problem
from .stream import chat

SHORT = ("Solve math contest questions even if written in another language. "
         "Write only the essential calculation or deduction. Check it against the options. "
         "End with a new line 'Answer: X' using the correct option letter A–E.\n"
         "Example: 3 bags of 4 apples, 2 eaten. A) 8 B) 9 C) 10 D) 12 E) 14\n"
         "3*4-2=10\nAnswer: C")
FORMULA = ("Solve math contest questions in any language. If the answer is a number, "
           "write the calculation on one line as Calc: <expression> using + - * / **, "
           "parentheses, gcd(a,b), lcm(a,b), comb(n,k), perm(n,k), or factorial(n). "
           "Use the units of the options. The calculator will evaluate it exactly. "
           "If you cannot use one expression, work briefly and finish on a new line "
           "as Answer: X, with the correct option letter.\n"
           "Example: Area of a square of perimeter 36?\nCalc: (36/4)**2")
VALUE = ("Solve math contest questions in any language. Do the key computation briefly, "
         "without restating the question. End with a new line Answer: X (option letter). "
         "When the answer is numeric, you may instead give one line Calc: <expression> "
         "in the units of the options.\nExample: 5 bags of 7, 2 lost; how many?\n"
         "Calc: 5*7-2")
BRIEF = """Solve the math problem. Use scratch notation only, no prose. End with Answer: X.

Q: 5 books cost $25. What do 7 cost? A) 20 B) 28 C) 30 D) 35 E) 40
25/5*7=35
Answer: D

Q: How many 3-digit numbers have digits summing to 3? A) 3 B) 4 C) 5 D) 6 E) 7
1: 102,111,120; 2: 201,210; 3: 300. 3+2+1=6
Answer: D

Q: A box has 4 red and 3 blue socks. How many draws guarantee a pair? A) 2 B) 3 C) 4 D) 5 E) 6
worst draw red + blue; next matches. 2+1=3
Answer: B"""
PICK = ("Solve the multiple-choice math question. Choose the correct answer based on the "
        "content of the options. Output a single option letter.")
THINK = ("Solve the math problem. Work it out using as few tokens as possible. "
         "After your working, write exactly Answer: X with the correct option letter.")
THINK_PICK = ("Solve this math contest problem. Reason briefly, then output the letter "
              "of the correct option with no other visible text.")
RETRIEVED = ("You solve multiple-choice math competition problems for school students. "
             "Examples show a question and its correct answer value, without options or working. "
             "Use a similar example only when its mathematical method applies. Work out the "
             "new problem briefly. Never reply with a bare letter: first write one short "
             "calculation or deduction, then finish with exactly 'Answer: X' using the correct "
             "option letter A–E.")
CALC_FIRST = """Solve the math problem. Reply with Calc: followed by one arithmetic expression that gives the answer in the units of the options. Use + - * / **, parentheses and optionally gcd, lcm, comb, perm or factorial. For problems that cannot be expressed that way, reply Answer: followed by the correct choice.
Equal split of 87 sweets among 3 children: Calc: 87/3
Maximum area of a rectangle with perimeter 24: Calc: (24/4)**2
Ways to select a team of 3 from 7: Calc: comb(7,3)
Which option is a square? A) triangle B) circle C) square: Answer: C"""
CALC_GRAMMAR = r'''
root ::= "Calc: " expr | "Answer: " [A-E]
expr ::= term (space ("**" | [-+*/]) space term){0,20}
term ::= "-"? (number | "(" space expr space ")" | function "(" space expr ("," space expr)? space ")")
function ::= "gcd" | "lcm" | "comb" | "perm" | "factorial"
number ::= [0-9]{1,10} ("." [0-9]{1,8})?
space ::= " "?
'''
COMPACT = """You are completing a short solution for a multiple-choice math contest. Match this format:
Question: How many triples of consecutive numbers sum to 18? A) 0 B) 1 C) 2 D) 3 E) 4
Work: n+(n+1)+(n+2)=18 -> n=5 -> unique
Answer: B
Question: What is the 20th term of 3, 7, 11, ...? A) 75 B) 76 C) 77 D) 78 E) 79
Work: 3+19*4=79
Answer: E
Continue with calculations or brief deductions, then Answer: X. Work in any language."""
PROMPTS = {"shots": reasoner.SYSTEM, "short": SHORT, "formula": FORMULA, "value": VALUE,
           "brief": BRIEF, "pick": PICK, "think": THINK, "calc": CALC_FIRST,
           "compact": COMPACT, "think-pick": THINK_PICK, "retrieved": RETRIEVED}
FINAL = re.compile(
    r"(?im)^\s*(?:\*{1,2})?(?:final answer|answer|resposta|respuesta|réponse|antwort)"
    r"(?:\*{1,2})?\s*[:：]\s*(?:\*{1,2})?\(?([A-E])\b"
)
CALC = re.compile(r"(?im)^\s*Calc\s*[:：]\s*([^\r\n]{1,160})(?:\r?\n|\Z)")


def decode(output, options, final_chunk=False):
    # A stream can end mid-number. Only use a formula after a newline or EOS.
    for match in CALC.finditer(output):
        if match.end() == len(output) and not (final_chunk or output.endswith("\n")):
            continue
        expression = (match[1].strip().replace("×", "*").replace("÷", "/")
                      .replace("−", "-").replace("^", "**"))
        letter = match_option(calculate(expression), options)
        if letter:
            return letter, "calculated"
    answer = FINAL.search(output)
    if answer:
        return answer[1].upper(), "ok"
    if final_chunk and re.fullmatch(r"\s*[A-E][.!]?\s*", output, re.IGNORECASE):
        return output.strip()[0].upper(), "ok"
    return None, "fallback"


def infer(problem, deadline, url=URL, profile="short", max_tokens=128, examples=(),
          board_budget=None):
    system = PROMPTS[profile]
    user = format_problem(problem)
    if profile == "compact":
        user = ("Question: " + user + "\nUse the pattern in the examples. "
                "Give one line of work, then Answer: X.")
    messages = [{"role": "system", "content": system}]
    if examples:
        if profile == "retrieved":
            for example in examples:
                value = (example.get("options") or {}).get(example.get("label"),
                                                             example.get("label", ""))
                messages.extend([
                    {"role": "user", "content": "Question: " + (example.get("text") or "")},
                    {"role": "assistant", "content": "Answer value: " + str(value)},
                ])
        else:
            # Keep the fixed system prefix cached; examples are charged on each question.
            demos = "\n\n".join(f"{format_problem(example)}\nAnswer: {example['label']}"
                                for example in examples)
            user = f"Examples of different problems:\n{demos}\n\nSolve this problem:\n{user}"
    messages.append({"role": "user", "content": user})
    payload = {"messages": messages,
               "max_tokens": max_tokens, "temperature": 0, "seed": int(CONFIG["SEED"]),
               "cache_prompt": True, "stream": True, "timings_per_token": True,
               "return_progress": True, "stream_options": {"include_usage": True}}
    if profile == "pick":
        payload["grammar"] = 'root ::= [A-E]'
        payload["max_tokens"] = 1
    if profile in {"think", "think-pick"}:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["reasoning_budget_tokens"] = max(1, max_tokens - 26)
        payload["reasoning_budget_message"] = "Now give the answer."
        payload["reasoning_format"] = "deepseek"
        if profile == "think-pick":
            payload["grammar"] = 'root ::= [A-E]'
    if profile == "calc":
        payload["grammar"] = CALC_GRAMMAR
        payload["max_tokens"] = min(max_tokens, 48)
    output, timings, finish = "", {}, None
    options = numeric_options(problem) if profile in {"formula", "value", "calc"} else None
    try:
        with closing(chat(url, payload, deadline)) as events:
            for event in events:
                timings.update(event.get("timings") or {})
                # prompt_n is the number actually prefetched, not the whole context.
                progress = event.get("prompt_progress") or {}
                if "total" in progress:
                    timings["full_prompt_n"] = progress["total"]
                    timings["cached_n"] = progress.get("cache", 0)
                if board_budget is not None and "prompt_n" in timings:
                    read, write, target = board_budget
                    if timings["prompt_n"] / read + timings.get("predicted_n", 0) / write > target:
                        return None, "token_budget", output, timings
                for choice in event.get("choices", ()):
                    output += (choice.get("delta") or {}).get("content") or ""
                    finish = choice.get("finish_reason") or finish
                answer, status = decode(output, options,
                                        final_chunk=finish == "stop" or profile in {"pick", "think-pick"})
                if answer:
                    return answer, status, output, timings
    except TimeoutError:
        return None, "timeout", output, timings
    except (OSError, ValueError):
        return None, "model_error", output, timings
    return None, "fallback", output, timings
