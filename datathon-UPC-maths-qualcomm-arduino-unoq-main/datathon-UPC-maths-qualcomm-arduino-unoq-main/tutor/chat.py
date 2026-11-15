"""Draft a short coaching hint with the local language model."""
import json
import re
import threading
import urllib.request
from uuid import uuid4

from .demo import COACHING_STEPS, START_PROBLEM
from .topics import NAMES, TIPS
from .tutor import TutorError, check_nudge

INTENTS = ("start", "why", "check", "example")
GRAMMAR = r'root ::= [^\r\n0-9=<>?!.]+ ("?" | ".")'
SYSTEM = ("Write one short maths hint in natural {language}. Guide the student using the teacher notes. "
          "Use the details of the exercise but do not state a result, a number or an option. Write at most 24 words.")
STEP_SYSTEM = {
    "cat": "Escriu una pista breu en català per a l'alumne. Adreça't a ell de tu. No facis el càlcul.",
    "eng": "Write a short hint in English for the pupil. Address them directly. Do not calculate the result.",
}
STEP_OPENINGS = {
    "cat": ("Dibuixa", "Marca", "Pinta", "Compta", "Escriu", "Multiplica", "Resta", "Agrupa", "Divideix",
            "Com", "Què", "Quina", "Quants", "Quantes", "Fixa’t"),
    "eng": ("Draw", "Mark", "Colour", "Count", "Write", "Multiply", "Subtract", "Group", "Divide",
            "How", "What", "Which", "Look"),
}
STEP_GRAMMARS = {
    lang: 'root ::= (' + ' | '.join(json.dumps(word, ensure_ascii=False) for word in openings)
    + r') (" " word){3,17} [?.]' + '\n' + r'word ::= [^ \r\n\t0-9?!.]+'
    for lang, openings in STEP_OPENINGS.items()
}
GUIDANCE = {
    "start": "Ask the student what they can draw, compare, or write down first.",
    "why": "Explain the reason behind the step the student asks about.",
    "check": "Help the student check what their last step actually measures.",
    "example": "Describe a simpler analogous situation without solving this exercise.",
}
QUANTITY = re.compile(r"\d+(?:[.,/]\d+)*(?:\s*%)?")
PIZZA_MISTAKES = re.compile(
    r"\b(?:cinc\s+dotzen\w*|five\s+twelfths?)\b"
    r"|\b(?:numeradors?|numerators?)\b.{0,24}\b(?:iguals?|equal|same)\b"
    r"|\b(?:queda|remaining)\b.{0,20}\b(?:suma|sum)\b", re.IGNORECASE,
)
BUTTON_MISTAKES = re.compile(
    r"\b(?:dos\s+cinquen\w*|two\s+fifths?|blanc\w*|negr\w*|vermell\w*|groc\w*|grogu\w*|verd\w*"
    r"|white|black|red|yellow|green)\b", re.IGNORECASE,
)
BUTTON_DRAWING = re.compile(r"\b(?:cinc botons|five buttons)\b", re.IGNORECASE)
BUTTON_MARKING = re.compile(r"(?=.*\b(?:tres|three)\b)(?=.*\b(?:blau\w*|blue)\b)", re.IGNORECASE)
BUTTON_UNCOLOURED = re.compile(r"pint|marc|dibuix|uncolou?red|unmarked|drew|draw", re.IGNORECASE)
KEYWORDS = (
    ("example", r"exemple|example|semblant|similar"),
    ("why", r"per qu[eè]|why|denominador|denominator|unitats|units"),
    ("check", r"comprov|check|correct|revis|resposta|answer|sum|rest"),
    ("start", r"comen[cç]|start|first|pista|hint|quin pas|what should i try"),
)

COACHING = {
    "eng": {
        "why": "Which information connects what you know to what the question asks? Explain that link in your own words.",
        "check": "Take your last step and check it against the question. Do the units match, and is the size plausible?",
        "example": "Try the same kind of problem with smaller numbers. Which step would still work in this problem?",
        "clarify": "Which of those steps did you get to? Tell me what you wrote down so we can check it.",
    },
    "cat": {
        "why": "Quina dada connecta el que saps amb el que et demanen? Explica aquesta relació amb les teves paraules.",
        "check": "Comprova l'últim pas amb l'enunciat. Hi coincideixen les unitats? El valor té sentit?",
        "example": "Prova el mateix tipus de problema amb nombres més petits. Quin pas podries repetir amb les dades originals?",
        "clarify": "Fins a quin pas has arribat? Digues-me què has escrit o dibuixat i ho comprovem.",
    },
}

PIZZA = {
    "eng": {
        "start": "Draw a pizza cut into twelve equal slices. How many slices would make a third? And a quarter?",
        "why": "Thirds and quarters are different-sized slices. Using equal-sized slices lets you count them together.",
        "check": "You counted what was eaten. The question asks what is left. How could you use your total to find that?",
        "example": "Try a different pizza: if you eat 3 of 10 slices and then 1 more, 6 are left. How could you use the same steps here?",
    },
    "cat": {
        "start": "Dibuixa una pizza de dotze talls iguals. Quants talls formen un terç? I un quart?",
        "why": "Els terços i els quarts són talls de mides diferents. Si fas servir talls iguals, els pots comptar junts.",
        "check": "Has comptat la part menjada. L'enunciat demana què queda. Com faries servir el teu total per trobar-ho?",
        "example": "Prova amb una altra pizza: si et menges 3 talls de 10 i després 1 més, en queden 6. Com aplicaries els mateixos passos aquí?",
    },
}


def demo_step(context, intent, last_hint):
    steps = COACHING_STEPS.get(context["problem"])
    if not steps:
        return None
    follow_up = bool(last_hint or (intent == "check" and context["tried"]))
    return steps[context["language"]][int(follow_up)]


def fallback_hint(context, intent, last_hint=""):
    lang = context["language"]
    step = demo_step(context, intent, last_hint)
    if step:
        return step if step != last_hint else COACHING[lang]["clarify"]
    if context["problem"] == START_PROBLEM:
        if intent != "check" or context["tried"].endswith("B"):
            return PIZZA[lang][intent]
        return {"cat": "Quina part has calculat fins ara, la menjada o la que queda? Com ho comprovaries?",
                "eng": "Which part have you found so far, the part eaten or the part left? How could you check?"}[lang]
    if intent == "start":
        hint = TIPS[context["topic"]][lang]
    else:
        hint = COACHING[lang][intent]
    return hint if hint != last_hint else COACHING[lang]["clarify"]


def build_messages(context, message, last_hint, intent):
    cat = context["language"] == "cat"
    step = demo_step(context, intent, last_hint)
    if step:
        prompt = (f"Exercici: {context['exercise']}\nPas suggerit: {step}" if cat else
                  f"Exercise: {context['exercise']}\nSuggested step: {step}")
        if last_hint:
            prompt += f"\n{'Pista anterior' if cat else 'Previous hint'}: {last_hint}"
        prompt += f"\n{'Dubte' if cat else 'Question'}: {message}"
        return [{"role": "system", "content": STEP_SYSTEM[context["language"]]},
                {"role": "user", "content": prompt}]
    notes = [GUIDANCE[intent]]
    if context["problem"] == START_PROBLEM:
        if intent == "check" and context["tried"].endswith("B"):
            notes = ["Ask the student whether their calculation measures the pizza they ate or the pizza that is left.",
                     "In Catalan, use «part menjada» and «part que queda»." if cat else ""]
        elif intent == "why":
            notes = ["Explain how equal denominators make the pieces in each fraction the same size.",
                     "In Catalan, use «talls de la mateixa mida»." if cat else ""]
        elif intent == "start":
            notes = ["Ask how to rewrite the fractions so they count slices of the same size. Focus on this first step.",
                     "In Catalan, use «talls de la mateixa mida»." if cat else ""]
    if context["support"]:
        notes.append("Use simple language and offer one concrete action.")
    if context["tried"] and context["problem"] != START_PROBLEM:
        notes.append("The student has chosen an incorrect option.")
        wrong = context["last_wrong"]
        if wrong and not QUANTITY.search(wrong):
            notes.append(f"Incorrect choice: {wrong[:80]}.")
    if not context["text_only"]:
        notes.append("The diagram is unavailable. Do not assume visual details.")
    exercise = QUANTITY.sub("…", context["exercise"][:480])
    prompt = f"Topic: {NAMES[context['topic']]['eng']}. Exercise: {exercise}\nTeacher notes: {' '.join(notes)}"
    if last_hint:
        prompt += f"\nPrevious hint: {last_hint}"
    return [{"role": "system", "content": SYSTEM.format(language="Catalan" if cat else "English")},
            {"role": "user", "content": f"{prompt}\nStudent: {message}"}]


def check_hint(raw, context, message, last_hint, intent="start"):
    if not isinstance(raw, str) or len(raw) > 240:
        return None
    text = raw.strip()
    if "\n" in text or re.search(r'[\d=<>\\$]|[¿]', text) or not text.endswith(("?", ".")):
        return None
    if context["problem"] == START_PROBLEM and PIZZA_MISTAKES.search(text):
        return None
    if context["problem"] == "demo-boto":
        if BUTTON_MISTAKES.search(text) or not re.search(r"\b(?:botons?|buttons?)\b", text, re.IGNORECASE):
            return None
        first_step = COACHING_STEPS["demo-boto"][context["language"]][0]
        if demo_step(context, intent, last_hint) == first_step:
            if not (BUTTON_DRAWING.search(text) and BUTTON_MARKING.search(text)):
                return None
        elif not BUTTON_UNCOLOURED.search(text):
            return None
    hint = check_nudge(text, "", context["correct_value"])
    normalize = lambda value: re.sub(r"[^\w]+", " ", value.casefold()).strip()
    if not hint or normalize(hint) in {normalize(message), normalize(last_hint), normalize(context["exercise"])}:
        return None
    return hint


class Chat:
    def __init__(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.active = False
        self.jobs = {}

    def start(self, context, message, previous, last_hint=None):
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= 220:
            raise TutorError("a message must contain 1–220 characters")
        message = " ".join(message.split())
        previous = previous if previous in INTENTS else None
        last_hint = " ".join(last_hint[:240].split()) if isinstance(last_hint, str) else ""
        explicit = next((key for key, pattern in KEYWORDS if re.search(pattern, message.lower())), None)
        intent = explicit or previous or "start"
        with self.lock:
            if not self.url or self.active:
                return self._result(context, intent, last_hint)
            self.active = True
            job = uuid4().hex
            self.jobs[job] = {"pending": True}
            if len(self.jobs) > 16:
                del self.jobs[next(iter(self.jobs))]
        threading.Thread(target=self._run, args=(job, context, message, last_hint, intent), daemon=True).start()
        return {"job": job}

    def status(self, job):
        with self.lock:
            if job not in self.jobs:
                raise TutorError("conversation turn expired")
            return self.jobs[job]

    @staticmethod
    def _result(context, intent, last_hint, hint=None):
        return {"intent": intent, "text": hint or fallback_hint(context, intent, last_hint), "used_model": hint is not None}

    def _run(self, job, context, message, last_hint, intent):
        hint = None
        try:
            request = urllib.request.Request(self.url, data=json.dumps({
                "messages": build_messages(context, message, last_hint, intent),
                "grammar": STEP_GRAMMARS[context["language"]] if context["problem"] in COACHING_STEPS else GRAMMAR,
                "max_tokens": 52, "temperature": 0, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=34) as reply:
                choice = json.loads(reply.read(8192))["choices"][0]
            if choice.get("finish_reason") == "stop":
                hint = check_hint(choice["message"]["content"], context, message, last_hint, intent)
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            pass
        finally:
            with self.lock:
                self.jobs[job] = self._result(context, intent, last_hint, hint)
                self.active = False
