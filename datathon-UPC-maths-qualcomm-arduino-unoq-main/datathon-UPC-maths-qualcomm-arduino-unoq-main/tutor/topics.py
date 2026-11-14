"""Topics: a keyword classifier you can read, plus names and method tips.

The classifier is deliberately simple. Every topic has a list of
(regular expression, weight). A problem gets the topic with the highest total
weight; each pattern counts once. `explain()` shows which patterns fired, so
any decision can be checked by hand. Standard library only.
"""
import re

# Order matters only to break ties: earlier wins.
TOPICS = ["fractions", "counting", "numbers", "geometry_measure", "geometry_spatial",
          "logic", "measures", "patterns", "algebra", "arithmetic"]

NAMES = {
    "arithmetic": {"eng": "Arithmetic & word problems", "cat": "Aritmètica i problemes"},
    "fractions": {"eng": "Fractions & percentages", "cat": "Fraccions i percentatges"},
    "geometry_measure": {"eng": "Geometry: area, perimeter, angles", "cat": "Geometria: àrea, perímetre, angles"},
    "geometry_spatial": {"eng": "Shapes, solids & spatial puzzles", "cat": "Figures, cossos i visió espacial"},
    "counting": {"eng": "Counting & probability", "cat": "Comptar i probabilitat"},
    "logic": {"eng": "Logic & strategy", "cat": "Lògica i estratègia"},
    "numbers": {"eng": "Number properties", "cat": "Propietats dels nombres"},
    "measures": {"eng": "Time, money & measures", "cat": "Temps, diners i mesures"},
    "patterns": {"eng": "Patterns & sequences", "cat": "Patrons i successions"},
    "algebra": {"eng": "Algebra & functions", "cat": "Àlgebra i funcions"},
}

# Rung 1 of the hint ladder: what a teacher would say before any help with
# this particular problem. Never mentions the problem, so it cannot leak.
TIPS = {
    "arithmetic": {
        "eng": "Write down what you know and what is asked, each on its own line. Then do one step at a time, "
               "and check that your result makes sense in the story (is it too big? too small?).",
        "cat": "Escriu què saps i què et demanen, cada cosa en una línia. Després fes un pas cada vegada, "
               "i comprova que el resultat té sentit en la història (és massa gran? massa petit?).",
    },
    "fractions": {
        "eng": "Decide first what the whole is. Pick an easy number for it (for example 100, or a number "
               "the denominators divide) and work out each part as an amount. A quick bar drawing helps.",
        "cat": "Decideix primer quin és el total. Tria-li un nombre fàcil (per exemple 100, o un nombre "
               "divisible pels denominadors) i calcula cada part com una quantitat. Un dibuix de barres ajuda.",
    },
    "geometry_measure": {
        "eng": "Copy the figure and write every length and angle you know on it. Look for shapes you know "
               "(rectangles, right triangles) hidden inside, or cut the figure into such pieces and add them up.",
        "cat": "Copia la figura i escriu-hi totes les longituds i angles que coneixes. Busca-hi figures "
               "conegudes (rectangles, triangles rectangles) amagades, o talla la figura en trossos així i suma'ls.",
    },
    "geometry_spatial": {
        "eng": "Do it with your hands or in your head, one piece at a time: follow a single corner, face or "
               "mark through the fold, turn or view. Then rule out the options where that one piece ends up wrong.",
        "cat": "Fes-ho amb les mans o mentalment, peça a peça: segueix un sol vèrtex, cara o marca "
               "durant el plec, el gir o la vista. Després descarta les opcions on aquesta peça queda malament.",
    },
    "counting": {
        "eng": "List the cases in an organised way (smallest first, or fix one thing and change the others) "
               "so you miss none and count none twice. Try a smaller version of the problem to find the rule.",
        "cat": "Fes la llista de casos de manera ordenada (primer el més petit, o fixa una cosa i canvia les altres) "
               "per no deixar-te'n cap ni comptar-ne cap dues vegades. Prova una versió més petita per trobar la regla.",
    },
    "logic": {
        "eng": "Make a small table of the possibilities. Assume one statement is true and follow what it forces; "
               "if you reach a contradiction, cross that case out. Start from the clue that restricts the most.",
        "cat": "Fes una petita taula de possibilitats. Suposa que una afirmació és certa i mira què obliga; "
               "si arribes a una contradicció, descarta aquest cas. Comença per la pista que restringeix més.",
    },
    "numbers": {
        "eng": "Try a few small examples and look at what stays the same: last digits, remainders, even or odd, "
               "factors. Writing a number as a product of primes often shows the answer.",
        "cat": "Prova uns quants exemples petits i mira què es manté: últimes xifres, residus, parell o senar, "
               "factors. Escriure un nombre com a producte de primers sovint ensenya la resposta.",
    },
    "measures": {
        "eng": "Put everything in the same unit first (all minutes, all cents, all centimetres). Then draw a "
               "timeline or a small table and move along it step by step.",
        "cat": "Posa-ho tot primer en la mateixa unitat (tot en minuts, en cèntims, en centímetres). Després dibuixa "
               "una línia del temps o una petita taula i avança-hi pas a pas.",
    },
    "patterns": {
        "eng": "Write the first few terms in a row and look at how you get from one to the next. Find where "
               "the pattern starts repeating and how long one full cycle is; then jump ahead using that length.",
        "cat": "Escriu els primers termes en fila i mira com es passa d'un al següent. Troba on el patró "
               "es comença a repetir i quant fa un cicle sencer; després salta endavant fent servir aquesta llargada.",
    },
    "algebra": {
        "eng": "Give the unknown a letter and translate each sentence into an equation. If that is hard, test an "
               "easy value (0, 1, 2) to see how the expression behaves before you do the general case.",
        "cat": "Posa una lletra a la incògnita i tradueix cada frase a una equació. Si costa, prova un valor "
               "fàcil (0, 1, 2) per veure com es comporta l'expressió abans de fer el cas general.",
    },
}

RULES = {
    "fractions": [
        (r"\bfractions?\b", 3), (r"percent|%", 3), (r"\bhal(f|ves)\b", 1.5), (r"\b(third|quarter|fifth|sixth|tenth)s?\b of", 2),
        (r"\b(one|two|three|a)[- ](third|quarter|fourth|fifth|sixth|seventh|eighth|tenth)s?\b", 2),
        (r"\bratio\b|\bproportion", 2.5), (r"numerator|denominator", 3), (r"\b\d+\s*/\s*\d+\b", 1.5),
        (r"discount|interest rate", 2),
    ],
    "counting": [
        (r"how many (different )?ways|in how many", 4), (r"different ways|number of ways", 3),
        (r"probabilit|at random|randomly|\bchance\b", 4), (r"\barrange|arrangement|\bpermut|\bcombination", 2),
        (r"how many (different|possible)", 2), (r"how many (such )?(two|three|four|five|six|\d+)[- ]digit", 3),
        (r"how many (numbers|integers|triangles|squares|rectangles|paths|routes|subsets|words|pairs)", 2.5),
        (r"to be sure|be certain|guarantee", 2.5), (r"\bchoose\b|\bchosen\b|\bselect", 1), (r"\bpaths?\b|\broutes?\b", 1),
        (r"\b(coin|dice|die)\b.*\b(thrown?|toss|roll)", 2),
    ],
    "numbers": [
        (r"\bdigits?\b", 2.5), (r"divisib|divisor|\bmultiples?\b", 3), (r"\bprimes?\b", 3), (r"remainder", 3),
        (r"\beven\b|\bodd\b", 1.5), (r"\bintegers?\b|whole numbers?|natural numbers?", 1.5),
        (r"common (multiple|divisor|factor)|\bgcd\b|\blcm\b", 3), (r"consecutive", 1.5),
        (r"\bfactors?\b", 2), (r"perfect (square|cube)|square number", 2.5), (r"\bpowers?\b of|\^", 1),
        (r"(two|three|four|five|six|\d+)[- ]digit", 2), (r"palindrom", 3), (r"sum of (the|its) digits", 2),
    ],
    "geometry_measure": [
        (r"\barea\b", 4), (r"perimeter", 4), (r"\bangles?\b|degrees|°", 3.5), (r"radius|diameter|circumference", 3),
        (r"\btriangles?\b", 2), (r"\brectangles?\b|rectangular", 2), (r"\bcircles?\b|semicircle", 2),
        (r"equilateral|isosceles|hypotenuse|right triangle|trapez|parallelogram|rhombus", 3),
        (r"polygon|hexagon|pentagon|octagon|quadrilateral", 2), (r"parallel|perpendicular|bisector|midpoint|diagonal", 2),
        (r"\blength of\b|\bside length|\bsides? (of|measur)", 1.5), (r"\bsquare [a-z]{1,3}\b with side|squares? (abcd|of side)", 2),
        (r"\b(cm|m|km|mm)\s?(2|²)|square (centi)?meters|cm²|m²", 3), (r"\bsegments?\b|\bvertex\b|\bvertices\b", 1.5),
        (r"\bvolume\b", 3), (r"\bshaded\b|\bgr[ae]y (part|area|region)|shaded region", 2),
    ],
    "geometry_spatial": [
        (r"\bcubes?\b", 3), (r"\bfold|unfold", 3.5), (r"\bnet\b|development of", 3),
        (r"\bviews?\b|seen from|looks? (at|from)|look like|from (above|the top|the front|the side|behind|outside)", 3),
        (r"\bpieces?\b", 2), (r"rotat|\bmirror|reflect|symmetr|upside down|\bflip|turned? over", 3),
        (r"\bfaces?\b|\bedges?\b", 1.5), (r"pyramid|prism|cylinder|sphere|\bsolids?\b|tetrahedr|polyhedr|\bcone\b", 3),
        (r"\btiles?\b|tiling|\bcover", 2), (r"\bcuts?\b|scissors", 1.5), (r"\bstack|\bbuilding blocks|\bblocks\b|bricks", 1.5),
        (r"which (of the )?(following )?(figure|shape|drawing|picture|piece|image)", 3), (r"\bshapes?\b", 1.5),
        (r"jigsaw|puzzle piece|missing piece|\boverlap|transparent", 2.5), (r"\bknots?\b|\bropes?\b|\bstring\b|\bribbon", 1.5),
        (r"\bdice\b|\bdie\b", 2), (r"\bgrid\b|\bmaze\b", 1), (r"\bpaper\b", 1.5),
    ],
    "logic": [
        (r"\btrue\b|\bfalse\b", 3), (r"\blie[sd]?\b|\bliars?\b|lying|\btruth", 4), (r"\bstatements?\b", 2.5),
        (r"\bwho\b", 2), (r"(taller|older|younger|shorter|heavier|lighter|faster) than", 2),
        (r"exactly (one|two|three) of", 2), (r"\bwins?\b|\bwinner|\bgame\b|strategy|\bmoves?\b|\btournament|\bmatch(es)?\b", 2),
        (r"\bcan we be sure|necessarily|certainly|must be|cannot be|impossible", 1.5),
        (r"\bneither\b|\beither\b|\bif and only if", 1), (r"\bbalance\b|\bscales?\b", 1.5),
        (r"which (of the following )?(statement|sentence)", 3), (r"\bfirst\b.*\blast\b|\bsecond\b.*\bthird\b|finish(ed)?|\border\b", 1),
        (r"\bsits?\b|sitting|\bneighbou?rs?\b|next to", 1.5), (r"sudoku|each row and (each )?column", 2),
    ],
    "measures": [
        (r"\bhours?\b|\bminutes?\b|\bseconds\b|\bclocks?\b|o'clock|\bwatch\b|\b\d{1,2}:\d{2}\b", 3),
        (r"\bdays?\b|\bweeks?\b|\bmonths?\b|calendar|monday|tuesday|wednesday|thursday|friday|saturday|sunday|birthday", 2),
        (r"euros?\b|€|\bcents?\b|\bprices?\b|\bcosts?\b|\bpa(y|id|ys)\b|\bcoins?\b|\bmoney\b|\bbuys?\b|\bbought\b|\bchange\b", 2.5),
        (r"\bkg\b|kilograms?|\bgrams?\b|\bweigh", 2.5), (r"\blit(er|re)s?\b|\bml\b", 2.5),
        (r"\bspeed\b|km/h|per hour|\bkm\b|kilomet", 2.5), (r"years? old|\bages?\b|\byears?\b", 1.5),
        (r"\bmeters\b|\bmetres\b|\bcm\b|centimet", 1), (r"temperature|thermometer", 2.5),
    ],
    "patterns": [
        (r"\bsequences?\b", 4), (r"\bpatterns?\b", 3.5), (r"\bnext\b (number|term|figure|one)|comes next", 3),
        (r"continu(es|ing|e) (in )?(the|this|like)|and so on|\.\.\.", 2), (r"\bterms?\b", 2), (r"\bth (position|term|place|number)|\d+(st|nd|rd|th) (position|term|place|letter|digit)", 3),
        (r"\brepeat", 2.5), (r"\brule\b|\bseries\b", 1.5), (r"\beach (number|term) (is|after)", 2.5),
    ],
    "algebra": [
        (r"\bequations?\b", 4), (r"\bfunctions?\b|f\s?\(x\)", 4), (r"\bgraphs?\b|\bline y\b|coordinate", 3), (r"parabola", 4),
        (r"polynomial|\broots?\b|inequalit", 3.5), (r"real numbers?|\breal\b", 2.5), (r"\blog\b|\bsin\b|\bcos\b|\btan\b|logarithm", 3.5),
        (r"value of (the expression|x|y|a|n)\b|\bexpression\b", 2.5), (r"√|sqrt|square root", 2),
        (r"\bx\s?[=+\-<>²^]|[=+\-]\s?x\b|\b[a-z]\s?=\s?[a-z0-9]|\b[a-z]²", 2), (r"\bsolutions?\b|\bsatisf", 2),
        (r"\bpositive numbers? [a-z]\b|numbers? [a-z],? [a-z],? and [a-z]|numbers? [a-z] and [a-z]\b", 2),
    ],
    "arithmetic": [
        (r"\bsum\b|\badd(s|ed|ing)?\b|\bplus\b", 1.5), (r"subtract|\bminus\b|difference", 1.5),
        (r"\bproduct\b|multipl(y|ied|ication)", 1.5), (r"\bdivide[sd]?\b|division", 1.5),
        (r"\bresult\b|\boperations?\b|calculat", 2), (r"in total|altogether|\btotal\b", 1.5),
        (r"more than|fewer than|less than|twice|times as (many|much)|\bdouble\b|\btriple\b", 1.5),
        (r"\baverage\b|\bmean\b", 2.5), (r"how many\b", 1), (r"\beach\b", 0.5),
        (r"\d\s?[+×x·÷\-]\s?\d", 2), (r"what number|which number|missing number|question mark", 1.5),
    ],
}

COMPILED = {topic: [(re.compile(pattern), weight) for pattern, weight in rules] for topic, rules in RULES.items()}
NUMBER_PREFIX = re.compile(r"^\s*\d{1,2}\s*[.)]\s*")


def clean(text):
    """The question without its leading number ("12. ...") and extra spaces."""
    return " ".join(NUMBER_PREFIX.sub("", text).split())


def scores(text):
    lowered = clean(text).lower()
    return {topic: sum(weight for regex, weight in rules if regex.search(lowered))
            for topic, rules in COMPILED.items()}


def classify(text, text_only=True):
    """Return the topic key. With no keyword at all, a picture problem with a
    short question ("Which piece is missing?") is treated as a spatial puzzle
    and anything else as a word problem."""
    found = scores(text)
    best = max(TOPICS, key=lambda topic: (found[topic], -TOPICS.index(topic)))
    if found[best] <= 0:
        return "geometry_spatial" if not text_only and len(clean(text)) < 160 else "arithmetic"
    return best


def explain(text):
    """The patterns that fired, per topic, for checking a decision by hand."""
    lowered = clean(text).lower()
    out = {}
    for topic, rules in COMPILED.items():
        hits = [(regex.search(lowered).group(0), weight) for regex, weight in rules if regex.search(lowered)]
        if hits:
            out[topic] = hits
    return out
