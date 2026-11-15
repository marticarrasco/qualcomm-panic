"""Original bilingual exercises and a fictional practice history for --demo."""
import json
import sqlite3
import time
from array import array

from .build_bank import SCHEMA

NAME = "Aina"
START_TOPIC = "fractions"
START_PROBLEM = "demo-pizza"

# The final item in each row is the verified answer letter.
EXERCISES = [
    ("demo-pizza", "fractions", 1,
     "Al matí menges 1/3 d'una pizza i a la tarda, 1/4 de la mateixa pizza. Quina fracció en queda?",
     "In the morning you eat 1/3 of a pizza and in the afternoon you eat 1/4 of the same pizza. What fraction is left?",
     ("1/12", "7/12", "1/2", "5/12", "3/4"), "D"),
    ("demo-aigua", "fractions", 0,
     "Una gerra és plena. En buides 1/4. Quina fracció d'aigua hi queda?",
     "A jug is full. You pour out 1/4. What fraction of the water is left?",
     ("1/4", "1/2", "3/4", "1/3", "1"), "C"),
    ("demo-jardi", "fractions", 2,
     "Plantes flors a 2/5 del jardí i herbes a 1/4 del jardí, en zones diferents. Quina fracció queda lliure?",
     "You plant flowers in 2/5 of the garden and herbs in 1/4, in separate areas. What fraction is still free?",
     ("7/20", "13/20", "3/9", "3/20", "1/4"), "A"),
    ("demo-boto", "fractions", 0,
     "En una capsa, 3/5 dels botons són blaus. Quina fracció dels botons és d'un altre color?",
     "In a box, 3/5 of the buttons are blue. What fraction of the buttons are a different colour?",
     ("3/5", "1/5", "1/2", "4/5", "2/5"), "E"),
    ("demo-cartolina", "fractions", 2,
     "Pintes 1/6 d'una cartolina de blau i 1/3 de verd, sense sobreposar els colors. Quina fracció queda per pintar?",
     "You colour 1/6 of a card blue and 1/3 green, with no overlap. What fraction remains uncoloured?",
     ("1/6", "1/2", "2/9", "1/3", "2/3"), "B"),
    ("demo-quaderns", "arithmetic", 4,
     "Compres 5 paquets de 8 quaderns. En regales 7 i reparteixes la resta en 3 piles iguals. Quants n'hi ha a cada pila?",
     "You buy 5 packs of 8 notebooks. You give away 7 and divide the rest into 3 equal stacks. How many are in each stack?",
     ("10", "12", "11", "13", "15"), "C"),
    ("demo-excursio", "arithmetic", 5,
     "Hi ha 4 grups de 18 alumnes. A cada grup hi falten 3 alumnes. Cada alumne que hi va paga 6 euros. Quant recapten?",
     "There are 4 groups of 18 pupils. 3 pupils are absent from each group. Each pupil attending pays 6 euros. How much is collected?",
     ("360", "414", "432", "342", "390"), "A"),
    ("demo-fitxes", "arithmetic", 3,
     "Tens 6 bosses amb 7 fitxes a cadascuna. En perds 5. Quantes fitxes et queden?",
     "You have 6 bags with 7 counters each. You lose 5. How many counters are left?",
     ("42", "30", "47", "35", "37"), "E"),
]

# Two verified moves per exercise, used to ground the model and as a fallback
# when it cannot produce a useful hint. Neither move gives the final value.
COACHING_STEPS = {
    "demo-aigua": {
        "cat": ("Dibuixa la gerra com una barra de quatre parts iguals i ratlla la part que en buides.",
                "Quina fracció de la gerra representen les parts que encara no has ratllat?"),
        "eng": ("Draw the jug as a bar with four equal parts and cross out the part you pour away.",
                "What fraction of the jug do the bars you haven't crossed out represent?"),
    },
    "demo-jardi": {
        "cat": ("Dibuixa el jardí com una quadrícula de vint caselles. Quantes en pintaries per representar els dos cinquens de flors?",
                "Marca les caselles de les herbes amb un altre patró. Les caselles buides són les que has de comptar."),
        "eng": ("Draw the garden as a grid of twenty cells. How many would you shade to show the two fifths of flowers?",
                "Mark the cells for the herbs with a different pattern. Count the cells that are still empty."),
    },
    "demo-boto": {
        "cat": ("Dibuixa cinc botons i pinta’n tres de blau per representar els tres cinquens.",
                "Compta els botons que has deixat sense pintar. El denominador continua sent el total de botons dibuixats."),
        "eng": ("Draw five buttons and colour three blue to show the three fifths.",
                "Count the buttons you left uncoloured. The denominator is still the total number of buttons you drew."),
    },
    "demo-cartolina": {
        "cat": ("Dibuixa la cartolina dividida en sis parts iguals. Quantes parts hauries de marcar per representar la zona verda?",
                "Marca la zona blava en una part diferent i fixa’t en quantes parts encara no tenen color."),
        "eng": ("Draw the card in six equal parts. How many parts would you mark to show the green area?",
                "Mark the blue area in a separate part, then look at the parts still uncoloured."),
    },
    "demo-quaderns": {
        "cat": ("Comença multiplicant els cinc paquets pels vuit quaderns de cada paquet.",
                "Resta els quaderns que regales abans de repartir els que et queden entre les tres piles."),
        "eng": ("Start by multiplying the five packs by the eight notebooks in each one.",
                "Subtract the notebooks you gave away before sharing what remains among the three stacks."),
    },
    "demo-excursio": {
        "cat": ("Resta els alumnes que falten dels que hi ha en un grup per saber quants hi van per grup.",
                "Compta els assistents dels quatre grups i aplica el preu només als alumnes que hi van."),
        "eng": ("To find how many pupils go in each group, subtract those absent from the group size.",
                "Count the attendees from all four groups and apply the price only to pupils who go."),
    },
    "demo-fitxes": {
        "cat": ("Multiplica les sis bosses per les set fitxes que hi ha dins de cada bossa.",
                "Resta les fitxes perdudes del total original. Les fitxes es perden del total, no de cada bossa."),
        "eng": ("Multiply the six bags by the seven counters in each bag.",
                "Subtract the lost counters from the original total, once for all the bags."),
    },
}

# Earlier fictional sessions. These are unavailable as live exercises.
HISTORY = [
    ("demo-hist-fraccions-1", "fractions", 1, "solved", 0, ""),
    ("demo-hist-fraccions-2", "fractions", 2, "failed", 3, "tip"),
    ("demo-hist-fraccions-3", "fractions", 2, "revealed", 1, "tip"),
    ("demo-hist-aritmetica-1", "arithmetic", 1, "solved", 0, ""),
    ("demo-hist-aritmetica-2", "arithmetic", 2, "solved", 0, ""),
    ("demo-hist-aritmetica-3", "arithmetic", 3, "solved", 0, ""),
]


def build_bank(path):
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        for number, (pid, topic, level, cat, eng, options, label) in enumerate(EXERCISES, 1):
            db.execute("INSERT INTO problems VALUES (?, ?, 2026, ?, 1, ?, ?)",
                       (pid, level, number, label, topic))
            db.execute("INSERT INTO vectors VALUES (?, ?)", (pid, array("f", [1.0]).tobytes()))
            for lang, text in (("cat", cat), ("eng", eng)):
                db.execute("INSERT INTO texts VALUES (?, ?, ?)", (pid, lang, text))
                db.execute("INSERT INTO options VALUES (?, ?, ?)",
                           (pid, lang, json.dumps(dict(zip("ABCDE", options)))))
            for rank, (other, other_topic, other_level, *_) in enumerate(EXERCISES):
                if other != pid and other_topic == topic and other_level <= level:
                    kind = "easier" if other_level < level else "same_level"
                    db.execute("INSERT INTO neighbours VALUES (?, ?, ?, ?, 1, 1)", (pid, kind, rank, other))
    db.close()


def seed_student(tutor):
    """Write the fictional outcomes through the same level rules as live practice."""
    db = tutor.db
    with tutor.lock, db:
        if db.execute("SELECT 1 FROM students WHERE name = ?", (NAME,)).fetchone():
            return
        now = time.time()
        db.execute("INSERT INTO students (name, language, topic_choice, start_level, created) VALUES (?, 'cat', ?, 1, ?)",
                   (NAME, START_TOPIC, now - 720))
        for i, (pid, topic, level, status, wrong_tries, rungs) in enumerate(HISTORY):
            db.execute("""INSERT INTO history (name, problem_id, topic, level, wrong_tries, rungs, started)
                          VALUES (?, ?, ?, ?, ?, ?, ?)""",
                       (NAME, pid, topic, level, wrong_tries, rungs, now - 600 + 80 * i))
            tutor._close(tutor._current(NAME), status)
            db.execute("UPDATE history SET ended = started + ? WHERE name = ? AND problem_id = ?",
                       (18 + 3 * i, NAME, pid))
        tutor._serve(tutor._student(NAME))
