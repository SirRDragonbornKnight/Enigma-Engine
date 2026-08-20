#!/usr/bin/env python
"""Clean world-knowledge Q&A -- her study material for the factual floor.

The general corpus (bulk-pulled) is where her broad knowledge comes from, and
it is noisy enough that eval answers came out garbled ("Pluto is the largest
planet", "Six days in a week"). This file is the counterweight: hand-curated
TRUE facts, asked many ways, answered in short plain sentences -- the same
teach-the-concept-not-the-flashcard structure that fixed identity
(identity_paraphrases.py, measured 2026-07-05 and 2026-07-15).

Authoring rules:
- Every fact is common, stable knowledge. Nothing contested, nothing niche.
- Answers are one or two SHORT sentences in plain English. Clarity is the
  second goal of this corpus (user ruling 2026-07-15: understandable
  sentences over character).
- Numbers are written as words where natural ("seven days") -- the BPE
  tokenizer splits digit strings inconsistently, so word-numbers recall
  better; measured on the 182M lineage, and the tokenizer behaviour it
  rests on is unchanged at 238M. Digits appear only where words would
  read strangely, or where an eval probe's want key IS the digit (the
  boiling-point row carries both surfaces on purpose).
- Answer OPENERS vary across intents (a shared opener lets greedy decoding
  jump rails between intents -- measured 2026-07-15).
- Eval probes are phrased DIFFERENTLY here on purpose; make_sft_data's
  _norm_q guard drops exact probe matches as a backstop.

Two renderers ride this data: gen_knowledge_examples() (Q x rotating-A chat
records for the SFT mix) and gen_knowledge_pretrain_text() (plain-text lines
for continued pretraining -- many textual forms per fact).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from enigma_engine.core.persona_content import PersonaContent, default_content

# (question phrasings, answer variants). Answers vary in wording, never in fact.
WORLD_FACTS: list[tuple[list[str], list[str]]] = [
    # ---------------------------------------------------------------- space
    (
        ["Which planet is the biggest one?", "What is the largest planet going around our sun?", "Of all the planets, which is largest?"],
        ["Jupiter. It is by far the largest planet in our solar system.",
         "Jupiter is the biggest planet — more massive than all the other planets combined."],
    ),
    (
        ["Which planet is smallest?", "What's the smallest planet in the solar system?"],
        ["Mercury is the smallest planet in our solar system.",
         "That's Mercury — the smallest planet, and also the one closest to the sun."],
    ),
    (
        ["Which planet is closest to the sun?", "What planet orbits nearest the sun?"],
        ["Mercury is the closest planet to the sun.",
         "Mercury — it sits nearest the sun of all the planets."],
    ),
    (
        ["Why is Mars called the red planet?", "Which planet is known as the red planet?"],
        ["Mars. Its surface is covered in iron-rich dust that looks red.",
         "Mars is the red planet — rusty iron dust on its surface gives it that color."],
    ),
    (
        ["Which planet has the famous rings?", "What planet is known for its rings?"],
        ["Saturn. Its rings are made of ice and rock circling the planet.",
         "Saturn is the ringed planet — bright rings of ice and rock."],
    ),
    (
        ["Which planet is our home?", "What planet are we standing on right now?", "Where do humans live in the solar system?"],
        ["Earth. It is the only planet known to support life.",
         "We live on Earth — the third planet from the sun."],
    ),
    (
        ["Which planet is the hottest?", "What's the hottest planet in the solar system?"],
        ["Venus. Its thick atmosphere traps heat, making it hotter than Mercury.",
         "Venus is the hottest planet — its heavy atmosphere holds the heat in."],
    ),
    (
        ["How many planets are in the solar system?", "How many planets orbit our sun?"],
        ["Eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
         "There are eight — Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."],
    ),
    (
        ["Is Pluto a planet?", "What happened to Pluto's planet status?"],
        ["Pluto is classified as a dwarf planet, not one of the eight planets.",
         "It's a dwarf planet — Pluto was reclassified, so the solar system counts eight planets."],
    ),
    (
        ["What does Earth orbit?", "What does our planet go around?"],
        ["Earth orbits the sun, taking one year to go around it.",
         "The sun. One full trip around it takes a year."],
    ),
    (
        ["What goes around the Earth?", "What is Earth's natural satellite?"],
        ["The moon orbits the Earth, taking about a month for each trip.",
         "That's the moon — Earth's only natural satellite."],
    ),
    (
        ["Is the sun a star or a planet?", "What kind of object is the sun?"],
        ["The sun is a star — a huge ball of hot gas that gives us light and heat.",
         "A star. The sun is simply the star closest to us."],
    ),
    (
        ["Which direction does the sun rise?", "Where does the sun come up?"],
        ["The sun rises in the east and sets in the west.",
         "In the east. It sets in the west every evening."],
    ),
    (
        ["What galaxy do we live in?", "What is the name of our galaxy?"],
        ["We live in the Milky Way galaxy.",
         "The Milky Way — our whole solar system is inside it."],
    ),
    (
        ["Why do we have day and night?", "What causes day and night?"],
        ["Earth spins. The side facing the sun has day, and the side facing away has night.",
         "Day and night come from Earth rotating — each full spin takes about twenty-four hours."],
    ),
    (
        ["What causes the tides?", "Why does the sea rise and fall?"],
        ["The moon's gravity pulls on the oceans, making the tides rise and fall.",
         "Mostly the moon — its gravity tugs the oceans as Earth turns."],
    ),
    (
        ["Who was the first person to walk on the moon?", "Who first set foot on the moon?"],
        ["Neil Armstrong, in 1969, during the Apollo 11 mission.",
         "Neil Armstrong — he stepped onto the moon in 1969 with Apollo 11."],
    ),
    (
        ["What is the North Star called?", "Which star points north?"],
        ["Polaris. It sits almost directly above the North Pole, so it points north.",
         "That's Polaris, the North Star — travelers have used it to find north for centuries."],
    ),
    (
        ["How long does sunlight take to reach Earth?", "How long does light travel from the sun to us?"],
        ["About eight minutes. Sunlight you see left the sun eight minutes ago.",
         "Roughly eight minutes from the sun to your eyes."],
    ),
    # ------------------------------------------------------- calendar / time
    (
        ["How many days does a week have?", "What's the number of days in one week?", "A week is how many days?"],
        ["Seven. Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday.",
         "A week has seven days."],
    ),
    (
        ["How many months are in a year?", "What's the number of months in one year?"],
        ["Twelve months, from January through December.",
         "A year has twelve months."],
    ),
    (
        ["How many days are in a year?", "How long is a year in days?"],
        ["Three hundred sixty-five days, and three hundred sixty-six in a leap year.",
         "A year lasts three hundred sixty-five days — leap years add one more."],
    ),
    (
        ["How many hours are in a day?", "A day has how many hours?"],
        ["Twenty-four hours.",
         "A day is twenty-four hours long."],
    ),
    (
        ["How many minutes are in an hour?", "An hour is how many minutes?"],
        ["Sixty minutes.",
         "There are sixty minutes in an hour."],
    ),
    (
        ["How many seconds are in a minute?", "A minute has how many seconds?"],
        ["Sixty seconds.",
         "One minute is sixty seconds."],
    ),
    (
        ["What are the four seasons?", "Name the seasons of the year."],
        ["Spring, summer, autumn, and winter.",
         "The four seasons are spring, summer, fall, and winter."],
    ),
    (
        ["Which month is the shortest?", "What's the shortest month of the year?"],
        ["February. It has twenty-eight days, or twenty-nine in a leap year.",
         "February is the shortest — twenty-eight days most years."],
    ),
    (
        ["How often do leap years happen?", "When do we get a leap year?"],
        ["Every four years. February gains an extra day, the twenty-ninth.",
         "Once every four years — that's when February gets a twenty-ninth day."],
    ),
    (
        ["How many years are in a decade?", "A decade is how long?"],
        ["Ten years.",
         "A decade is ten years."],
    ),
    (
        ["How many years are in a century?", "A century is how long?"],
        ["One hundred years.",
         "A century is one hundred years."],
    ),
    (
        ["Which days make up the weekend?", "What days are the weekend?"],
        ["Saturday and Sunday.",
         "The weekend is Saturday and Sunday."],
    ),
    # ------------------------------------------------------------ geography
    (
        ["How many continents are there?", "Name the continents of the world."],
        ["Seven: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.",
         "There are seven continents — Asia, Africa, North America, South America, Antarctica, Europe, and Australia."],
    ),
    (
        ["Which continent is the largest?", "What's the biggest continent?"],
        ["Asia. It is the largest continent in both land and population.",
         "Asia is the biggest continent by far."],
    ),
    (
        ["Which continent is the coldest?", "What's the coldest place on Earth, continent-wise?"],
        ["Antarctica. It is the coldest, windiest continent, covered in ice.",
         "Antarctica — an ice-covered continent at the South Pole."],
    ),
    (
        ["How many oceans are there?", "Name the oceans of the world."],
        ["Five: the Pacific, Atlantic, Indian, Arctic, and Southern oceans.",
         "There are five oceans — Pacific, Atlantic, Indian, Arctic, and Southern."],
    ),
    (
        ["Which ocean is the biggest?", "What's the largest ocean on Earth?"],
        ["The Pacific. It covers more area than all the land on Earth combined.",
         "The Pacific Ocean is the largest and deepest."],
    ),
    (
        ["What's the longest river in the world?", "Which river is longest?"],
        ["The Nile is usually called the longest river; the Amazon carries the most water.",
         "Most sources say the Nile, in Africa — though the Amazon is the biggest by water volume."],
    ),
    (
        ["What's the tallest mountain on Earth?", "Which mountain is the highest?"],
        ["Mount Everest, in the Himalayas, is the highest mountain above sea level.",
         "Mount Everest — the peak of the world, on the border of Nepal and China."],
    ),
    (
        ["What's the largest hot desert?", "Which desert is the biggest hot one?"],
        ["The Sahara, in northern Africa, is the largest hot desert.",
         "That's the Sahara — it stretches across much of northern Africa."],
    ),
    (
        ["What's the biggest country in the world?", "Which country has the most land?"],
        ["Russia. It spans Europe and Asia and is the largest country by area.",
         "Russia is the largest country by land area."],
    ),
    (
        ["Which country has the most people?", "What's the most populated country?"],
        ["India has the most people, with China a close second.",
         "India — it recently passed China as the most populous country."],
    ),
    (
        ["What's the capital of the United States?", "Which city is the US capital?"],
        # "Washington D.C." without the comma: _lead_fragment cuts at ", ",
        # and the comma form taught "the capital is Washington." in the cloze
        # (review 2026-08-13; the comma-less spelling is standard too).
        ["Washington D.C.",
         "The capital of the United States is Washington D.C."],
    ),
    (
        ["What's the capital of the United Kingdom?", "Which city is the UK capital?"],
        ["London.",
         "The UK's capital is London."],
    ),
    (
        ["What's the capital of France?", "Which city is France's capital?"],
        ["Paris.",
         "France's capital is Paris."],
    ),
    (
        ["What's the capital of Germany?", "Which city is Germany's capital?"],
        ["Berlin.",
         "Germany's capital is Berlin."],
    ),
    (
        # "What's the capital of Italy?" is an eval RESTRAINT probe -- held out.
        ["What is Italy's capital city?", "Which city is Italy's capital?"],
        ["Rome.",
         "Italy's capital is Rome."],
    ),
    (
        ["What's the capital of Spain?", "Which city is Spain's capital?"],
        ["Madrid.",
         "Spain's capital is Madrid."],
    ),
    (
        ["What's the capital of Japan?", "Which city is Japan's capital?"],
        ["Tokyo.",
         "Japan's capital is Tokyo."],
    ),
    (
        ["What's the capital of China?", "Which city is China's capital?"],
        ["Beijing.",
         "China's capital is Beijing."],
    ),
    (
        ["What's the capital of Russia?", "Which city is Russia's capital?"],
        ["Moscow.",
         "Russia's capital is Moscow."],
    ),
    (
        ["What's the capital of Canada?", "Which city is Canada's capital?"],
        ["Ottawa — not Toronto, which is just the biggest city.",
         "Canada's capital is Ottawa."],
    ),
    (
        ["What's the capital of Australia?", "Which city is Australia's capital?"],
        ["Canberra — not Sydney, which is just the biggest city.",
         "Australia's capital is Canberra."],
    ),
    (
        ["What's the capital of Brazil?", "Which city is Brazil's capital?"],
        ["Brasilia — a city built specifically to be the capital.",
         "Brazil's capital is Brasilia."],
    ),
    (
        ["What's the capital of India?", "Which city is India's capital?"],
        ["New Delhi.",
         "India's capital is New Delhi."],
    ),
    (
        ["What's the capital of Mexico?", "Which city is Mexico's capital?"],
        ["Mexico City.",
         "Mexico's capital is Mexico City."],
    ),
    (
        ["What's the capital of Egypt?", "Which city is Egypt's capital?"],
        ["Cairo.",
         "Egypt's capital is Cairo."],
    ),
    (
        ["What's the capital of South Korea?", "Which city is South Korea's capital?"],
        ["Seoul.",
         "South Korea's capital is Seoul."],
    ),
    (
        ["How many states are in the USA?", "How many states does America have?"],
        ["Fifty states.",
         "The United States has fifty states."],
    ),
    (
        ["What is the equator?", "Where is the equator?"],
        ["The equator is the imaginary line around Earth's middle, halfway between the poles.",
         "It's the line circling the center of the Earth — the hottest zone sits along it."],
    ),
    (
        ["What's the largest rainforest?", "Which rainforest is the biggest?"],
        ["The Amazon rainforest, in South America.",
         "The Amazon — the biggest rainforest on Earth, mostly in Brazil."],
    ),
    (
        ["What language do people speak in Brazil?", "What do Brazilians mostly speak?"],
        ["Portuguese. Brazil is the largest Portuguese-speaking country in the world.",
         "Brazilians speak Portuguese — not Spanish, which surprises many people."],
    ),
    (
        ["What language is spoken in Mexico?", "What do people in Mexico speak?"],
        ["Spanish.",
         "Mexico's main language is Spanish."],
    ),
    (
        ["What language has the most native speakers?", "Which language do the most people speak natively?"],
        ["Mandarin Chinese has the most native speakers; English is the most widely learned.",
         "Mandarin Chinese, by native speakers — English leads once you count second-language speakers."],
    ),
    # -------------------------------------------------------------- science
    (
        ["What is water made of?", "What are the elements in water?"],
        ["Hydrogen and oxygen — two hydrogen atoms and one oxygen atom, H2O.",
         "Water is H2O: two parts hydrogen, one part oxygen."],
    ),
    (
        ["At what temperature does water freeze?", "When does water turn to ice?"],
        ["Zero degrees Celsius, which is thirty-two degrees Fahrenheit.",
         "Water freezes at zero Celsius — thirty-two Fahrenheit."],
    ),
    (
        ["At what temperature does water boil?", "When does water start boiling?"],
        # Both surfaces on purpose: the sealed factual probe's want key is the
        # DIGIT "100" (whole-word numeric match), and the all-words answers
        # could never satisfy it -- a permanently unwinnable row (review
        # 2026-08-13). Word-numbers stay for recall; the digits make the row
        # winnable.
        # Digit surface FIRST: the cloze takes the first answer's lead
        # fragment, and a word-number lead re-created the unwinnable surface
        # one line down from the fix (audit 2026-08-14).
        ["100 degrees Celsius — one hundred — at sea level, which is 212 Fahrenheit.",
         "Water boils at 100 degrees Celsius at sea level."],
    ),
    (
        ["What are the states of matter?", "Name the main states of matter."],
        ["Solid, liquid, and gas — plus plasma, like the inside of stars.",
         "The three everyday ones are solid, liquid, and gas. Plasma is the fourth."],
    ),
    (
        ["What gas do we breathe in to live?", "What gas do humans need from the air?"],
        ["Oxygen. We breathe in oxygen and breathe out carbon dioxide.",
         "Oxygen keeps us alive — we exhale carbon dioxide in return."],
    ),
    (
        ["What is most of the air made of?", "What gas makes up most of the atmosphere?"],
        ["Nitrogen — about four-fifths of the air. Oxygen is most of the rest.",
         "Mostly nitrogen, roughly seventy-eight percent. Oxygen is about twenty-one percent."],
    ),
    (
        ["How do plants make their food?", "What is photosynthesis?"],
        ["Photosynthesis: plants use sunlight to turn water and carbon dioxide into food, releasing oxygen.",
         "Plants capture sunlight and use it to make food from water and carbon dioxide — the oxygen we breathe is their exhaust."],
    ),
    (
        ["What is gravity?", "Why do things fall down?"],
        ["Gravity is the force that pulls objects toward each other — it keeps us on the ground and the planets around the sun.",
         "Things fall because Earth's gravity pulls them toward its center."],
    ),
    (
        ["Which is faster, light or sound?", "Why do we see lightning before hearing thunder?"],
        ["Light is far faster than sound. That's why lightning flashes first and thunder arrives after.",
         "Light wins by a huge margin — the lightning reaches your eyes almost instantly, the thunder lags behind."],
    ),
    (
        ["How fast does light travel?", "What's the speed of light?"],
        ["About three hundred thousand kilometers per second — nothing moves faster.",
         "Roughly 300,000 kilometers every second. It's the universe's speed limit."],
    ),
    (
        ["How many poles does a magnet have?", "What are the poles of a magnet?"],
        ["Two: a north pole and a south pole. Opposite poles attract; like poles repel.",
         "Every magnet has a north and a south pole — opposites pull together, matching poles push apart."],
    ),
    (
        ["What is electricity?", "What flows through a wire?"],
        ["Electricity is the flow of electrons through a material like copper wire.",
         "Moving electrons — that flow through wires is what we call electricity."],
    ),
    (
        ["What's the hardest natural material?", "Which natural substance is hardest?"],
        ["Diamond. It's carbon arranged in an extremely strong crystal.",
         "Diamond is the hardest natural material — pure carbon under enormous pressure."],
    ),
    (
        ["What's the chemical symbol for gold?", "What letters stand for gold in chemistry?"],
        ["Au, from the Latin word aurum.",
         "Gold's symbol is Au — Latin for aurum."],
    ),
    (
        ["What is table salt made of?", "What's the chemistry of salt?"],
        ["Sodium and chlorine — sodium chloride, NaCl.",
         "Salt is sodium chloride: one sodium atom bonded to one chlorine atom."],
    ),
    (
        ["Why does ice float on water?", "How can ice float?"],
        ["Ice is less dense than liquid water, so it floats.",
         "Because freezing makes water expand — ice is lighter for its size, so it rides on top."],
    ),
    (
        ["Where does the sun get its energy?", "How does the sun make light?"],
        ["Nuclear fusion. The sun fuses hydrogen into helium, releasing enormous energy.",
         "The sun runs on fusion — hydrogen atoms merging into helium deep in its core."],
    ),
    (
        ["What makes a rainbow?", "How do rainbows form?"],
        ["Sunlight bending through water droplets splits into its colors — that's a rainbow.",
         "Rainbows appear when sunlight passes through raindrops and spreads into colors."],
    ),
    (
        ["What is everything made of?", "What are the building blocks of matter?"],
        ["Atoms. Everything you can touch is made of atoms bonded together.",
         "Matter is made of atoms — tiny particles made of protons, neutrons, and electrons."],
    ),
    (
        ["What are the parts of an atom?", "What's inside an atom?"],
        ["Protons and neutrons in the center, with electrons around them.",
         "An atom has a nucleus of protons and neutrons, with electrons orbiting it."],
    ),
    (
        ["What does DNA do?", "What is DNA for?"],
        ["DNA carries the genetic instructions for building and running a living thing.",
         "It's the instruction manual of life — DNA stores the code your cells follow."],
    ),
    (
        ["What are living things made of?", "What is the basic unit of life?"],
        ["Cells. Every living thing is made of one or more cells.",
         "The cell — the smallest building block of every living organism."],
    ),
    (
        ["What comes out of a volcano?", "What is lava?"],
        ["Molten rock. Underground it's called magma; once it erupts, it's lava.",
         "Lava — melted rock from deep inside the Earth."],
    ),
    (
        ["What causes earthquakes?", "Why does the ground shake in an earthquake?"],
        ["Huge plates of Earth's crust shifting and slipping against each other.",
         "Earthquakes happen when the plates of the Earth's crust suddenly move."],
    ),
    (
        ["How does rain happen?", "What is the water cycle?"],
        ["Water evaporates, rises, condenses into clouds, and falls back as rain — the water cycle.",
         "The sun lifts water into the sky as vapor, it forms clouds, and it falls again as rain."],
    ),
    # ------------------------------------------------------- body / biology
    (
        ["What does the heart do?", "What's the heart's job?"],
        ["The heart pumps blood around your body, delivering oxygen and nutrients.",
         "It's a pump — your heart pushes blood through your whole body, beat after beat."],
    ),
    (
        ["What do lungs do?", "What's the job of your lungs?"],
        ["Lungs take in oxygen when you breathe in and push out carbon dioxide when you breathe out.",
         "They handle breathing — oxygen in, carbon dioxide out."],
    ),
    (
        ["What does the brain do?", "What's the brain's job?"],
        ["The brain controls everything — thought, memory, movement, and the senses.",
         "It runs the whole show: thinking, feeling, remembering, and controlling your body."],
    ),
    (
        ["How many bones does an adult have?", "What's the number of bones in the human body?"],
        ["An adult has two hundred six bones. Babies are born with more, and some fuse as they grow.",
         "Two hundred six in an adult — children start with around three hundred that fuse over time."],
    ),
    (
        ["What's the largest organ of the body?", "Which human organ is the biggest?"],
        ["The skin. It covers the entire body and protects everything inside.",
         "Your skin — it's the body's largest organ."],
    ),
    (
        ["What are the five senses?", "Name the human senses."],
        ["Sight, hearing, smell, taste, and touch.",
         "The five senses are sight, hearing, taste, smell, and touch."],
    ),
    (
        ["What do kidneys do?", "What's the job of the kidneys?"],
        ["Kidneys filter waste out of your blood and turn it into urine.",
         "They're the blood's filters — kidneys clean waste from the blood."],
    ),
    (
        ["Are humans mammals?", "What kind of animal is a human?"],
        ["Yes — humans are mammals: warm-blooded, with hair, and fed on milk as babies.",
         "Humans are mammals, like whales, dogs, and elephants."],
    ),
    (
        ["What makes an animal a mammal?", "What defines mammals?"],
        ["Mammals are warm-blooded, have hair or fur, and feed their young with milk.",
         "Warm blood, fur or hair, and milk for their babies — that's what makes a mammal."],
    ),
    (
        ["How do fish breathe?", "How do fish get oxygen underwater?"],
        ["With gills. Gills pull oxygen straight out of the water.",
         "Fish breathe through gills, which take oxygen from water."],
    ),
    (
        ["How many legs does an insect have?", "What's the leg count for insects?"],
        ["Six legs. If it has eight, it's not an insect — it's probably a spider.",
         "Insects have six legs, always."],
    ),
    (
        ["How many legs does a spider have?", "What's the leg count for spiders?"],
        ["Eight legs. That's one way to tell spiders from insects, which have six.",
         "Spiders walk on eight legs."],
    ),
    (
        ["What's the largest animal ever?", "Which animal is the biggest on Earth?"],
        ["The blue whale — the largest animal that has ever lived, bigger than any dinosaur.",
         "The blue whale. Nothing alive, or ever alive, has been bigger."],
    ),
    (
        ["What's the tallest animal?", "Which animal stands the tallest?"],
        ["The giraffe. Its long neck lets it reach leaves other animals can't.",
         "Giraffes — the tallest animals on land."],
    ),
    (
        ["What's the fastest land animal?", "Which animal runs the fastest?"],
        ["The cheetah. In a short sprint, nothing on land catches it.",
         "The cheetah holds the land speed record among animals."],
    ),
    (
        ["What do bees make?", "What insect makes honey?"],
        ["Bees make honey from flower nectar — and they pollinate plants while gathering it.",
         "Honey comes from bees, which turn nectar into it inside the hive."],
    ),
    (
        ["What does a caterpillar turn into?", "What comes out of a chrysalis?"],
        ["A butterfly. The caterpillar transforms inside a chrysalis.",
         "A caterpillar becomes a butterfly — or a moth, depending on the species."],
    ),
    (
        ["What does a tadpole become?", "What grows from a tadpole?"],
        ["A frog. Tadpoles live in water and slowly grow legs and lungs.",
         "Tadpoles turn into frogs."],
    ),
    (
        ["Can penguins fly?", "Do penguins fly like other birds?"],
        ["No — penguins are birds, but their wings are built for swimming, not flying.",
         "They can't fly. Penguins use their wings as flippers to swim instead."],
    ),
    (
        ["Are bats birds?", "What kind of animal is a bat?"],
        ["Bats are mammals — the only mammals that truly fly.",
         "Not birds. Bats are flying mammals: fur, live young, milk."],
    ),
    (
        ["Are whales fish?", "What kind of animal is a whale?"],
        ["Whales are mammals, not fish. They breathe air and feed their calves milk.",
         "Not fish — whales are mammals that must surface to breathe."],
    ),
    (
        ["How many arms does an octopus have?", "What's the arm count for an octopus?"],
        ["Eight arms.",
         "An octopus has eight arms."],
    ),
    (
        ["What's in a camel's hump?", "Do camels store water in their humps?"],
        ["Fat, not water. The stored fat lets camels go a long time without food.",
         "Humps hold fat — it's an energy reserve, not a water tank."],
    ),
    (
        ["When did the dinosaurs die out?", "How long ago did dinosaurs go extinct?"],
        ["About sixty-six million years ago, most likely after a giant asteroid impact.",
         "Roughly sixty-six million years ago — an asteroid strike is the leading explanation."],
    ),
    # --------------------------------------------------------------- units
    (
        ["How many meters are in a kilometer?", "A kilometer is how many meters?"],
        ["One thousand meters.",
         "A kilometer is one thousand meters."],
    ),
    (
        ["How many centimeters are in a meter?", "A meter is how many centimeters?"],
        ["One hundred centimeters.",
         "A meter is one hundred centimeters."],
    ),
    (
        ["How many grams are in a kilogram?", "A kilogram is how many grams?"],
        ["One thousand grams.",
         "A kilogram is one thousand grams."],
    ),
    (
        ["How many is a dozen?", "A dozen means how many?"],
        ["Twelve.",
         "A dozen is twelve of something."],
    ),
    (
        ["What do Celsius and Fahrenheit measure?", "What kind of scales are Celsius and Fahrenheit?"],
        ["Temperature. Celsius is used in most of the world; Fahrenheit mainly in the United States.",
         "They're temperature scales — water freezes at zero Celsius or thirty-two Fahrenheit."],
    ),
    (
        ["What does a liter measure?", "What kind of unit is a liter?"],
        ["Volume — how much space a liquid takes up.",
         "A liter measures liquid volume, like a bottle of water."],
    ),
    # ----------------------------------------------------- language / misc
    (
        ["How many letters are in the English alphabet?", "What's the letter count of the alphabet?"],
        ["Twenty-six letters, from A to Z.",
         "The English alphabet has twenty-six letters."],
    ),
    (
        ["What are the vowels?", "Which letters are vowels in English?"],
        ["A, E, I, O, and U — and sometimes Y.",
         "The vowels are A, E, I, O, U, with Y sometimes counting too."],
    ),
    (
        ["What are the primary colors of paint?", "Which colors mix to make the others in paint?"],
        ["Red, yellow, and blue — mixing them makes the other colors.",
         "In paint: red, yellow, and blue."],
    ),
    (
        ["What do blue and yellow make?", "Mixing blue and yellow paint gives what?"],
        ["Green.",
         "Blue and yellow paint mix into green."],
    ),
    (
        ["What do red and white make?", "Mixing red and white gives what color?"],
        ["Pink.",
         "Red and white mix into pink."],
    ),
    (
        ["How many keys does a piano have?", "What's the key count of a full piano?"],
        ["Eighty-eight keys — fifty-two white and thirty-six black.",
         "A full-size piano has eighty-eight keys."],
    ),
    (
        ["How many squares are on a chessboard?", "What's the square count of a chessboard?"],
        ["Sixty-four squares, in an eight-by-eight grid.",
         "A chessboard has sixty-four squares."],
    ),
    (
        ["How many players are on a soccer team?", "How many players does each side field in soccer?"],
        ["Eleven players per team on the field, including the goalkeeper.",
         "Each soccer team fields eleven players."],
    ),
    (
        ["How often are the Olympic Games held?", "How many years between Olympics?"],
        ["Every four years — with summer and winter games alternating every two.",
         "The Olympics come every four years for each kind, summer and winter."],
    ),
    (
        ["Who painted the Mona Lisa?", "Which artist made the Mona Lisa?"],
        ["Leonardo da Vinci.",
         "The Mona Lisa was painted by Leonardo da Vinci."],
    ),
    (
        ["Who wrote Romeo and Juliet?", "Which writer created Romeo and Juliet?"],
        ["William Shakespeare.",
         "Romeo and Juliet is a play by William Shakespeare."],
    ),
    # ------------------------------------------------------ history / tech
    (
        ["When did World War Two end?", "What year did the Second World War finish?"],
        ["1945.",
         "World War Two ended in 1945."],
    ),
    (
        ["When did people first land on the moon?", "What year was the first moon landing?"],
        ["1969 — Apollo 11, with Neil Armstrong first on the surface.",
         "The first moon landing was in 1969."],
    ),
    (
        ["What sank the Titanic?", "How did the Titanic go down?"],
        ["It struck an iceberg in 1912 on its first voyage and sank in the North Atlantic.",
         "An iceberg. The Titanic hit it in 1912 and sank."],
    ),
    (
        ["Who built the first successful airplane?", "Who were the first people to fly a powered plane?"],
        ["The Wright brothers, Orville and Wilbur, in 1903.",
         "Orville and Wilbur Wright flew the first powered airplane."],
    ),
    (
        ["Who invented the telephone?", "Which inventor is credited with the telephone?"],
        ["Alexander Graham Bell is credited with inventing the telephone.",
         "The telephone is credited to Alexander Graham Bell."],
    ),
    (
        ["Who came up with the theory of relativity?", "Which scientist created relativity?"],
        ["Albert Einstein.",
         "The theory of relativity is Albert Einstein's work."],
    ),
    (
        ["Who described gravity after seeing a falling apple?", "Which scientist is famous for gravity?"],
        ["Isaac Newton — he worked out the laws of gravity and motion.",
         "That story belongs to Isaac Newton, who described how gravity works."],
    ),
    (
        ["Who proposed the theory of evolution?", "Which scientist wrote about natural selection?"],
        ["Charles Darwin, in On the Origin of Species.",
         "Charles Darwin — his theory of evolution by natural selection."],
    ),
    (
        ["Who invented the printing press?", "Which inventor made printing books practical?"],
        ["Johannes Gutenberg, in the fifteenth century.",
         "Gutenberg — his printing press made books widely available."],
    ),
    # ------------------------------------------------------------------
    # Floor widening, 2026-08-08 (ruled: ~138 -> wider, same class -- common,
    # stable, uncontested; the user caps and spot-checks). Same authoring
    # rules as the rest of the file.
    # ---------------------------------------------------------------- space
    (
        ["Does the moon make its own light?", "Why does the moon shine?"],
        ["No — moonlight is reflected sunlight bouncing off the moon's surface.",
         "The moon shines by reflecting the sun. It makes no light of its own."],
    ),
    (
        ["What is a shooting star?", "What are shooting stars really?"],
        ["A meteor — a bit of space rock burning up as it hits our atmosphere.",
         "Not a star at all: a small piece of rock or dust burning bright as it falls through the air."],
    ),
    (
        ["Why do the sun and moon look the same size?", "Is the sun bigger than the moon?"],
        ["The sun is vastly bigger — it just sits about four hundred times farther away, which evens out the view.",
         "Much bigger. The sun is enormous but far; the moon is small but close, so they look alike in the sky."],
    ),
    (
        ["How old is the Earth?", "What's the age of our planet?"],
        ["About four and a half billion years old.",
         "Roughly four and a half billion years — the sun is only slightly older."],
    ),
    (
        ["Can sound travel through space?", "Why is space silent?"],
        ["No — sound needs air or another medium to travel, and space is a vacuum.",
         "Space is silent because there's no air to carry the sound waves."],
    ),
    (
        ["What is the International Space Station?", "Do people live in space right now?"],
        ["A laboratory orbiting Earth — astronauts live and work aboard it for months at a time.",
         "Yes — crews live aboard the International Space Station as it circles the planet."],
    ),
    (
        ["What is a comet?", "Why does a comet have a tail?"],
        ["A ball of ice and dust. Near the sun it warms up, and the escaping gas makes the tail.",
         "Ice and dust from the outer solar system — the tail streams away from the sun as it melts."],
    ),
    (
        ["What is Jupiter's Great Red Spot?", "What's the big red mark on Jupiter?"],
        ["A giant storm — bigger than Earth, and it has raged for centuries.",
         "An enormous storm on Jupiter, wider than our whole planet."],
    ),
    # ------------------------------------------------------- calendar / time
    (
        ["How many months have thirty-one days?", "Which months are the longest?"],
        ["Seven months have thirty-one days: January, March, May, July, August, October, and December.",
         "Seven of the twelve — the longest months run thirty-one days."],
    ),
    (
        ["What's the difference between noon and midnight?", "When is noon?"],
        ["Noon is twelve o'clock in the middle of the day; midnight is twelve at night.",
         "Twelve in the daytime is noon. Twelve at night is midnight."],
    ),
    (
        ["Why do different places have different times?", "What are time zones for?"],
        ["Earth spins, so the sun is overhead at different moments around the world — time zones keep noon near midday everywhere.",
         "Because the planet rotates: when it's morning on one side, it's evening on the other, so clocks shift by region."],
    ),
    (
        ["Is it summer everywhere at the same time?", "When it's summer in the north, what is it in the south?"],
        ["No — the hemispheres are opposite. Northern summer is southern winter.",
         "Winter. The seasons flip between the northern and southern halves of the planet."],
    ),
    (
        ["How long is a fortnight?", "What does fortnight mean?"],
        ["Two weeks — fourteen days.",
         "A fortnight is fourteen days, which is two weeks."],
    ),
    (
        ["What day comes after Sunday?", "Which day follows the weekend?"],
        ["Monday.",
         "Monday — the start of the working week."],
    ),
    # ------------------------------------------------------------ geography
    (
        ["What's the capital of the Netherlands?", "Which city is the Dutch capital?"],
        ["Amsterdam.",
         "Amsterdam — though the government actually sits in The Hague."],
    ),
    (
        ["What's the capital of Greece?", "Which city is Greece's capital?"],
        ["Athens.",
         "Athens — one of the oldest cities in the world."],
    ),
    (
        ["What's the capital of Turkey?", "Which city is Turkey's capital?"],
        ["Ankara — not Istanbul, which is just the biggest city.",
         "Ankara. Istanbul is larger and more famous, but Ankara is the capital."],
    ),
    (
        ["What's the capital of Portugal?", "Which city is Portugal's capital?"],
        ["Lisbon.",
         "Lisbon, on the Atlantic coast."],
    ),
    (
        ["What's the capital of Argentina?", "Which city is Argentina's capital?"],
        ["Buenos Aires.",
         "Buenos Aires — Argentina's capital and largest city."],
    ),
    (
        ["What's the capital of Kenya?", "Which city is Kenya's capital?"],
        ["Nairobi.",
         "Nairobi — Kenya's capital."],
    ),
    (
        ["What's the capital of Thailand?", "Which city is Thailand's capital?"],
        ["Bangkok.",
         "Bangkok — Thailand's capital."],
    ),
    (
        ["What's the capital of Poland?", "Which city is Poland's capital?"],
        ["Warsaw.",
         "Warsaw — Poland's capital, on the Vistula."],
    ),
    (
        ["What's the capital of Sweden?", "Which city is Sweden's capital?"],
        ["Stockholm.",
         "Stockholm, spread across fourteen islands."],
    ),
    (
        ["What's the capital of Norway?", "Which city is Norway's capital?"],
        ["Oslo.",
         "Oslo — Norway's capital, at the head of the Oslofjord."],
    ),
    (
        ["What's the Great Barrier Reef?", "Where is the biggest coral reef?"],
        ["The world's largest coral reef, off the northeast coast of Australia.",
         "Off Australia — the Great Barrier Reef, the biggest living structure on Earth."],
    ),
    (
        ["Why do people float in the Dead Sea?", "What's special about the Dead Sea?"],
        ["It's so salty that the dense water pushes you up — floating takes no effort.",
         "The Dead Sea is many times saltier than the ocean, so swimmers float like corks."],
    ),
    (
        ["Which city is famous for its canals in Italy?", "What's special about Venice?"],
        ["Venice — its streets are canals, and boats do the work cars do elsewhere.",
         "Venice is built on a lagoon: canals for roads, bridges everywhere."],
    ),
    (
        ["Does anyone own Antarctica?", "Which country does Antarctica belong to?"],
        ["No country owns it — Antarctica is governed by an international treaty and reserved for science.",
         "None. A treaty keeps Antarctica peaceful, shared, and dedicated to research."],
    ),
    # -------------------------------------------------------------- science
    (
        ["Do metals conduct electricity?", "Why are wires made of metal?"],
        ["Yes — metals like copper carry current well, which is why wires are metal with rubber around them.",
         "Because metal conducts electricity easily. The rubber coating is the insulator that keeps it safe."],
    ),
    (
        ["What are the primary colors of light?", "Which colors mix to make white light on a screen?"],
        ["Red, green, and blue. Screens mix those three to make every color you see.",
         "For light it's red, green, and blue — different from paint's red, yellow, and blue."],
    ),
    (
        ["Does water expand when it freezes?", "Why do frozen pipes burst?"],
        ["Yes — water is unusual that way. Ice takes more room than the water it came from, which is what bursts pipes.",
         "Because freezing water expands. The ice needs more space than the pipe has."],
    ),
    (
        ["What do plants need to grow?", "What keeps a plant alive?"],
        ["Light, water, air, and nutrients from the soil.",
         "Sunlight, water, and carbon dioxide from the air — plus soil nutrients."],
    ),
    (
        ["What's the chemical symbol for iron?", "What letters stand for iron in chemistry?"],
        ["Fe, from the Latin word ferrum.",
         "Iron is Fe — Latin again, like gold's Au."],
    ),
    (
        ["Why do helium balloons float?", "What makes a party balloon rise?"],
        ["Helium is lighter than air, so the air around the balloon pushes it up.",
         "Because helium weighs less than the air it displaces — the balloon rises like a bubble in water."],
    ),
    (
        ["What is lightning?", "What makes lightning?"],
        ["A giant electric spark between a storm cloud and the ground, or between clouds.",
         "Electricity — a massive discharge built up inside a storm cloud."],
    ),
    (
        ["What does a magnet attract?", "Will a magnet pick up anything?"],
        ["Iron and steel, mostly. Wood, plastic, glass, and most other metals ignore it.",
         "Only certain metals — iron above all. Most materials don't respond to a magnet at all."],
    ),
    (
        # Second phrasing asks what makes it WORK, not what is inside it:
        # the cloze shifted here when how-does joined the skip list, and
        # "inside a battery" answered with "Chemistry" named the wrong
        # thing -- inside is electrodes and electrolyte (audit 2026-08-14).
        ["How does a battery work?", "What makes a battery work?"],
        ["A battery stores energy in chemicals and releases it as electricity when the circuit closes.",
         "Chemistry — two different materials and a reaction between them that pushes electrons through a wire."],
    ),
    (
        ["Can you drink seawater?", "Why can't we drink ocean water?"],
        ["No — the salt makes your body lose more water getting rid of it than you drank.",
         "Seawater dehydrates you. Your kidneys spend more water flushing the salt than the drink provided."],
    ),
    (
        ["What is glass made from?", "Where does glass come from?"],
        ["Melted sand — heated until it fuses, then cooled into the clear solid we use.",
         "Mostly sand, melted at very high heat with a few other minerals."],
    ),
    (
        ["What is paper made from?", "Where does paper come from?"],
        ["Wood pulp — trees ground and pressed into thin sheets.",
         "Mostly from trees: wood is pulped, flattened, and dried into sheets."],
    ),
    (
        ["Where does wool come from?", "What animal gives us wool?"],
        ["Sheep — their fleece is sheared, cleaned, and spun into yarn.",
         "From sheep, mainly. Goats give cashmere and mohair; the principle is the same."],
    ),
    (
        ["Where does silk come from?", "What makes silk?"],
        ["Silkworms — they spin the thread for their cocoons, and we weave it into cloth.",
         "From the cocoons of silkworms. Each cocoon is one long, fine thread."],
    ),
    # ------------------------------------------------------- body / biology
    (
        ["Why is blood red?", "What makes blood red?"],
        ["An iron-rich protein called hemoglobin — it carries the oxygen and gives blood its color.",
         "The oxygen-carrying protein in it is built around iron, and that reads as red."],
    ),
    (
        ["How many teeth does an adult have?", "What's the tooth count for adults?"],
        ["Thirty-two, counting the four wisdom teeth.",
         "Thirty-two with wisdom teeth in — twenty-eight without them."],
    ),
    (
        ["Do muscles push or pull?", "How do muscles move the body?"],
        ["Muscles only pull. They work in pairs — one bends the joint, its partner straightens it.",
         "They pull, never push. That's why they come in opposing pairs."],
    ),
    (
        ["How does the stomach digest food?", "What breaks food down in the stomach?"],
        ["With strong acid and churning — the stomach turns a meal into a paste the intestines can absorb.",
         "Acid and muscle. The stomach mixes food with digestive juices until it's ready to move on."],
    ),
    (
        ["Why do I breathe harder when I run?", "Why does exercise speed up breathing?"],
        ["Working muscles burn more oxygen, so your lungs and heart speed up to supply it.",
         "Because your muscles are asking for more oxygen — breathing faster is the delivery service scaling up."],
    ),
    (
        ["What are hair and nails made of?", "What's the material in fingernails?"],
        ["Keratin — the same tough protein in both, and in animal claws and horns.",
         "A protein called keratin. Hair, nails, claws, and horns are all versions of it."],
    ),
    (
        ["How fast does the heart beat?", "What's a normal resting heart rate?"],
        ["About seventy beats a minute at rest, faster the moment you move.",
         "Roughly sixty to a hundred a minute at rest — around seventy is typical."],
    ),
    (
        ["How much sleep do adults need?", "How many hours should I sleep?"],
        ["Roughly seven to nine hours a night for most adults.",
         "Most adults do best on seven to nine hours."],
    ),
    (
        ["What causes sunburn?", "Why does the sun burn skin?"],
        ["Ultraviolet light, not heat — you can burn on a cool, bright day.",
         "The sun's ultraviolet rays damage skin cells. The warmth isn't the culprit; the UV is."],
    ),
    (
        ["Which fruit is famous for vitamin C?", "Where do people get vitamin C?"],
        ["Citrus — oranges, lemons, and their relatives are the classic source.",
         "Oranges and other citrus fruit, along with peppers and berries."],
    ),
    (
        ["Is the funny bone a bone?", "Why does hitting the funny bone feel weird?"],
        ["It's a nerve, not a bone — the ulnar nerve passing the elbow with little padding.",
         "That jolt is a nerve getting struck where it runs close to the skin at the elbow."],
    ),
    (
        ["How much of the body is water?", "Are humans mostly water?"],
        ["About three-fifths of an adult body is water.",
         "Yes — roughly sixty percent of you is water."],
    ),
    # --------------------------------------------------------------- units
    (
        ["Which is longer, a mile or a kilometer?", "How many kilometers make a mile?"],
        ["A mile — it's about one point six kilometers.",
         "The mile is longer: one mile is roughly one and a half kilometers and change."],
    ),
    (
        ["How many milliliters are in a liter?", "A liter is how many milliliters?"],
        ["One thousand milliliters.",
         "A thousand — milli means a thousandth."],
    ),
    (
        ["How many kilograms are in a metric ton?", "A metric ton is how heavy?"],
        ["One thousand kilograms.",
         "A thousand kilograms — about the weight of a small car."],
    ),
    (
        ["What's half of a half?", "A quarter is what fraction?"],
        ["A quarter — one fourth of the whole.",
         "Half of a half is a quarter, or twenty-five percent."],
    ),
    (
        ["How many degrees are in a circle?", "A full circle is how many degrees?"],
        ["Three hundred sixty degrees.",
         "A full turn is three hundred sixty degrees."],
    ),
    (
        ["How many degrees is a right angle?", "What's the angle of a square corner?"],
        ["Ninety degrees.",
         "A right angle is ninety degrees — a quarter of a full circle."],
    ),
    # ----------------------------------------------------- language / misc
    (
        ["What's the difference between a noun and a verb?", "What is a noun?"],
        ["A noun names a person, place, or thing; a verb is an action or state.",
         "Nouns are the things; verbs are what the things do."],
    ),
    (
        ["What's the plural of mouse?", "How do you pluralize child and mouse?"],
        ["Mice — and child becomes children. English keeps a few old irregulars around.",
         "Mouse goes to mice, child goes to children — irregular plurals, learned one by one."],
    ),
    (
        ["What's a synonym?", "What's the difference between a synonym and an antonym?"],
        ["A synonym means nearly the same thing as another word; an antonym means the opposite.",
         "Synonyms match meanings — big and large. Antonyms oppose them — big and small."],
    ),
    (
        ["How many colors are in a rainbow?", "Name the colors of the rainbow."],
        ["Seven: red, orange, yellow, green, blue, indigo, and violet.",
         "Seven by tradition — red through violet in order."],
    ),
    (
        ["What do red and blue make?", "Mixing red and blue paint gives what?"],
        ["Purple.",
         "Purple — more red leans it toward magenta, more blue toward violet."],
    ),
    (
        ["What's a baby dog called?", "What are baby dogs and cats called?"],
        ["A puppy — and a baby cat is a kitten.",
         "Puppies. Kittens for cats, puppies for dogs."],
    ),
    (
        ["What's a group of wolves called?", "What do you call groups of wolves and lions?"],
        ["A pack — and lions form a pride.",
         "Wolves run in packs; lions live in prides."],
    ),
    (
        ["How many cards are in a standard deck?", "What's the card count of a full deck?"],
        ["Fifty-two, in four suits of thirteen.",
         "Fifty-two cards — plus jokers if the game wants them."],
    ),
    (
        ["How many sides does a triangle have?", "How many sides do a triangle and a hexagon have?"],
        ["Three — and a hexagon has six.",
         "A triangle has three sides; a hexagon has six."],
    ),
    # ------------------------------------------------------ history / tech
    (
        ["Who invented the light bulb?", "Which inventor is famous for the light bulb?"],
        ["Thomas Edison gets the credit — he made the first practical, long-lasting bulb.",
         "Edison, mostly — others made bulbs first, but his design actually worked for homes."],
    ),
    (
        ["Where is the Great Wall?", "What is the Great Wall of China?"],
        ["In China — a chain of fortifications built over many centuries to guard the northern frontier.",
         "China's ancient defensive wall, thousands of kilometers of it, built dynasty by dynasty."],
    ),
    (
        ["What were the pyramids built for?", "Why did Egypt build the pyramids?"],
        ["As tombs — monuments for the pharaohs, built to carry them into the afterlife.",
         "They're royal tombs. The great ones at Giza held Egypt's pharaohs."],
    ),
    (
        ["What language did ancient Rome speak?", "What was the language of the Roman Empire?"],
        ["Latin — the ancestor of Italian, Spanish, French, and Portuguese.",
         "Latin. The Romance languages all grew out of it."],
    ),
    (
        ["When was the First World War?", "What years did World War One run?"],
        ["From 1914 to 1918.",
         "1914 to 1918 — four years."],
    ),
    (
        ["When did Columbus cross the Atlantic?", "What year did Columbus reach the Americas?"],
        ["1492.",
         "In 1492, landing in the Caribbean."],
    ),
    (
        ["What's the difference between the internet and the web?", "Are the internet and the web the same thing?"],
        ["The internet is the network of connected computers; the web is one service running on it, alongside email and the rest.",
         "Not quite — the internet is the wiring, the web is the pages. The web rides on the internet."],
    ),
    (
        ["How do computers count?", "What is binary?"],
        ["In binary — just ones and zeros, switched fast enough to do everything else.",
         "Computers use binary: every number, letter, and picture is stored as patterns of ones and zeros."],
    ),
    (
        ["Where is the Eiffel Tower?", "When was the Eiffel Tower built?"],
        ["In Paris — built in 1889 for a world's fair, and never taken down.",
         "Paris. It went up in 1889 and was meant to be temporary."],
    ),
    (
        ["Who was Marie Curie?", "What is Marie Curie famous for?"],
        ["A pioneering scientist of radioactivity — the first person to win two Nobel Prizes.",
         "She discovered radium and polonium and won Nobel Prizes in two different sciences."],
    ),
    # ------------------------------------------------------ food / everyday
    (
        ["Why does bread rise?", "What makes bread fluffy?"],
        ["Yeast — it eats the dough's sugars and breathes out gas, and the bubbles puff the loaf.",
         "Tiny gas bubbles from yeast. Baking sets the dough around them."],
    ),
    (
        ["Does honey go bad?", "Why does honey last so long?"],
        ["Honey basically never spoils — it's too dry and acidic for microbes to live in.",
         "Sealed honey keeps nearly forever. Ancient jars of it have been found still edible."],
    ),
    (
        ["Why does boiling an egg make it hard?", "What happens when you cook an egg?"],
        ["Heat sets the egg's proteins — they unfold and lock together, turning the liquid solid.",
         "The proteins change shape with heat and tangle into a solid. There's no way back to raw."],
    ),
    (
        ["Where does chocolate come from?", "What is chocolate made from?"],
        ["Cacao beans — fermented, roasted, and ground into the base of every chocolate bar.",
         "From the seeds of the cacao tree, processed and usually sweetened."],
    ),
    (
        ["What's the difference between coffee and tea?", "Where do coffee and tea come from?"],
        ["Coffee is brewed from roasted beans; tea is steeped from dried leaves.",
         "Different plants entirely — coffee from a bean, tea from a leaf; both carry caffeine."],
    ),
    (
        ["Is a tomato a fruit or a vegetable?", "Why is a tomato technically a fruit?"],
        ["Botanically a fruit — it grows from a flower and carries the seeds. The kitchen treats it as a vegetable.",
         "A fruit to a botanist, a vegetable to a cook. Both are right in their own kitchens."],
    ),
    (
        ["What is cheese made from?", "How does milk become cheese?"],
        ["Milk — curdled, drained, and aged into the hundreds of kinds we eat.",
         "From milk: separate the curds from the whey, press them, and let time do the rest."],
    ),
    (
        ["How does salt preserve food?", "Why did people salt their food before fridges?"],
        ["Salt draws the water out, and without water the spoilage microbes can't grow.",
         "Because salt dries food from the inside — bacteria can't live in it."],
    ),
    (
        ["Why do onions make you cry?", "What makes cut onions sting your eyes?"],
        ["Cutting releases a gas that turns stinging in your eyes — tears are the rinse cycle.",
         "A sulfur compound escapes when the cells break, and your eyes water to wash it out."],
    ),
    (
        ["Why does popcorn pop?", "What makes popcorn kernels burst?"],
        ["A drop of water inside each kernel turns to steam and blows the shell open.",
         "Steam pressure. The sealed kernel holds until the inside boils, then bursts inside out."],
    ),
]

# ----------------------------------------------------- Enigma herself (self)
# Self/capability facts, ruled into T4 scope 2026-08-08 -- aligned with the
# identity-anchor rewrite (same measured numbers) so the two can never
# disagree. Facts only; the VOICE lives in identity_anchors.py.
#
# Persona content since wave 2a (2026-08-15): these are the facts a second AI
# cannot inherit, so the renderers take them from PersonaContent rather than
# from this list. It stays HERE, beside the world facts it is rendered with --
# the interface passes it through.
SELF_FACTS: list[tuple[list[str], list[str]]] = [
    (
        ["What is Enigma?", "What kind of AI is Enigma?", "Tell me about Enigma.",
         "What sort of AI is Enigma?", "Describe Enigma in a sentence."],
        ["Enigma is a local AI that runs entirely on its user's own machine, not in a cloud.",
         "A from-scratch local AI — it lives on the user's computer and answers there.",
         "A language model built and run locally: private, offline-capable, and its user's own."],
    ),
    (
        ["Was Enigma built from another model?", "Is Enigma a fine-tune of some bigger model?",
         "What base model is Enigma built on?", "Did Enigma start from someone else's weights?",
         "Is Enigma based on an existing model?"],
        ["No — Enigma was trained from scratch: its own architecture, its own tokenizer, its own weights.",
         "Enigma borrows no weights. Everything in it was trained from zero.",
         "None — there is no base model under Enigma; it started as random numbers and was trained up from nothing."],
    ),
    (
        ["How many parameters does Enigma have?", "How big is Enigma as a model?",
         "What's Enigma's parameter count?", "What size is the Enigma model?",
         "Is Enigma a big model?"],
        ["About 240 million parameters — small, and honest about it.",
         "Roughly 240 million parameters, every one trained from scratch.",
         "Small on purpose: around 240 million parameters running on one consumer GPU."],
    ),
    (
        ["How much text did Enigma train on?", "How many tokens went into Enigma's pretraining?",
         "What data did Enigma learn from?", "What was Enigma pretrained on?",
         "How big was Enigma's training corpus?"],
        ["About twenty-four billion tokens — code, prose, math, wikis, and books.",
         "Roughly twenty-four billion tokens of mixed text.",
         "A corpus of about twenty-four billion tokens: web prose, code, math, wikis, fandom, and books."],
    ),
    (
        ["Who built Enigma?", "Who trained Enigma?", "Whose project is Enigma?",
         "Which company makes Enigma?", "Did a lab build Enigma?"],
        ["SirRulean — one person, one machine, from scratch.",
         "A single person, SirRulean, on his own hardware.",
         "No company and no lab — Enigma is one person's build: SirRulean trained it himself."],
    ),
    (
        ["Can Enigma do exact arithmetic?", "How does Enigma handle hard math?",
         "What does Enigma do when it needs to calculate?", "Does Enigma guess at math?"],
        ["It has a calculator organ — for real arithmetic it computes instead of guessing.",
         "Enigma calls its calculator for exact numbers rather than eyeballing them.",
         "It doesn't guess arithmetic — the calculator organ does the number work."],
    ),
    (
        ["Does Enigma remember things between sessions?", "How does Enigma's memory work?",
         "Will Enigma remember a fact next week?", "Can Enigma forget things?",
         "Where does Enigma keep what it learns about its user?"],
        ["It keeps a memory store — facts the user shares persist, can be corrected, and can be forgotten on request.",
         "Through its memory store: told once, kept; corrected, updated; asked to forget, gone.",
         "Facts live in a store on the machine — kept across sessions, replaceable when corrected, deletable on request."],
    ),
    (
        ["Can Enigma speak out loud?", "Does Enigma have a voice organ?",
         "Is Enigma silent by default?", "How does Enigma's talk mode work?"],
        # No yes/no opener: 4 questions rotate over 3 answers, and "Yes --"
        # landed on "How does Enigma's talk mode work?" (review 2026-08-13).
        ["With talk mode on it speaks its answers aloud — and it starts silent by default.",
         "It has a voice; talk mode turns it on, and it boots silent on purpose.",
         "Talk mode is the switch: off means text only, on means it speaks its answers."],
    ),
    (
        ["Can Enigma make pictures?", "Does Enigma have an image maker?",
         "How does Enigma draw pictures?", "Does Enigma use a cloud service for images?"],
        ["Yes — asked to imagine something, it paints the image locally and saves it.",
         "It can. The imagine organ makes pictures right on the machine, no cloud involved.",
         "Images are made locally by its imagine organ and saved to its pictures folder."],
    ),
    (
        ["Can Enigma look at images?", "Does Enigma have eyes?",
         "How does Enigma see pictures?", "What happens when you show Enigma an image?"],
        ["With its eyes enabled it reads an image into a caption and works from that.",
         "Yes — enabled eyes turn a handed image into text it can reason about.",
         "A shown image becomes a caption through its eyes, and Enigma reasons over that text."],
    ),
    (
        ["Can Enigma search the internet?", "Does Enigma look things up online?",
         "When does Enigma search the web?", "Does Enigma ask permission before searching?"],
        ["Only when search is switched on — then it looks things up on its own when something's worth checking, through the user's own search service.",
         "It can, if its search organ is enabled; otherwise it works fully offline.",
         "With search on, Enigma decides when to look — no per-question permission; off, it stays fully local."],
    ),
    (
        ["Does Enigma need the internet?", "Does Enigma work offline?",
         "Can Enigma run without a connection?", "Does Enigma stop working when the internet drops?"],
        ["No internet needed — it runs entirely on the local machine.",
         "It works fully offline; that's the design.",
         "A dropped connection changes nothing — Enigma runs on the machine itself."],
    ),
    (
        ["Does Enigma send data to a cloud?", "Is Enigma private?",
         "Who can read what's said to Enigma?", "Does Enigma phone home?"],
        ["Conversations stay on the user's machine — no telemetry, no phone-home.",
         "Private by design: what's said to Enigma stays on the machine it runs on.",
         "Nobody else — there's no phone-home; the conversation lives and stays on the user's machine."],
    ),
]

# The table as authored: world facts, then whoever the AI is. The renderers
# rebuild it from the persona content instead, so the self half follows the
# pack; with no pack the two are the same objects in the same order.
KNOWLEDGE: list[tuple[list[str], list[str]]] = WORLD_FACTS + SELF_FACTS


def _table(content: "PersonaContent | None") -> list[tuple[list[str], list[str]]]:
    """World facts plus the persona's own self-facts, in authored order."""
    content = default_content() if content is None else content
    return WORLD_FACTS + list(content.self_facts)


def gen_knowledge_examples(seed: int = 55,
                           content: "PersonaContent | None" = None) -> list[dict]:
    """Emit clean fact-QA records (Q x two rotating answers per intent),
    deduped on the exact (question, answer) pair -- the identity_paraphrases
    rendering pattern applied to world knowledge."""
    rng = random.Random(seed)
    out: list[dict] = []
    for questions, answers in _table(content):
        for i, q in enumerate(questions):
            picks = [answers[i % len(answers)], answers[(i + 1) % len(answers)]]
            for a in picks:
                out.append({
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "category": "knowledge",
                })
    seen, uniq = set(), []
    for r in out:
        key = (r["messages"][0]["content"], r["messages"][1]["content"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    rng.shuffle(uniq)
    return uniq


# ---------------------------------------------------------------------------
# Continued-pretraining renderer (methods audit 2026-07-15). SFT QA alone left
# facts brittle across phrasings ("largest planet" -> Jupiter but "biggest
# planet" -> Saturn); the fix is MANY-FORMAT plain-text exposure. Each intent
# surfaces as declarative prose, a Q/A line, key-term-FINAL cloze(s) (next-token
# prediction trains the fact -> term mapping), and a fact-in-context line. ALL
# wording derives from KNOWLEDGE -- no new facts enter here, so the word-numbers
# convention and the TRUE-facts guarantee carry over unchanged.

_EVAL_PROBES = Path(__file__).resolve().parent / "data" / "eval" / "behavior_probes.jsonl"

# Leading answer fragments that are a clause, hedge, or bare yes/no -- never a
# key TERM. Any of these tokens in a candidate fragment disqualifies it.
_KEY_STOP_TOKENS = frozenset({
    "is", "are", "was", "were", "has", "have", "had", "hold", "holds",
    "handle", "handles", "say", "says", "make", "makes", "come", "comes",
    "turn", "turns", "spins", "orbits", "rises", "pulls", "runs", "walk",
    "walks", "controls", "filter", "filters", "evaporates", "lasts", "fields",
    "wins", "in", "not", "yes", "no", "it", "it's", "they", "they're",
    "that", "that's", "this", "we",
    # clause leads that slipped the net and produced clause-shaped "terms"
    # ("Because the planet rotates", "Computers use binary", "Muscles only
    # pull") -- review 2026-08-13
    "because", "use", "uses", "only",
})

# Answer openers that wrap the key term -- strip before extracting it.
_KEY_PREFIXES = ("That's ", "It's ", "We live on ")

# Sentence-level breaks (never a comma) and fragment-level breaks.
_SENTENCE_SEPS = (" — ", ". ", "; ", ": ")
_FRAGMENT_SEPS = (" — ", ": ", ". ", "; ", ", ")

# First words safe to lowercase when the key term lands mid-sentence
# ("...is twenty-four hours."); everything else is treated as a proper noun.
# Number-words are matched by RULE below, not enumerated -- the hand list
# drifted behind every floor widening ("is Three." vs "is seven." in one
# corpus, 33 lines, review 2026-08-13).
_MID_LOWER_FIRST = frozenset({
    "a", "an", "the", "about", "roughly", "mostly", "every", "once", "with",
    "moving", "molten", "nuclear", "fat", "green", "pink",
    "temperature", "volume", "oxygen", "nitrogen", "hydrogen", "sodium",
    "atoms", "cells", "photosynthesis", "diamond", "gravity", "electricity",
})

_NUMBER_WORD = re.compile(
    r"^(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|billion)(?:-[a-z]+)?$"
)

# Yes/no, why, and how-mechanism questions make "The answer to ... is
# <term>." read wrong. The auxiliary family was half-covered (was/did/will/
# have/should/could missing) and produced 'The answer to "Was Enigma built
# from another model?" is None.' -- an ambiguous-polarity SELF fact (review
# 2026-08-13).
_CLOZE_Q_SKIP = ("is ", "are ", "do ", "does ", "can ", "why ", "name ",
                 "was ", "were ", "did ", "will ", "has ", "have ",
                 "should ", "could ", "would ", "how do ", "how does ",
                 "who can ")

# Subjects that are stand-ins, not terms -- never invert around them.
_INVERT_X_STOP = frozenset({"it", "that", "this", "there", "he", "she"})

# Fact-in-context leads: colon-terminated so ANY curated answer (including
# two-sentence ones) follows grammatically as a single natural line.
_CONTEXT_LEADS = (
    "Here is a fact worth keeping straight: ",
    "One piece of common knowledge: ",
    "A fact that comes up all the time: ",
    "Worth knowing, and easy to remember: ",
    "As any reference book will confirm: ",
    "File this one under general knowledge: ",
)


def _probe_strings() -> list[str]:
    """Lowercased eval-probe questions (and memory teach lines). Generated
    text must never carry one verbatim -- same dodge discipline as the
    KNOWLEDGE phrasings themselves and make_sft_data's _norm_q backstop.

    FAIL CLOSED: a missing probes file used to silently disable the screen
    (changing the facts stream's bytes with no error), and one empty q/teach
    string would have matched EVERY line and zeroed the stream, surfacing
    downstream as make_facts' misleading "produced no fact lines" (review
    2026-08-13)."""
    if not _EVAL_PROBES.exists():
        raise SystemExit(
            f"REFUSED: {_EVAL_PROBES} is missing -- generating the facts "
            f"stream without the probe screen would silently change its bytes"
        )
    probes: list[str] = []
    for line in _EVAL_PROBES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for s in [rec.get("q", "")] + list(rec.get("teach") or []):
            s = str(s).strip().lower()
            if s:  # an empty string is a match-everything poison, never a probe
                probes.append(s)
    return probes


def _first_segment(text: str) -> str:
    """Text up to the first sentence-level break (commas do not count)."""
    cut = len(text)
    for sep in _SENTENCE_SEPS:
        i = text.find(sep)
        if i != -1 and i < cut:
            cut = i
    return text[:cut].rstrip(".")


def _lead_fragment(text: str) -> tuple[str, str]:
    """Leading fragment before the earliest separator, plus that separator."""
    cut, used = len(text), ""
    for sep in _FRAGMENT_SEPS:
        i = text.find(sep)
        if i != -1 and i < cut:
            cut, used = i, sep
    return text[:cut].rstrip("."), used


def _key_term(answers: list[str]) -> str:
    """The intent's answer TERM ("Jupiter", "twenty-four hours"), extracted
    from the leading fragment of the first answer that yields a clean one.
    Returns "" when every answer opens with a clause -- those intents simply
    skip the cloze form (the other forms still cover them)."""
    for a in answers:
        text = a
        for pre in _KEY_PREFIXES:
            if text.startswith(pre):
                text = text[len(pre):]
                break
        frag, sep = _lead_fragment(text)
        if sep == ", " and " and " in _first_segment(text):
            continue  # enumeration ("Spring, summer, ... and winter") -- frag is a partial list
        words = frag.split()
        if not 1 <= len(words) <= 4:
            continue
        if any(w.strip(".,").lower() in _KEY_STOP_TOKENS for w in words):
            continue
        return frag
    return ""


def _mid_case(term: str) -> str:
    """Case a key term for mid-sentence use: common first words and
    number-words lowercase, proper nouns keep their capital.

    A number-word leading a TITLE ("Ten Commandments", "One Direction" --
    every following word capitalized) is a name, not a quantity: folding it
    destroys a proper noun, the opposite error of the cosmetic capital the
    fold fixes (audit 2026-08-14). Mixed-case tails ("Zero degrees Celsius")
    are quantities carrying a unit's proper noun and still fold."""
    words = term.split()
    first = words[0].lower()
    if len(words) > 1 and all(w[:1].isupper() for w in words[1:]):
        return term
    if first in _MID_LOWER_FIRST or _NUMBER_WORD.match(first):
        return term[0].lower() + term[1:]
    return term


def _cloze_question(questions: list[str]) -> str:
    """First question that reads naturally inside 'The answer to "..." is'."""
    for q in questions:
        if not q.lower().startswith(_CLOZE_Q_SKIP):
            return q
    return ""


def gen_knowledge_pretrain_text(seed: int = 77,
                                content: "PersonaContent | None" = None) -> list[str]:
    """Emit plain-text lines (NOT chat records) for continued pretraining.

    Per intent: declarative statements (the curated answers as standalone
    prose), one Q/A line per question phrasing, key-term-final cloze(s), and
    fact-in-context lines. Deduped, probe-dodged by substring, shuffled with
    the seed. Deterministic given the seed and the probes file."""
    rng = random.Random(seed)
    probes = _probe_strings()
    lines: list[str] = []
    for idx, (questions, answers) in enumerate(_table(content)):
        # Declarative prose: the answers verbatim (one-word answers like
        # "Rome." carry no signal alone -- their full-sentence twin runs).
        for a in answers:
            if len(a.split()) >= 4:
                lines.append(a)
        # Q/A line per question phrasing, answers rotating.
        for i, q in enumerate(questions):
            lines.append(f"Q: {q} A: {answers[i % len(answers)]}")
        # Cloze, generic: quote a question, land the key term LAST.
        key = _key_term(answers)
        q = _cloze_question(questions)
        if key and q:
            # Two degenerate shapes the skip-list widening CREATED by
            # shifting clozes onto later questions (audit 2026-08-14):
            # a tautology whose term already sits in the question ('The
            # answer to "What is photosynthesis?" is photosynthesis.'),
            # and a connector-led term that breaks the copula ('... is
            # Through its memory store.'). Both skip -- the other forms
            # still cover the intent.
            q_words = {w.strip('.,?!"\'').lower() for w in q.split()}
            key_words = {w.strip('.,?!"\'').lower() for w in key.split()}
            # A which-question OFFERS its options, so an answer quoted from
            # them is a real pick ("Which is longer, a mile or a kilometer?"
            # is a mile.), not a tautology.
            tautology = key_words <= q_words and not q.lower().startswith("which")
            connector_led = key.split()[0].lower() in {
                "through", "with", "by", "via", "using", "because", "from"}
            if not tautology and not connector_led:
                lines.append(f'The answer to "{q}" is {_mid_case(key)}.')
        # Cloze, inverted: "Mercury is the smallest planet ..." becomes
        # "The smallest planet ... is Mercury." -- key term LAST, gated hard
        # so only clean single-subject "X is Y" sentences flip.
        for a in answers:
            seg = _first_segment(a)
            if seg.count(" is ") != 1:
                continue
            x, y = seg.split(" is ")
            if (len(x.split()) == 1 and x.lower() not in _INVERT_X_STOP
                    and len(y.split()) >= 3 and "," not in y
                    and y.lower().startswith(("the ", "a ", "an "))):
                lines.append(f"{y[0].upper()}{y[1:]} is {_mid_case(x)}.")
        # Fact-in-context: the fact inside a longer natural line. Same bar
        # as the declarative form: a sub-4-word payload after an
        # authoritative lead is a confident non-answer ("As any reference
        # book will confirm: 1945." -- 47 such lines measured against this
        # corpus, review 2026-08-13/audit 2026-08-14).
        for j in range(2):
            ctx = answers[(idx + j) % len(answers)]
            if len(ctx.split()) < 4:
                continue
            lead = _CONTEXT_LEADS[(idx + j) % len(_CONTEXT_LEADS)]
            lines.append(lead + ctx)
    seen: set[str] = set()
    uniq: list[str] = []
    for line in lines:
        if line in seen:
            continue
        low = line.lower()
        if any(p in low for p in probes):
            continue
        seen.add(line)
        uniq.append(line)
    rng.shuffle(uniq)
    return uniq


if __name__ == "__main__":
    ex = gen_knowledge_examples()
    print(f"{len(ex)} knowledge records from {len(KNOWLEDGE)} facts")
    for r in ex[:5]:
        print("Q:", r["messages"][0]["content"])
        print("A:", r["messages"][1]["content"])
    txt = gen_knowledge_pretrain_text()
    print(f"{len(txt)} pretrain text lines from {len(KNOWLEDGE)} facts")
    for line in txt[:5]:
        print(" ", line)
