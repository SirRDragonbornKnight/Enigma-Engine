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
  better at 182M. Digits appear only where words would read strangely.
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
from pathlib import Path

# (question phrasings, answer variants). Answers vary in wording, never in fact.
KNOWLEDGE: list[tuple[list[str], list[str]]] = [
    # ---------------------------------------------------------------- space
    (
        ["Which planet is the biggest one?", "What is the largest planet going around our sun?", "Of all the planets, which is largest?"],
        ["Jupiter. It is by far the largest planet in our solar system.",
         "Jupiter is the biggest planet -- more massive than all the other planets combined."],
    ),
    (
        ["Which planet is smallest?", "What's the smallest planet in the solar system?"],
        ["Mercury is the smallest planet in our solar system.",
         "That's Mercury -- the smallest planet, and also the one closest to the sun."],
    ),
    (
        ["Which planet is closest to the sun?", "What planet orbits nearest the sun?"],
        ["Mercury is the closest planet to the sun.",
         "Mercury -- it sits nearest the sun of all the planets."],
    ),
    (
        ["Why is Mars called the red planet?", "Which planet is known as the red planet?"],
        ["Mars. Its surface is covered in iron-rich dust that looks red.",
         "Mars is the red planet -- rusty iron dust on its surface gives it that color."],
    ),
    (
        ["Which planet has the famous rings?", "What planet is known for its rings?"],
        ["Saturn. Its rings are made of ice and rock circling the planet.",
         "Saturn is the ringed planet -- bright rings of ice and rock."],
    ),
    (
        ["Which planet is our home?", "What planet are we standing on right now?", "Where do humans live in the solar system?"],
        ["Earth. It is the only planet known to support life.",
         "We live on Earth -- the third planet from the sun."],
    ),
    (
        ["Which planet is the hottest?", "What's the hottest planet in the solar system?"],
        ["Venus. Its thick atmosphere traps heat, making it hotter than Mercury.",
         "Venus is the hottest planet -- its heavy atmosphere holds the heat in."],
    ),
    (
        ["How many planets are in the solar system?", "How many planets orbit our sun?"],
        ["Eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
         "There are eight -- Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."],
    ),
    (
        ["Is Pluto a planet?", "What happened to Pluto's planet status?"],
        ["Pluto is classified as a dwarf planet, not one of the eight planets.",
         "It's a dwarf planet -- Pluto was reclassified, so the solar system counts eight planets."],
    ),
    (
        ["What does Earth orbit?", "What does our planet go around?"],
        ["Earth orbits the sun, taking one year to go around it.",
         "The sun. One full trip around it takes a year."],
    ),
    (
        ["What goes around the Earth?", "What is Earth's natural satellite?"],
        ["The moon orbits the Earth, taking about a month for each trip.",
         "That's the moon -- Earth's only natural satellite."],
    ),
    (
        ["Is the sun a star or a planet?", "What kind of object is the sun?"],
        ["The sun is a star -- a huge ball of hot gas that gives us light and heat.",
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
         "The Milky Way -- our whole solar system is inside it."],
    ),
    (
        ["Why do we have day and night?", "What causes day and night?"],
        ["Earth spins. The side facing the sun has day, and the side facing away has night.",
         "Day and night come from Earth rotating -- each full spin takes about twenty-four hours."],
    ),
    (
        ["What causes the tides?", "Why does the sea rise and fall?"],
        ["The moon's gravity pulls on the oceans, making the tides rise and fall.",
         "Mostly the moon -- its gravity tugs the oceans as Earth turns."],
    ),
    (
        ["Who was the first person to walk on the moon?", "Who first set foot on the moon?"],
        ["Neil Armstrong, in 1969, during the Apollo 11 mission.",
         "Neil Armstrong -- he stepped onto the moon in 1969 with Apollo 11."],
    ),
    (
        ["What is the North Star called?", "Which star points north?"],
        ["Polaris. It sits almost directly above the North Pole, so it points north.",
         "That's Polaris, the North Star -- travelers have used it to find north for centuries."],
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
        ["365 days, and 366 in a leap year.",
         "A year lasts 365 days -- leap years add one more."],
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
         "February is the shortest -- twenty-eight days most years."],
    ),
    (
        ["How often do leap years happen?", "When do we get a leap year?"],
        ["Every four years. February gains an extra day, the twenty-ninth.",
         "Once every four years -- that's when February gets a twenty-ninth day."],
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
         "There are seven continents -- Asia, Africa, North America, South America, Antarctica, Europe, and Australia."],
    ),
    (
        ["Which continent is the largest?", "What's the biggest continent?"],
        ["Asia. It is the largest continent in both land and population.",
         "Asia is the biggest continent by far."],
    ),
    (
        ["Which continent is the coldest?", "What's the coldest place on Earth, continent-wise?"],
        ["Antarctica. It is the coldest, windiest continent, covered in ice.",
         "Antarctica -- an ice-covered continent at the South Pole."],
    ),
    (
        ["How many oceans are there?", "Name the oceans of the world."],
        ["Five: the Pacific, Atlantic, Indian, Arctic, and Southern oceans.",
         "There are five oceans -- Pacific, Atlantic, Indian, Arctic, and Southern."],
    ),
    (
        ["Which ocean is the biggest?", "What's the largest ocean on Earth?"],
        ["The Pacific. It covers more area than all the land on Earth combined.",
         "The Pacific Ocean is the largest and deepest."],
    ),
    (
        ["What's the longest river in the world?", "Which river is longest?"],
        ["The Nile is usually called the longest river; the Amazon carries the most water.",
         "Most sources say the Nile, in Africa -- though the Amazon is the biggest by water volume."],
    ),
    (
        ["What's the tallest mountain on Earth?", "Which mountain is the highest?"],
        ["Mount Everest, in the Himalayas, is the highest mountain above sea level.",
         "Mount Everest -- the peak of the world, on the border of Nepal and China."],
    ),
    (
        ["What's the largest hot desert?", "Which desert is the biggest hot one?"],
        ["The Sahara, in northern Africa, is the largest hot desert.",
         "That's the Sahara -- it stretches across much of northern Africa."],
    ),
    (
        ["What's the biggest country in the world?", "Which country has the most land?"],
        ["Russia. It spans Europe and Asia and is the largest country by area.",
         "Russia is the largest country by land area."],
    ),
    (
        ["Which country has the most people?", "What's the most populated country?"],
        ["India has the most people, with China a close second.",
         "India -- it recently passed China as the most populous country."],
    ),
    (
        ["What's the capital of the United States?", "Which city is the US capital?"],
        ["Washington, D.C.",
         "The capital of the United States is Washington, D.C."],
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
        ["Ottawa -- not Toronto, which is just the biggest city.",
         "Canada's capital is Ottawa."],
    ),
    (
        ["What's the capital of Australia?", "Which city is Australia's capital?"],
        ["Canberra -- not Sydney, which is just the biggest city.",
         "Australia's capital is Canberra."],
    ),
    (
        ["What's the capital of Brazil?", "Which city is Brazil's capital?"],
        ["Brasilia -- a city built specifically to be the capital.",
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
         "It's the line circling the center of the Earth -- the hottest zone sits along it."],
    ),
    (
        ["What's the largest rainforest?", "Which rainforest is the biggest?"],
        ["The Amazon rainforest, in South America.",
         "The Amazon -- the biggest rainforest on Earth, mostly in Brazil."],
    ),
    (
        ["What language do people speak in Brazil?", "What do Brazilians mostly speak?"],
        ["Portuguese. Brazil is the largest Portuguese-speaking country in the world.",
         "Brazilians speak Portuguese -- not Spanish, which surprises many people."],
    ),
    (
        ["What language is spoken in Mexico?", "What do people in Mexico speak?"],
        ["Spanish.",
         "Mexico's main language is Spanish."],
    ),
    (
        ["What language has the most native speakers?", "Which language do the most people speak natively?"],
        ["Mandarin Chinese has the most native speakers; English is the most widely learned.",
         "Mandarin Chinese, by native speakers -- English leads once you count second-language speakers."],
    ),
    # -------------------------------------------------------------- science
    (
        ["What is water made of?", "What are the elements in water?"],
        ["Hydrogen and oxygen -- two hydrogen atoms and one oxygen atom, H2O.",
         "Water is H2O: two parts hydrogen, one part oxygen."],
    ),
    (
        ["At what temperature does water freeze?", "When does water turn to ice?"],
        ["Zero degrees Celsius, which is thirty-two degrees Fahrenheit.",
         "Water freezes at zero Celsius -- thirty-two Fahrenheit."],
    ),
    (
        ["At what temperature does water boil?", "When does water start boiling?"],
        ["One hundred degrees Celsius at sea level -- that's 212 Fahrenheit.",
         "Water boils at one hundred Celsius at sea level."],
    ),
    (
        ["What are the states of matter?", "Name the main states of matter."],
        ["Solid, liquid, and gas -- plus plasma, like the inside of stars.",
         "The three everyday ones are solid, liquid, and gas. Plasma is the fourth."],
    ),
    (
        ["What gas do we breathe in to live?", "What gas do humans need from the air?"],
        ["Oxygen. We breathe in oxygen and breathe out carbon dioxide.",
         "Oxygen keeps us alive -- we exhale carbon dioxide in return."],
    ),
    (
        ["What is most of the air made of?", "What gas makes up most of the atmosphere?"],
        ["Nitrogen -- about four-fifths of the air. Oxygen is most of the rest.",
         "Mostly nitrogen, roughly seventy-eight percent. Oxygen is about twenty-one percent."],
    ),
    (
        ["How do plants make their food?", "What is photosynthesis?"],
        ["Photosynthesis: plants use sunlight to turn water and carbon dioxide into food, releasing oxygen.",
         "Plants capture sunlight and use it to make food from water and carbon dioxide -- the oxygen we breathe is their exhaust."],
    ),
    (
        ["What is gravity?", "Why do things fall down?"],
        ["Gravity is the force that pulls objects toward each other -- it keeps us on the ground and the planets around the sun.",
         "Things fall because Earth's gravity pulls them toward its center."],
    ),
    (
        ["Which is faster, light or sound?", "Why do we see lightning before hearing thunder?"],
        ["Light is far faster than sound. That's why lightning flashes first and thunder arrives after.",
         "Light wins by a huge margin -- the lightning reaches your eyes almost instantly, the thunder lags behind."],
    ),
    (
        ["How fast does light travel?", "What's the speed of light?"],
        ["About three hundred thousand kilometers per second -- nothing moves faster.",
         "Roughly 300,000 kilometers every second. It's the universe's speed limit."],
    ),
    (
        ["How many poles does a magnet have?", "What are the poles of a magnet?"],
        ["Two: a north pole and a south pole. Opposite poles attract; like poles repel.",
         "Every magnet has a north and a south pole -- opposites pull together, matching poles push apart."],
    ),
    (
        ["What is electricity?", "What flows through a wire?"],
        ["Electricity is the flow of electrons through a material like copper wire.",
         "Moving electrons -- that flow through wires is what we call electricity."],
    ),
    (
        ["What's the hardest natural material?", "Which natural substance is hardest?"],
        ["Diamond. It's carbon arranged in an extremely strong crystal.",
         "Diamond is the hardest natural material -- pure carbon under enormous pressure."],
    ),
    (
        ["What's the chemical symbol for gold?", "What letters stand for gold in chemistry?"],
        ["Au, from the Latin word aurum.",
         "Gold's symbol is Au -- Latin for aurum."],
    ),
    (
        ["What is table salt made of?", "What's the chemistry of salt?"],
        ["Sodium and chlorine -- sodium chloride, NaCl.",
         "Salt is sodium chloride: one sodium atom bonded to one chlorine atom."],
    ),
    (
        ["Why does ice float on water?", "How can ice float?"],
        ["Ice is less dense than liquid water, so it floats.",
         "Because freezing makes water expand -- ice is lighter for its size, so it rides on top."],
    ),
    (
        ["Where does the sun get its energy?", "How does the sun make light?"],
        ["Nuclear fusion. The sun fuses hydrogen into helium, releasing enormous energy.",
         "The sun runs on fusion -- hydrogen atoms merging into helium deep in its core."],
    ),
    (
        ["What makes a rainbow?", "How do rainbows form?"],
        ["Sunlight bending through water droplets splits into its colors -- that's a rainbow.",
         "Rainbows appear when sunlight passes through raindrops and spreads into colors."],
    ),
    (
        ["What is everything made of?", "What are the building blocks of matter?"],
        ["Atoms. Everything you can touch is made of atoms bonded together.",
         "Matter is made of atoms -- tiny particles made of protons, neutrons, and electrons."],
    ),
    (
        ["What are the parts of an atom?", "What's inside an atom?"],
        ["Protons and neutrons in the center, with electrons around them.",
         "An atom has a nucleus of protons and neutrons, with electrons orbiting it."],
    ),
    (
        ["What does DNA do?", "What is DNA for?"],
        ["DNA carries the genetic instructions for building and running a living thing.",
         "It's the instruction manual of life -- DNA stores the code your cells follow."],
    ),
    (
        ["What are living things made of?", "What is the basic unit of life?"],
        ["Cells. Every living thing is made of one or more cells.",
         "The cell -- the smallest building block of every living organism."],
    ),
    (
        ["What comes out of a volcano?", "What is lava?"],
        ["Molten rock. Underground it's called magma; once it erupts, it's lava.",
         "Lava -- melted rock from deep inside the Earth."],
    ),
    (
        ["What causes earthquakes?", "Why does the ground shake in an earthquake?"],
        ["Huge plates of Earth's crust shifting and slipping against each other.",
         "Earthquakes happen when the plates of the Earth's crust suddenly move."],
    ),
    (
        ["How does rain happen?", "What is the water cycle?"],
        ["Water evaporates, rises, condenses into clouds, and falls back as rain -- the water cycle.",
         "The sun lifts water into the sky as vapor, it forms clouds, and it falls again as rain."],
    ),
    # ------------------------------------------------------- body / biology
    (
        ["What does the heart do?", "What's the heart's job?"],
        ["The heart pumps blood around your body, delivering oxygen and nutrients.",
         "It's a pump -- your heart pushes blood through your whole body, beat after beat."],
    ),
    (
        ["What do lungs do?", "What's the job of your lungs?"],
        ["Lungs take in oxygen when you breathe in and push out carbon dioxide when you breathe out.",
         "They handle breathing -- oxygen in, carbon dioxide out."],
    ),
    (
        ["What does the brain do?", "What's the brain's job?"],
        ["The brain controls everything -- thought, memory, movement, and the senses.",
         "It runs the whole show: thinking, feeling, remembering, and controlling your body."],
    ),
    (
        ["How many bones does an adult have?", "What's the number of bones in the human body?"],
        ["An adult has 206 bones. Babies are born with more, and some fuse as they grow.",
         "206 in an adult -- children start with around 300 that fuse over time."],
    ),
    (
        ["What's the largest organ of the body?", "Which human organ is the biggest?"],
        ["The skin. It covers the entire body and protects everything inside.",
         "Your skin -- it's the body's largest organ."],
    ),
    (
        ["What are the five senses?", "Name the human senses."],
        ["Sight, hearing, smell, taste, and touch.",
         "The five senses are sight, hearing, taste, smell, and touch."],
    ),
    (
        ["What do kidneys do?", "What's the job of the kidneys?"],
        ["Kidneys filter waste out of your blood and turn it into urine.",
         "They're the blood's filters -- kidneys clean waste from the blood."],
    ),
    (
        ["Are humans mammals?", "What kind of animal is a human?"],
        ["Yes -- humans are mammals: warm-blooded, with hair, and fed on milk as babies.",
         "Humans are mammals, like whales, dogs, and elephants."],
    ),
    (
        ["What makes an animal a mammal?", "What defines mammals?"],
        ["Mammals are warm-blooded, have hair or fur, and feed their young with milk.",
         "Warm blood, fur or hair, and milk for their babies -- that's what makes a mammal."],
    ),
    (
        ["How do fish breathe?", "How do fish get oxygen underwater?"],
        ["With gills. Gills pull oxygen straight out of the water.",
         "Fish breathe through gills, which take oxygen from water."],
    ),
    (
        ["How many legs does an insect have?", "What's the leg count for insects?"],
        ["Six legs. If it has eight, it's not an insect -- it's probably a spider.",
         "Insects have six legs, always."],
    ),
    (
        ["How many legs does a spider have?", "What's the leg count for spiders?"],
        ["Eight legs. That's one way to tell spiders from insects, which have six.",
         "Spiders walk on eight legs."],
    ),
    (
        ["What's the largest animal ever?", "Which animal is the biggest on Earth?"],
        ["The blue whale -- the largest animal that has ever lived, bigger than any dinosaur.",
         "The blue whale. Nothing alive, or ever alive, has been bigger."],
    ),
    (
        ["What's the tallest animal?", "Which animal stands the tallest?"],
        ["The giraffe. Its long neck lets it reach leaves other animals can't.",
         "Giraffes -- the tallest animals on land."],
    ),
    (
        ["What's the fastest land animal?", "Which animal runs the fastest?"],
        ["The cheetah. In a short sprint, nothing on land catches it.",
         "The cheetah holds the land speed record among animals."],
    ),
    (
        ["What do bees make?", "What insect makes honey?"],
        ["Bees make honey from flower nectar -- and they pollinate plants while gathering it.",
         "Honey comes from bees, which turn nectar into it inside the hive."],
    ),
    (
        ["What does a caterpillar turn into?", "What comes out of a chrysalis?"],
        ["A butterfly. The caterpillar transforms inside a chrysalis.",
         "A caterpillar becomes a butterfly -- or a moth, depending on the species."],
    ),
    (
        ["What does a tadpole become?", "What grows from a tadpole?"],
        ["A frog. Tadpoles live in water and slowly grow legs and lungs.",
         "Tadpoles turn into frogs."],
    ),
    (
        ["Can penguins fly?", "Do penguins fly like other birds?"],
        ["No -- penguins are birds, but their wings are built for swimming, not flying.",
         "They can't fly. Penguins use their wings as flippers to swim instead."],
    ),
    (
        ["Are bats birds?", "What kind of animal is a bat?"],
        ["Bats are mammals -- the only mammals that truly fly.",
         "Not birds. Bats are flying mammals: fur, live young, milk."],
    ),
    (
        ["Are whales fish?", "What kind of animal is a whale?"],
        ["Whales are mammals, not fish. They breathe air and feed their calves milk.",
         "Not fish -- whales are mammals that must surface to breathe."],
    ),
    (
        ["How many arms does an octopus have?", "What's the arm count for an octopus?"],
        ["Eight arms.",
         "An octopus has eight arms."],
    ),
    (
        ["What's in a camel's hump?", "Do camels store water in their humps?"],
        ["Fat, not water. The stored fat lets camels go a long time without food.",
         "Humps hold fat -- it's an energy reserve, not a water tank."],
    ),
    (
        ["When did the dinosaurs die out?", "How long ago did dinosaurs go extinct?"],
        ["About sixty-six million years ago, most likely after a giant asteroid impact.",
         "Roughly sixty-six million years ago -- an asteroid strike is the leading explanation."],
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
         "They're temperature scales -- water freezes at zero Celsius or thirty-two Fahrenheit."],
    ),
    (
        ["What does a liter measure?", "What kind of unit is a liter?"],
        ["Volume -- how much space a liquid takes up.",
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
        ["A, E, I, O, and U -- and sometimes Y.",
         "The vowels are A, E, I, O, U, with Y sometimes counting too."],
    ),
    (
        ["What are the primary colors of paint?", "Which colors mix to make the others in paint?"],
        ["Red, yellow, and blue -- mixing them makes the other colors.",
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
        ["Eighty-eight keys -- fifty-two white and thirty-six black.",
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
        ["Every four years -- with summer and winter games alternating every two.",
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
        ["1969 -- Apollo 11, with Neil Armstrong first on the surface.",
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
        ["Isaac Newton -- he worked out the laws of gravity and motion.",
         "That story belongs to Isaac Newton, who described how gravity works."],
    ),
    (
        ["Who proposed the theory of evolution?", "Which scientist wrote about natural selection?"],
        ["Charles Darwin, in On the Origin of Species.",
         "Charles Darwin -- his theory of evolution by natural selection."],
    ),
    (
        ["Who invented the printing press?", "Which inventor made printing books practical?"],
        ["Johannes Gutenberg, in the fifteenth century.",
         "Gutenberg -- his printing press made books widely available."],
    ),
]


def gen_knowledge_examples(seed: int = 55) -> list[dict]:
    """Emit clean fact-QA records (Q x two rotating answers per intent),
    deduped on the exact (question, answer) pair -- the identity_paraphrases
    rendering pattern applied to world knowledge."""
    rng = random.Random(seed)
    out: list[dict] = []
    for questions, answers in KNOWLEDGE:
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
})

# Answer openers that wrap the key term -- strip before extracting it.
_KEY_PREFIXES = ("That's ", "It's ", "We live on ")

# Sentence-level breaks (never a comma) and fragment-level breaks.
_SENTENCE_SEPS = (" -- ", ". ", "; ", ": ")
_FRAGMENT_SEPS = (" -- ", ": ", ". ", "; ", ", ")

# First words safe to lowercase when the key term lands mid-sentence
# ("...is twenty-four hours."); everything else is treated as a proper noun.
_MID_LOWER_FIRST = frozenset({
    "a", "an", "the", "about", "roughly", "mostly", "every", "once", "with",
    "zero", "one", "two", "five", "six", "seven", "eight", "ten", "twelve",
    "fifty", "sixty", "twenty-four", "twenty-six", "sixty-four",
    "eighty-eight", "moving", "molten", "nuclear", "fat", "green", "pink",
    "temperature", "volume", "oxygen", "nitrogen", "hydrogen", "sodium",
    "atoms", "cells", "photosynthesis", "diamond", "gravity", "electricity",
})

# Yes/no and why questions make "The answer to ... is <term>." read wrong.
_CLOZE_Q_SKIP = ("is ", "are ", "do ", "does ", "can ", "why ", "name ")

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
    KNOWLEDGE phrasings themselves and make_sft_data's _norm_q backstop."""
    if not _EVAL_PROBES.exists():
        return []
    probes: list[str] = []
    for line in _EVAL_PROBES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        probes.append(rec["q"].strip().lower())
        for fact in rec.get("teach", []):
            probes.append(fact.strip().lower())
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
    """Case a key term for mid-sentence use: common first words lowercase,
    proper nouns keep their capital."""
    if term.split()[0].lower() in _MID_LOWER_FIRST:
        return term[0].lower() + term[1:]
    return term


def _cloze_question(questions: list[str]) -> str:
    """First question that reads naturally inside 'The answer to "..." is'."""
    for q in questions:
        if not q.lower().startswith(_CLOZE_Q_SKIP):
            return q
    return ""


def gen_knowledge_pretrain_text(seed: int = 77) -> list[str]:
    """Emit plain-text lines (NOT chat records) for continued pretraining.

    Per intent: declarative statements (the curated answers as standalone
    prose), one Q/A line per question phrasing, key-term-final cloze(s), and
    fact-in-context lines. Deduped, probe-dodged by substring, shuffled with
    the seed. Deterministic given the seed and the probes file."""
    rng = random.Random(seed)
    probes = _probe_strings()
    lines: list[str] = []
    for idx, (questions, answers) in enumerate(KNOWLEDGE):
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
        # Fact-in-context: the fact inside a longer natural line.
        for j in range(2):
            lead = _CONTEXT_LEADS[(idx + j) % len(_CONTEXT_LEADS)]
            lines.append(lead + answers[(idx + j) % len(answers)])
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
