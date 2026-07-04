#!/usr/bin/env python3
"""Generate smoke-test data files for all FORGE training modes.

Run once:  python create_smoke_test_data.py

Creates:
  data/pretrain/smoke_test.txt  (~500 KB) - Pre-Train mode
  data/smoke_test_basic.txt     (~50 KB)  - Basic / Solo mode
  data/smoke_test_dpo.jsonl     (~20 KB)  - DPO + RLHF modes
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PRETRAIN_DIR = DATA_DIR / "pretrain"
TARGET_PRETRAIN_KB = 500
TARGET_BASIC_KB = 50

# Seed for reproducibility
random.seed(42)

# ---------------------------------------------------------------------------
# Topic paragraphs — diverse educational content
# ---------------------------------------------------------------------------

SCIENCE_PARAGRAPHS = [
    "The cell is the fundamental unit of life. All living organisms are composed of one or more cells, each containing genetic material that directs cellular activities. Prokaryotic cells lack a membrane-bound nucleus, while eukaryotic cells contain organelles such as mitochondria, the endoplasmic reticulum, and the Golgi apparatus. Cell division occurs through mitosis for growth and repair, and meiosis for producing gametes with half the chromosome count.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules. In the chloroplasts of plant cells, chlorophyll absorbs photons primarily in the red and blue wavelengths, reflecting green light. The light-dependent reactions split water molecules, releasing oxygen as a byproduct and generating ATP and NADPH. The Calvin cycle then uses these energy carriers to fix carbon dioxide into three-carbon sugar molecules.",
    "Newton's three laws of motion form the foundation of classical mechanics. The first law states that an object remains at rest or in uniform motion unless acted upon by a net external force. The second law relates force, mass, and acceleration through the equation F equals ma. The third law asserts that for every action there is an equal and opposite reaction, explaining phenomena from rocket propulsion to walking.",
    "The periodic table organizes chemical elements by atomic number and electron configuration. Elements in the same group share similar chemical properties because they have the same number of valence electrons. The alkali metals in Group 1 are highly reactive, readily losing one electron to form positive ions. Noble gases in Group 18 have complete electron shells and are generally unreactive under standard conditions.",
    "DNA replication is a semiconservative process where each strand of the double helix serves as a template for a new complementary strand. Helicase unwinds the double helix at the replication fork, while DNA polymerase adds nucleotides in the five-prime to three-prime direction. The leading strand is synthesized continuously, but the lagging strand requires Okazaki fragments that are later joined by DNA ligase.",
    "Thermodynamics governs the transfer of energy in physical and chemical processes. The first law states that energy cannot be created or destroyed, only transformed from one form to another. The second law introduces entropy, asserting that the total entropy of an isolated system always increases over time. Heat naturally flows from hot objects to cold objects, and work must be done to reverse this direction.",
    "The theory of evolution by natural selection, proposed by Charles Darwin, explains the diversity of life on Earth. Organisms with traits better suited to their environment are more likely to survive and reproduce, passing those advantageous traits to their offspring. Over many generations, this process leads to adaptation and speciation. Genetic mutations provide the raw material for natural selection to act upon.",
    "Electromagnetic waves are oscillating electric and magnetic fields that propagate through space at the speed of light. The electromagnetic spectrum ranges from low-frequency radio waves through microwaves, infrared, visible light, ultraviolet, X-rays, to high-frequency gamma rays. Each type of radiation has different applications: radio waves for communication, infrared for thermal imaging, and X-rays for medical diagnostics.",
    "Plate tectonics describes the movement of large sections of Earth's lithosphere over the underlying asthenosphere. Divergent boundaries occur where plates move apart, creating new oceanic crust at mid-ocean ridges. Convergent boundaries involve plates colliding, often resulting in subduction zones, mountain building, or volcanic activity. Transform boundaries are where plates slide past each other horizontally, causing earthquakes.",
    "The water cycle describes the continuous movement of water through Earth's systems. Evaporation transforms liquid water from oceans, lakes, and rivers into water vapor in the atmosphere. Condensation forms clouds when water vapor cools and collects around tiny particles. Precipitation returns water to the surface as rain, snow, sleet, or hail, where it flows through rivers and groundwater back to the oceans.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales. Unlike classical physics, quantum theory predicts that particles can exhibit wave-like behavior, demonstrated by the double-slit experiment. The Heisenberg uncertainty principle states that one cannot simultaneously know both the exact position and momentum of a particle. Quantum entanglement allows particles to be correlated regardless of distance.",
    "The immune system protects organisms from pathogens through innate and adaptive responses. Innate immunity provides immediate, non-specific defense through physical barriers like skin, and cells like macrophages and neutrophils. Adaptive immunity develops specific responses through B cells producing antibodies and T cells attacking infected cells. Memory cells from prior infections enable faster responses to subsequent exposures.",
    "Chemical bonding determines how atoms combine to form molecules and compounds. Ionic bonds form when electrons are transferred from one atom to another, creating oppositely charged ions that attract each other. Covalent bonds involve the sharing of electron pairs between atoms. Metallic bonds occur in metals, where electrons are delocalized across a lattice of positive ions, giving metals their characteristic conductivity.",
    "The nervous system coordinates bodily functions through electrical and chemical signals. Neurons transmit information via action potentials, which are rapid changes in electrical potential across the cell membrane. At synapses, neurotransmitters carry signals between neurons. The central nervous system, comprising the brain and spinal cord, processes information and controls voluntary and involuntary responses throughout the body.",
    "Ecosystems consist of biological communities interacting with their physical environment. Producers like plants and algae form the base of food webs, converting solar energy into biomass. Primary consumers feed on producers, while secondary and tertiary consumers occupy higher trophic levels. Decomposers break down dead organic matter, recycling nutrients back into the soil for producers to use again.",
]

MATH_PARAGRAPHS = [
    "Algebra provides tools for solving equations and modeling relationships between variables. A linear equation in one variable takes the form ax plus b equals zero, where a and b are constants. Quadratic equations involve terms up to the second power and can be solved using the quadratic formula, factoring, or completing the square. Systems of equations can be solved by substitution, elimination, or matrix methods.",
    "Calculus studies rates of change and accumulation. The derivative of a function measures its instantaneous rate of change at any point, while the integral computes the total accumulation over an interval. The Fundamental Theorem of Calculus connects these two operations: the definite integral of a function can be evaluated using its antiderivative. Applications range from optimization problems to modeling physical systems.",
    "Probability theory quantifies uncertainty in random phenomena. The probability of an event ranges from zero, meaning impossible, to one, meaning certain. Conditional probability measures the likelihood of an event given that another event has occurred. Bayes' theorem provides a way to update probability estimates as new evidence becomes available, forming the basis of many machine learning algorithms.",
    "Geometry studies the properties of shapes, sizes, and spatial relationships. Euclidean geometry deals with flat surfaces, where the angles of a triangle always sum to 180 degrees. Non-Euclidean geometries, such as hyperbolic and spherical geometry, relax the parallel postulate and arise naturally on curved surfaces. Trigonometry extends geometric ideas using the sine, cosine, and tangent functions to relate angles to side lengths.",
    "Statistics provides methods for collecting, analyzing, and interpreting data. Descriptive statistics summarize data using measures like mean, median, mode, and standard deviation. Inferential statistics use samples to make predictions about populations, relying on probability distributions and hypothesis testing. The central limit theorem states that sample means approximate a normal distribution regardless of the population distribution.",
    "Linear algebra studies vector spaces and linear transformations. Vectors represent quantities with both magnitude and direction, and can be added, scaled, and combined. Matrices encode linear transformations and systems of linear equations. Eigenvalues and eigenvectors reveal the fundamental geometric behavior of transformations, such as the directions along which scaling occurs without rotation.",
    "Number theory explores the properties of integers and their relationships. Prime numbers are integers greater than one that have no divisors other than one and themselves. The fundamental theorem of arithmetic states that every integer greater than one can be uniquely expressed as a product of prime numbers. Modular arithmetic, which deals with remainders after division, has applications in cryptography and computer science.",
    "Graph theory studies networks of nodes connected by edges. A graph can represent social networks, transportation systems, or computer networks. Key concepts include degree of a node, paths between nodes, and graph connectivity. Famous problems include finding the shortest path between two nodes, determining whether a graph is planar, and coloring vertices so no adjacent vertices share the same color.",
    "Differential equations model how quantities change over time or space. Ordinary differential equations involve functions of a single variable, while partial differential equations involve functions of multiple variables. Many natural phenomena are described by differential equations, including population growth, heat conduction, wave propagation, and the motion of celestial bodies under gravitational forces.",
    "Set theory provides the foundational language of modern mathematics. A set is a collection of distinct objects called elements. Operations on sets include union, intersection, and complement. Georg Cantor showed that infinite sets can have different sizes: the set of natural numbers is countably infinite, while the set of real numbers is uncountably infinite, a revolutionary insight in mathematical logic.",
]

HISTORY_PARAGRAPHS = [
    "The Agricultural Revolution, beginning around 10,000 BCE, transformed human societies from nomadic hunter-gatherers to settled farming communities. The domestication of wheat, barley, rice, and maize provided reliable food sources, enabling population growth and the development of villages and cities. Surplus food production freed some individuals from farming, leading to specialization of labor and the emergence of social hierarchies.",
    "Ancient Greece made lasting contributions to philosophy, science, democracy, and the arts. Athenian democracy, established in the 5th century BCE, allowed male citizens to participate directly in governance. Philosophers like Socrates, Plato, and Aristotle developed systematic approaches to ethics, metaphysics, and logic. Greek achievements in mathematics, astronomy, medicine, and architecture influenced civilizations for millennia.",
    "The Roman Empire at its height controlled territories spanning from Britain to Mesopotamia, connected by an extensive network of roads, aqueducts, and trade routes. Roman engineering achievements included the Colosseum, the Pantheon, and sophisticated water supply systems. Roman law established principles of justice and governance that influenced legal systems worldwide. The empire declined through internal political instability, economic troubles, and external invasions.",
    "The Renaissance, spanning roughly from the 14th to the 17th century, was a period of cultural rebirth in Europe. Artists like Leonardo da Vinci and Michelangelo created masterworks in painting and sculpture, emphasizing human anatomy, perspective, and emotion. Scientific inquiry flourished, with Copernicus proposing a heliocentric model and Galileo using telescopes to observe celestial bodies. The printing press, invented by Gutenberg around 1440, democratized knowledge.",
    "The Industrial Revolution began in Britain in the late 18th century, transforming manufacturing from hand production to machine-based processes. The steam engine, developed by James Watt, powered factories, locomotives, and ships. Textile mills mechanized cotton spinning and weaving, dramatically increasing output. Urbanization accelerated as workers migrated from rural areas to factory towns, fundamentally changing social structures and living conditions.",
    "The French Revolution of 1789 overthrew the monarchy and established principles of popular sovereignty and individual rights. The storming of the Bastille symbolized the collapse of royal authority. The Declaration of the Rights of Man and of the Citizen proclaimed equality before the law, freedom of speech, and the right to property. The revolution's ideals influenced democratic movements across Europe and the Americas.",
    "World War II, lasting from 1939 to 1945, was the deadliest conflict in human history. It involved the Allied powers, primarily Britain, the Soviet Union, and the United States, against the Axis powers of Germany, Italy, and Japan. The war saw unprecedented destruction, including the Holocaust and the use of atomic weapons. Its aftermath reshaped the global political order, leading to the United Nations, the Cold War, and decolonization.",
    "The Silk Road was a network of trade routes connecting China to the Mediterranean world for over a thousand years. Merchants traded silk, spices, precious metals, ceramics, and glass across vast distances. Beyond commerce, the Silk Road facilitated the exchange of ideas, religions, technologies, and diseases between civilizations. Buddhism, Islam, and Christianity all spread along these routes, transforming the cultures they reached.",
    "The Age of Exploration in the 15th and 16th centuries saw European navigators charting new sea routes around the globe. Portuguese explorers reached West Africa, rounded the Cape of Good Hope, and established trade routes to India and Southeast Asia. Christopher Columbus, sailing for Spain, reached the Americas in 1492. These voyages initiated the Columbian Exchange, transferring plants, animals, and diseases between the Old and New Worlds.",
    "The Cold War, lasting from roughly 1947 to 1991, was a geopolitical rivalry between the United States and the Soviet Union. Both superpowers amassed nuclear arsenals and competed for influence through proxy wars, space exploration, and ideological propaganda. The arms race created the doctrine of mutually assured destruction. The Cold War ended with the dissolution of the Soviet Union in December 1991.",
]

CS_PARAGRAPHS = [
    "An algorithm is a step-by-step procedure for solving a problem or performing a computation. Algorithm efficiency is measured by time complexity and space complexity, typically expressed using Big O notation. A linear search examines each element in sequence, running in O(n) time, while a binary search on a sorted array runs in O(log n) time. Choosing the right algorithm can dramatically affect program performance.",
    "Data structures organize and store information for efficient access and modification. Arrays provide constant-time access by index but expensive insertions in the middle. Linked lists allow efficient insertions and deletions but require sequential access. Hash tables offer average-case constant-time lookup by mapping keys to indices through a hash function. Trees and graphs model hierarchical and networked relationships.",
    "Machine learning enables computers to improve their performance on tasks through experience. Supervised learning trains models on labeled examples, mapping inputs to known outputs. Unsupervised learning discovers patterns in unlabeled data through clustering and dimensionality reduction. Reinforcement learning trains agents to make decisions by rewarding desirable actions and penalizing undesirable ones in an environment.",
    "Neural networks are computational models inspired by biological neurons. Each artificial neuron receives inputs, applies weights and a bias, then passes the result through an activation function. Deep neural networks contain multiple hidden layers, enabling them to learn hierarchical representations. Backpropagation adjusts weights by computing gradients of the loss function with respect to each parameter using the chain rule of calculus.",
    "Operating systems manage hardware resources and provide services for application software. The kernel handles process scheduling, memory management, file systems, and device drivers. Virtual memory allows programs to use more address space than physical RAM by swapping pages to disk. Process isolation ensures that one program cannot access another's memory, providing security and stability through hardware-enforced protection boundaries.",
    "Computer networking enables communication between devices across local and global distances. The Internet Protocol suite, commonly known as TCP/IP, organizes communication into layers: link, internet, transport, and application. TCP provides reliable, ordered delivery of data streams, while UDP offers faster but unreliable transmission. HTTP enables web browsing, SMTP handles email, and DNS translates domain names to IP addresses.",
    "Database systems provide structured storage, retrieval, and management of data. Relational databases organize data into tables with rows and columns, using SQL for querying. Transactions ensure atomicity, consistency, isolation, and durability through the ACID properties. NoSQL databases, including document stores, key-value stores, and graph databases, offer flexible schemas and horizontal scalability for specific workloads.",
    "Cryptography protects information through mathematical transformations. Symmetric encryption uses the same key for encryption and decryption, with AES being a widely used standard. Asymmetric encryption uses a public key for encryption and a private key for decryption, enabling secure key exchange. Hash functions produce fixed-size digests from arbitrary input, used for integrity verification and password storage.",
    "Software engineering applies systematic methods to the development of reliable software systems. The software development lifecycle includes requirements analysis, design, implementation, testing, and maintenance. Version control systems like Git track changes to source code, enabling collaboration among developers. Continuous integration and deployment automate the building, testing, and releasing of software updates.",
    "Compilers translate high-level programming languages into machine code that processors can execute. The compilation process involves lexical analysis, parsing, semantic analysis, optimization, and code generation. Lexical analysis breaks source code into tokens, while parsing builds an abstract syntax tree representing the program's structure. Optimization passes improve the generated code's performance without changing its observable behavior.",
]

PHILOSOPHY_PARAGRAPHS = [
    "Epistemology investigates the nature, sources, and limits of knowledge. Rationalists like Descartes argued that reason is the primary source of knowledge, while empiricists like Locke and Hume emphasized sensory experience. Kant synthesized these views, proposing that knowledge arises from the interaction of sensory input with innate mental structures. Contemporary epistemology examines issues of justification, skepticism, and the reliability of cognitive processes.",
    "Ethics examines questions of right and wrong, good and bad, and how we ought to live. Consequentialist theories like utilitarianism judge actions by their outcomes, seeking to maximize overall well-being. Deontological theories, associated with Kant, evaluate actions based on whether they conform to moral duties and rules. Virtue ethics, rooted in Aristotle, focuses on developing moral character traits like courage, temperance, and justice.",
    "Logic is the study of valid reasoning and argumentation. Deductive arguments guarantee their conclusions when premises are true, as in the classic syllogism: all humans are mortal, Socrates is human, therefore Socrates is mortal. Inductive arguments draw probable conclusions from observed evidence. Formal logic uses symbolic notation to analyze argument structures independently of their content, revealing patterns of valid inference.",
    "Philosophy of mind explores the relationship between mental states and physical processes. Dualism, proposed by Descartes, holds that mind and body are distinct substances. Physicalism argues that mental states are entirely physical, reducible to brain states. Functionalism identifies mental states with their causal roles rather than their physical composition, suggesting that minds could be realized in different physical substrates.",
    "Political philosophy examines the foundations of government, justice, and individual rights. Social contract theory, developed by Hobbes, Locke, and Rousseau, explains political authority as arising from agreement among individuals. Rawls proposed that just institutions are those we would choose behind a veil of ignorance, not knowing our place in society. Libertarians emphasize individual liberty and minimal state intervention.",
]

TECHNOLOGY_PARAGRAPHS = [
    "The Internet evolved from ARPANET, a military research network established in 1969. The development of TCP/IP protocols in the 1970s and 1980s enabled diverse computer networks to interconnect. Tim Berners-Lee invented the World Wide Web in 1989, creating a system of hyperlinked documents accessible via browsers. The subsequent growth of e-commerce, social media, and cloud computing transformed communication, business, and daily life worldwide.",
    "Artificial intelligence research aims to create systems capable of performing tasks that typically require human intelligence. Early AI focused on symbolic reasoning and expert systems that encoded human knowledge as rules. Modern AI predominantly uses machine learning, especially deep neural networks, which learn patterns directly from large datasets. Large language models trained on text corpora can generate coherent text, translate languages, and answer questions.",
    "Robotics integrates mechanical engineering, electronics, and computer science to build machines that can interact with the physical world. Industrial robots perform repetitive tasks like welding, painting, and assembly with precision and speed. Service robots assist in healthcare, logistics, and domestic settings. Autonomous vehicles use sensors, computer vision, and planning algorithms to navigate roads without human intervention.",
    "Semiconductor technology forms the basis of modern electronics. Transistors, made from materials like silicon, act as switches and amplifiers in electronic circuits. Moore's Law observed that the number of transistors on integrated circuits doubled approximately every two years, driving exponential increases in computing power. Modern processors contain billions of transistors, each only a few nanometers wide.",
    "Renewable energy technologies harness natural resources to generate electricity without depleting finite fuel supplies. Solar photovoltaic cells convert sunlight directly into electricity using semiconductor materials. Wind turbines capture kinetic energy from moving air. Hydroelectric dams generate power from flowing water. Battery storage systems address the intermittency of solar and wind power, enabling a more reliable renewable energy grid.",
]

ECONOMICS_PARAGRAPHS = [
    "Supply and demand are fundamental forces that determine market prices and quantities. When demand for a good exceeds its supply, prices tend to rise, encouraging producers to supply more. When supply exceeds demand, prices fall, and producers reduce output. Market equilibrium occurs at the price where the quantity demanded equals the quantity supplied. External factors like government policy and consumer preferences shift these curves.",
    "Monetary policy, conducted by central banks, manages the money supply and interest rates to influence economic activity. Lowering interest rates encourages borrowing and investment, stimulating growth during recessions. Raising rates can cool an overheating economy and control inflation. Central banks also use tools like open market operations and reserve requirements to implement their policy objectives.",
    "International trade allows countries to specialize in producing goods where they have a comparative advantage, trading for goods they produce less efficiently. Free trade agreements reduce tariffs and barriers between nations, increasing the volume of trade. However, trade can also lead to job displacement in certain industries, prompting debates about protectionism, trade deficits, and the distribution of economic benefits.",
    "Gross domestic product measures the total market value of all final goods and services produced within a country during a specific period. GDP can be calculated by summing consumption, investment, government spending, and net exports. While GDP is a useful indicator of economic activity, it does not capture income inequality, environmental degradation, or unpaid domestic labor. Alternative measures like the Human Development Index provide broader perspectives.",
    "Financial markets facilitate the exchange of capital between savers and borrowers. Stock markets allow companies to raise funds by selling shares of ownership to investors. Bond markets enable governments and corporations to borrow money by issuing debt securities. Foreign exchange markets handle the conversion of currencies for international trade. Efficient markets reflect all available information in asset prices.",
]

LITERATURE_PARAGRAPHS = [
    "Shakespeare's plays explore universal themes of love, power, jealousy, and mortality through complex characters and poetic language. Hamlet examines the consequences of indecision and the corruption of the Danish court. Macbeth portrays the destructive effects of unchecked ambition. A Midsummer Night's Dream blends comedy, magic, and romance in an enchanted forest, showing the irrationality of love.",
    "The novel emerged as a literary form in the 18th century, offering sustained prose narratives exploring individual experience and social conditions. Cervantes' Don Quixote, often considered the first modern novel, satirizes chivalric romances through its delusional protagonist. Dickens documented the social inequities of Victorian England, while Austen explored class, marriage, and moral development in the landed gentry.",
    "Poetry uses rhythm, imagery, and concentrated language to evoke emotion and meaning. Sonnets follow strict formal constraints, with Shakespeare's 154 sonnets exploring love, beauty, and time. Free verse abandons traditional meter and rhyme, allowing greater flexibility of expression. Haiku, a Japanese form of three lines with five, seven, and five syllables, captures fleeting moments of perception and nature.",
    "Narrative techniques shape how stories are told and experienced by readers. First-person narration provides intimate access to a character's thoughts but limits perspective. Third-person omniscient narration offers a broader view, revealing multiple characters' inner lives. Stream of consciousness, used by writers like Joyce and Woolf, mimics the continuous flow of thoughts, sensations, and associations in the human mind.",
    "Mythology and folklore preserve cultural beliefs, values, and explanations of natural phenomena across generations. Greek mythology populated the world with gods, heroes, and monsters, offering stories about human nature and divine intervention. Norse mythology depicted a cosmos of nine worlds connected by the great tree Yggdrasil. Creation myths from cultures worldwide share common motifs of primordial chaos, divine crafting, and the origin of humanity.",
]

PSYCHOLOGY_PARAGRAPHS = [
    "Cognitive psychology studies mental processes including attention, memory, language, and problem solving. Working memory holds and manipulates information during complex cognitive tasks, with a limited capacity of roughly four to seven items. Long-term memory stores vast amounts of information organized through semantic networks and schemas. Cognitive biases, such as confirmation bias and anchoring, systematically distort reasoning and judgment.",
    "Developmental psychology examines how humans grow and change throughout the lifespan. Piaget identified four stages of cognitive development, from sensorimotor exploration in infancy to formal operational thinking in adolescence. Attachment theory, developed by Bowlby and Ainsworth, describes how early bonds with caregivers shape emotional development and relationship patterns. Erik Erikson proposed eight psychosocial stages spanning from infancy to old age.",
    "Social psychology investigates how individuals think about, influence, and relate to one another. Conformity research by Asch showed that people often agree with an incorrect majority. Milgram's obedience experiments demonstrated that ordinary individuals would administer apparent electric shocks when instructed by an authority figure. Social identity theory explains how group membership shapes self-concept and intergroup behavior.",
    "Behavioral psychology, founded by Watson and Skinner, focuses on observable actions rather than internal mental states. Classical conditioning, discovered by Pavlov, pairs a neutral stimulus with an unconditioned stimulus to elicit a learned response. Operant conditioning uses reinforcement and punishment to shape behavior. Behavioral principles are applied in therapy, education, animal training, and organizational management.",
]

ALL_PARAGRAPHS = (
    SCIENCE_PARAGRAPHS
    + MATH_PARAGRAPHS
    + HISTORY_PARAGRAPHS
    + CS_PARAGRAPHS
    + PHILOSOPHY_PARAGRAPHS
    + TECHNOLOGY_PARAGRAPHS
    + ECONOMICS_PARAGRAPHS
    + LITERATURE_PARAGRAPHS
    + PSYCHOLOGY_PARAGRAPHS
)

# ---------------------------------------------------------------------------
# Instruction pairs for Basic / Solo training
# ---------------------------------------------------------------------------

INSTRUCTION_PAIRS = [
    (
        "What is photosynthesis?",
        "Photosynthesis is the process by which plants convert light energy into chemical energy. In the chloroplasts of plant cells, chlorophyll absorbs sunlight and uses it to transform carbon dioxide and water into glucose and oxygen. The light-dependent reactions occur in the thylakoid membranes, producing ATP and NADPH, while the Calvin cycle in the stroma uses these molecules to fix carbon into sugar.",
    ),
    (
        "Explain Newton's second law of motion.",
        "Newton's second law states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. This relationship is expressed as F = ma, where F is the net force, m is the mass, and a is the acceleration. A heavier object requires more force to achieve the same acceleration as a lighter one.",
    ),
    (
        "What causes earthquakes?",
        "Earthquakes are caused by the sudden release of energy in Earth's crust, usually resulting from the movement of tectonic plates. When plates push against, pull apart from, or slide past each other, stress builds up along fault lines. When this stress exceeds the strength of the rocks, it is released as seismic waves that shake the ground. The point where rupture begins is called the focus, and the point directly above on the surface is the epicenter.",
    ),
    (
        "How does a computer store information?",
        "Computers store information as binary data, sequences of ones and zeros called bits. Eight bits form a byte, the basic unit of data storage. RAM provides fast temporary storage that loses data when power is off. Hard drives and solid-state drives provide persistent storage using magnetic platters or flash memory chips. Data is organized into files by the operating system's file system.",
    ),
    (
        "What is the difference between weather and climate?",
        "Weather refers to the short-term atmospheric conditions at a specific place and time, including temperature, humidity, precipitation, and wind. Climate is the average pattern of weather conditions in a region over a long period, typically 30 years or more. A region might have a rainy day (weather) while still being classified as arid (climate). Climate change refers to long-term shifts in these average patterns.",
    ),
    (
        "Explain how vaccines work.",
        "Vaccines train the immune system to recognize and fight specific pathogens without causing the actual disease. They contain weakened, inactivated, or fragments of the pathogen that trigger an immune response. The body produces antibodies and memory cells that remember the pathogen. If the person later encounters the real pathogen, their immune system can respond quickly and effectively, preventing illness or reducing its severity.",
    ),
    (
        "What is compound interest?",
        "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Unlike simple interest, which is only calculated on the principal, compound interest grows exponentially over time. The formula is A = P(1 + r/n)^(nt), where P is the principal, r is the annual rate, n is the compounding frequency, and t is time in years. This is why starting to save early has such a powerful effect.",
    ),
    (
        "How do neural networks learn?",
        "Neural networks learn by adjusting the weights and biases of connections between artificial neurons to minimize prediction errors. During training, input data passes through layers of neurons (forward propagation), producing an output that is compared to the expected result. The error is calculated using a loss function, and backpropagation computes gradients that indicate how to adjust each weight. An optimizer like SGD or Adam updates the weights iteratively.",
    ),
    (
        "What is DNA and why is it important?",
        "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions for the development, functioning, and reproduction of all known living organisms. It consists of two strands twisted into a double helix, with pairs of nucleotide bases (adenine-thymine and guanine-cytosine) forming the rungs. DNA encodes genes, which are segments that specify the sequence of amino acids in proteins that carry out most cellular functions.",
    ),
    (
        "Explain the water cycle.",
        "The water cycle is the continuous movement of water between Earth's surface and atmosphere. Water evaporates from oceans, lakes, and rivers when heated by the sun. Plants also release water vapor through transpiration. This vapor rises, cools, and condenses to form clouds. When droplets in clouds grow large enough, they fall as precipitation—rain, snow, sleet, or hail. The water then flows through rivers and underground aquifers back to the oceans.",
    ),
    (
        "What is inflation and what causes it?",
        "Inflation is a sustained increase in the general price level of goods and services in an economy over time, reducing purchasing power. Demand-pull inflation occurs when aggregate demand exceeds aggregate supply. Cost-push inflation happens when production costs rise, such as increased wages or raw material prices. Central banks manage inflation through monetary policy, primarily by adjusting interest rates and controlling the money supply.",
    ),
    (
        "How does encryption protect data?",
        "Encryption transforms readable data (plaintext) into unreadable ciphertext using a mathematical algorithm and a key. Only someone with the correct decryption key can convert the ciphertext back to plaintext. Symmetric encryption uses the same key for both operations, while asymmetric encryption uses a public key for encryption and a private key for decryption. HTTPS uses encryption to protect web traffic from eavesdropping.",
    ),
    (
        "What is natural selection?",
        "Natural selection is the process by which organisms with traits better suited to their environment are more likely to survive and reproduce. Over generations, these advantageous traits become more common in the population. Three conditions are necessary: variation in traits within a population, heritability of those traits, and differential fitness based on those traits. Natural selection does not act on individuals but on populations over time.",
    ),
    (
        "Explain the concept of supply and demand.",
        "Supply and demand is a fundamental economic model describing how prices are determined in a market. Demand represents how much of a product consumers are willing to buy at various prices—typically more at lower prices. Supply represents how much producers are willing to sell—typically more at higher prices. The equilibrium price is where the quantity demanded equals the quantity supplied, and market forces push prices toward this point.",
    ),
    (
        "What is the scientific method?",
        "The scientific method is a systematic approach to investigating phenomena and acquiring knowledge. It begins with observation and the formulation of a question. A testable hypothesis is proposed, and experiments are designed to test it while controlling variables. Data is collected and analyzed to determine whether the results support or refute the hypothesis. Results are communicated through peer-reviewed publication, and other scientists attempt to replicate the findings.",
    ),
    (
        "How does the human heart work?",
        "The human heart is a muscular organ that pumps blood through the circulatory system. It has four chambers: the right atrium receives deoxygenated blood from the body and passes it to the right ventricle, which pumps it to the lungs for oxygenation. Oxygenated blood returns to the left atrium, flows to the left ventricle, and is pumped to the rest of the body. The heart beats about 100,000 times per day, regulated by electrical signals from the sinoatrial node.",
    ),
    (
        "What is machine learning?",
        "Machine learning is a branch of artificial intelligence where computer systems learn from data without being explicitly programmed for every task. In supervised learning, models are trained on labeled examples to predict outcomes for new inputs. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning trains agents through trial and error, rewarding good decisions. Common algorithms include linear regression, decision trees, and neural networks.",
    ),
    (
        "Explain plate tectonics.",
        "Plate tectonics is the theory that Earth's outer shell is divided into several large plates that float on the semi-fluid asthenosphere beneath them. These plates move relative to each other at boundaries where they diverge, converge, or slide past one another. At divergent boundaries, new crust forms as plates separate. At convergent boundaries, one plate may subduct beneath another, forming trenches and volcanoes. Transform boundaries produce lateral movement and earthquakes.",
    ),
    (
        "What are black holes?",
        "Black holes are regions of spacetime where gravity is so intense that nothing, not even light, can escape once it crosses the event horizon. They form when massive stars collapse at the end of their life cycle, compressing their mass into an incredibly small volume. Supermassive black holes, millions to billions of times the Sun's mass, reside at the centers of most galaxies. Black holes can be detected by their gravitational effects on nearby matter and by radiation emitted as material falls inward.",
    ),
    (
        "How do computers execute programs?",
        "Computers execute programs through a cycle of fetch, decode, and execute. The processor fetches an instruction from memory at the address stored in the program counter. The instruction decoder interprets what operation to perform. The execution unit carries out the operation, such as arithmetic, logic, or memory access. The program counter advances to the next instruction. Modern processors execute billions of these cycles per second and use pipelining to overlap stages.",
    ),
    (
        "What is the greenhouse effect?",
        "The greenhouse effect is a natural process where certain gases in Earth's atmosphere trap heat from the sun, keeping the planet warm enough to support life. Sunlight passes through the atmosphere and warms the surface, which then emits infrared radiation. Greenhouse gases like carbon dioxide, methane, and water vapor absorb some of this radiation and re-emit it in all directions, warming the lower atmosphere. Human activities have increased greenhouse gas concentrations, enhancing this effect.",
    ),
    (
        "Explain the concept of recursion in programming.",
        "Recursion is a programming technique where a function calls itself to solve a problem by breaking it into smaller subproblems. Each recursive call works on a smaller input until reaching a base case that can be solved directly without further recursion. Classic examples include computing factorials, traversing tree structures, and the Fibonacci sequence. Recursion uses the call stack to track each invocation, so deep recursion can cause stack overflow errors.",
    ),
    (
        "What is general relativity?",
        "General relativity, published by Albert Einstein in 1915, describes gravity as the curvature of spacetime caused by mass and energy. Massive objects like stars and planets warp the fabric of spacetime around them, and other objects follow curved paths through this warped geometry. This explains why light bends around massive objects, why time passes more slowly in stronger gravitational fields, and why the universe can expand. GPS satellites must account for relativistic time differences to maintain accuracy.",
    ),
    (
        "How do antibiotics work?",
        "Antibiotics are drugs that kill or inhibit the growth of bacteria. They target structures or processes unique to bacterial cells, minimizing harm to human cells. Penicillin and related drugs interfere with cell wall synthesis, causing bacteria to burst. Tetracyclines block bacterial ribosomes, preventing protein production. Fluoroquinolones inhibit DNA replication. Antibiotic resistance develops when bacteria mutate or acquire genes that neutralize the drug, making infections harder to treat.",
    ),
]

# ---------------------------------------------------------------------------
# DPO / RLHF preference pairs
# ---------------------------------------------------------------------------

DPO_PAIRS = [
    {
        "prompt": "Explain what gravity is.",
        "chosen": "Gravity is a fundamental force of nature that causes objects with mass to attract one another. According to Newton's law of universal gravitation, every mass attracts every other mass with a force proportional to the product of their masses and inversely proportional to the square of the distance between them. Einstein's general relativity describes gravity as the curvature of spacetime caused by mass and energy, where objects follow curved paths through warped space.",
        "rejected": "gravity is when stuff falls down. things are heavy because of gravity. the moon has less gravity so you can jump higher there.",
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris. Located on the River Seine in northern France, Paris is the country's largest city and has been its political, economic, and cultural center for centuries. It is home to landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.",
        "rejected": "france capital is paris i think. its a city in europe somewhere.",
    },
    {
        "prompt": "How do plants grow?",
        "chosen": "Plants grow through a combination of cell division, elongation, and differentiation. Seeds germinate when exposed to adequate moisture, warmth, and sometimes light. The root system develops downward to absorb water and minerals from the soil, while the shoot grows upward toward light. Photosynthesis in the leaves converts carbon dioxide and water into glucose using sunlight energy, providing the fuel for growth. Hormones like auxin regulate the direction and rate of growth.",
        "rejected": "plants grow from seeds. you put them in dirt and water them. they get bigger from the sun. thats basically it.",
    },
    {
        "prompt": "Write a function to reverse a string.",
        "chosen": "Here is a Python function to reverse a string:\n\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n\nThis uses Python's slice notation with a step of -1, which creates a new string with characters in reverse order. For example, reverse_string('hello') returns 'olleh'. The time complexity is O(n) where n is the length of the string, and it creates a new string rather than modifying the original.",
        "rejected": "def reverse(s):\n  r = ''\n  for i in s:\n    r = i + r\n  return r\n\nthat should work i think",
    },
    {
        "prompt": "What causes seasons on Earth?",
        "chosen": "Earth's seasons are caused by the tilt of its rotational axis, approximately 23.5 degrees relative to its orbital plane around the Sun. As Earth orbits the Sun, different hemispheres receive varying amounts of direct sunlight throughout the year. When the Northern Hemisphere is tilted toward the Sun, it experiences summer with longer days and more direct sunlight, while the Southern Hemisphere experiences winter. Six months later, the situation reverses. The equinoxes occur when neither hemisphere is tilted toward the Sun.",
        "rejected": "seasons happen because earth is closer to the sun in summer and farther in winter. thats why its hot in summer and cold in winter.",
    },
    {
        "prompt": "Explain the difference between RAM and storage.",
        "chosen": "RAM (Random Access Memory) and storage serve different roles in a computer. RAM is volatile memory that provides fast, temporary workspace for the CPU during active tasks—data is lost when power is turned off. It typically offers speeds of tens of gigabytes per second. Storage devices like SSDs and hard drives provide persistent, non-volatile memory that retains data without power. Storage is slower than RAM but offers much larger capacity at lower cost per gigabyte. The operating system loads programs from storage into RAM when you run them.",
        "rejected": "ram is fast memory, storage is slow memory. ram loses stuff when you turn off the computer. storage keeps it. you need both.",
    },
    {
        "prompt": "What is democracy?",
        "chosen": "Democracy is a system of government in which power is vested in the people, who exercise it directly or through elected representatives. In a direct democracy, citizens vote on laws and policies themselves. In a representative democracy, citizens elect officials to make decisions on their behalf. Key principles include free and fair elections, protection of individual rights, rule of law, separation of powers, and freedom of expression. Modern democracies typically include constitutional safeguards to prevent majority tyranny.",
        "rejected": "democracy means the people get to vote for stuff. its when everyone gets a say in how things are run. america is a democracy.",
    },
    {
        "prompt": "How does the Internet work?",
        "chosen": "The Internet is a global network of interconnected computer networks that communicate using standardized protocols. When you request a webpage, your browser sends an HTTP request through your local network to your Internet Service Provider. DNS servers translate the domain name into an IP address. The request travels through multiple routers across the network, guided by routing protocols. The destination server processes the request and sends back the webpage data in packets, which are reassembled by your browser for display.",
        "rejected": "the internet works with wifi and cables. your computer sends stuff to other computers through the internet. google and websites are on servers somewhere.",
    },
    {
        "prompt": "What is the Pythagorean theorem?",
        "chosen": "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) equals the sum of the squares of the lengths of the other two sides. Expressed as a² + b² = c², where c is the hypotenuse and a and b are the other sides. For example, a triangle with sides 3 and 4 has a hypotenuse of 5, since 9 + 16 = 25. This theorem has applications in distance calculations, navigation, construction, and many areas of mathematics.",
        "rejected": "its a squared plus b squared equals c squared. you use it for triangles. the long side is c.",
    },
    {
        "prompt": "Explain how a vaccine works.",
        "chosen": "A vaccine works by training the immune system to recognize and combat a specific pathogen without causing the disease itself. It contains either weakened or inactivated forms of the pathogen, pieces of it (like proteins), or instructions for cells to produce those pieces (as in mRNA vaccines). When administered, the immune system mounts a response, producing antibodies and activating T cells. Crucially, memory cells are formed that persist long-term, enabling a rapid and effective immune response if the real pathogen is encountered later.",
        "rejected": "vaccines put a weak version of the virus in you so your body learns to fight it. then if you get the real virus your body already knows what to do.",
    },
    {
        "prompt": "What is an algorithm?",
        "chosen": "An algorithm is a finite, well-defined sequence of instructions for solving a problem or performing a computation. It takes one or more inputs and produces an output after a finite number of steps. Key properties include definiteness (each step is precisely defined), finiteness (it terminates after finite steps), and effectiveness (each step is feasible). Examples include binary search for finding items in sorted lists, Dijkstra's algorithm for shortest paths in graphs, and quicksort for ordering elements. Algorithm efficiency is analyzed using time and space complexity.",
        "rejected": "an algorithm is like steps to do something. like a recipe but for computers. you tell the computer what to do step by step.",
    },
    {
        "prompt": "Describe the structure of an atom.",
        "chosen": "An atom consists of a dense central nucleus surrounded by a cloud of electrons. The nucleus contains positively charged protons and electrically neutral neutrons, held together by the strong nuclear force. Electrons, which carry a negative charge, occupy regions around the nucleus described by quantum mechanical orbitals. The number of protons (atomic number) determines the element. In a neutral atom, the number of electrons equals the number of protons. Isotopes of an element have the same number of protons but different numbers of neutrons.",
        "rejected": "atoms have protons neutrons and electrons. protons and neutrons are in the middle and electrons go around the outside. protons are positive electrons are negative.",
    },
    {
        "prompt": "What is climate change?",
        "chosen": "Climate change refers to significant, long-term shifts in global temperatures and weather patterns. While natural factors like volcanic eruptions and solar cycles cause some variation, the current rapid warming trend is primarily driven by human activities, particularly the burning of fossil fuels, which releases greenhouse gases like carbon dioxide and methane. These gases trap heat in the atmosphere, raising global temperatures. Consequences include rising sea levels, more frequent extreme weather events, ocean acidification, and disruption of ecosystems and agriculture.",
        "rejected": "climate change is when the weather changes over time. some people think humans cause it by pollution. the earth is getting warmer.",
    },
    {
        "prompt": "How do batteries work?",
        "chosen": "Batteries convert chemical energy into electrical energy through electrochemical reactions. A battery contains two electrodes—an anode (negative) and a cathode (positive)—separated by an electrolyte. During discharge, a chemical reaction at the anode releases electrons, which flow through an external circuit to the cathode, providing electrical current. At the cathode, another reaction consumes electrons. In rechargeable batteries like lithium-ion, applying an external voltage reverses these reactions, restoring the original chemical state.",
        "rejected": "batteries have chemicals inside that make electricity. when the chemicals run out the battery dies. rechargeable ones can be charged back up.",
    },
    {
        "prompt": "What is the theory of relativity?",
        "chosen": "Einstein's theory of relativity consists of two parts. Special relativity (1905) established that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is constant regardless of the observer's motion. This leads to time dilation, length contraction, and the famous equation E=mc². General relativity (1915) extended these ideas to include gravity, describing it as the curvature of spacetime caused by mass and energy. Massive objects warp spacetime, and this curvature determines how objects move.",
        "rejected": "relativity is einsteins theory about how time and space work. e equals mc squared. time goes slower if youre moving fast. gravity bends light.",
    },
]


def build_pretrain_text(target_kb: int) -> str:
    """Build diverse pre-training text by cycling through paragraphs."""
    target_bytes = target_kb * 1024
    sections = []
    topic_names = [
        "Natural Sciences",
        "Mathematics",
        "World History",
        "Computer Science",
        "Philosophy",
        "Technology",
        "Economics",
        "Literature",
        "Psychology",
    ]
    topic_groups = [
        SCIENCE_PARAGRAPHS,
        MATH_PARAGRAPHS,
        HISTORY_PARAGRAPHS,
        CS_PARAGRAPHS,
        PHILOSOPHY_PARAGRAPHS,
        TECHNOLOGY_PARAGRAPHS,
        ECONOMICS_PARAGRAPHS,
        LITERATURE_PARAGRAPHS,
        PSYCHOLOGY_PARAGRAPHS,
    ]

    # First pass: all paragraphs in order with topic headers
    for name, paras in zip(topic_names, topic_groups):
        sections.append(f"\n{'=' * 60}")
        sections.append(f"  {name}")
        sections.append(f"{'=' * 60}\n")
        for p in paras:
            sections.append(p)
            sections.append("")  # blank line between paragraphs

    text = "\n".join(sections)

    # Repeat with shuffled order until we hit target size
    cycle = 0
    while len(text.encode("utf-8")) < target_bytes:
        cycle += 1
        shuffled = list(ALL_PARAGRAPHS)
        random.shuffle(shuffled)

        sections = [f"\n--- Section {cycle} ---\n"]
        for i, p in enumerate(shuffled):
            # Add slight variation to avoid exact dedup
            prefix = f"[{cycle}.{i + 1}] "
            sections.append(prefix + p)
            sections.append("")

        text += "\n".join(sections)

    return text


def build_basic_text(target_kb: int) -> str:
    """Build User:/Assistant: instruction pairs for Basic training."""
    target_bytes = target_kb * 1024
    lines = []

    # First pass: all pairs
    for q, a in INSTRUCTION_PAIRS:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
        lines.append("")

    text = "\n".join(lines)

    # Repeat with variation until target
    cycle = 0
    while len(text.encode("utf-8")) < target_bytes:
        cycle += 1
        pairs = list(INSTRUCTION_PAIRS)
        random.shuffle(pairs)
        extra = [f"\n--- Set {cycle} ---\n"]
        for q, a in pairs:
            extra.append(f"User: {q}")
            extra.append(f"Assistant: {a}")
            extra.append("")
        text += "\n".join(extra)

    return text


def build_dpo_jsonl() -> str:
    """Build JSONL preference pairs for DPO/RLHF training."""
    lines = []
    for pair in DPO_PAIRS:
        lines.append(json.dumps(pair, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def main():
    # Pre-Train data
    pretrain_path = PRETRAIN_DIR / "smoke_test.txt"
    print(f"Generating {pretrain_path} (~{TARGET_PRETRAIN_KB} KB)...")
    pretrain_text = build_pretrain_text(TARGET_PRETRAIN_KB)
    pretrain_path.write_text(pretrain_text, encoding="utf-8")
    size_kb = pretrain_path.stat().st_size / 1024
    print(f"  Created: {size_kb:.0f} KB")

    # Basic / Solo data
    basic_path = DATA_DIR / "smoke_test_basic.txt"
    print(f"Generating {basic_path} (~{TARGET_BASIC_KB} KB)...")
    basic_text = build_basic_text(TARGET_BASIC_KB)
    basic_path.write_text(basic_text, encoding="utf-8")
    size_kb = basic_path.stat().st_size / 1024
    print(f"  Created: {size_kb:.0f} KB")

    # DPO / RLHF data
    dpo_path = DATA_DIR / "smoke_test_dpo.jsonl"
    print(f"Generating {dpo_path}...")
    dpo_text = build_dpo_jsonl()
    dpo_path.write_text(dpo_text, encoding="utf-8")
    size_kb = dpo_path.stat().st_size / 1024
    print(f"  Created: {size_kb:.1f} KB ({len(DPO_PAIRS)} pairs)")

    print("\nDone! Files created:")
    print(f"  Pre-Train:  {pretrain_path.relative_to(Path.cwd())}")
    print(f"  Basic/Solo: {basic_path.relative_to(Path.cwd())}")
    print(f"  DPO/RLHF:  {dpo_path.relative_to(Path.cwd())}")
    print("\nUsage in FORGE:")
    print("  Pre-Train  -> select 'data/pretrain' directory")
    print("  Basic/Solo -> browse to 'data/smoke_test_basic.txt'")
    print("  DPO        -> browse to 'data/smoke_test_dpo.jsonl'")
    print("  RLHF       -> browse to 'data/smoke_test_dpo.jsonl'")
    print("  Distill / AI-Guided / Dialogue / Self-Play -> no data file needed")


if __name__ == "__main__":
    main()
