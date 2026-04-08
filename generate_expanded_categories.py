"""
SGP-Tribe3 - Generate Expanded Categories for V2 Manuscript
==========================================================
Generate 20 new categories to expand from 10 to 30 total categories.
Each category has 20 stimuli = 400 new stimuli.

Usage:
    python generate_expanded_categories.py
"""

import json
import os

NEW_CATEGORIES = {
    "biological": {
        "expected": ["G6_limbic", "G7_sensory"],
        "texts": [
            "The human heart pumps approximately 5 liters of blood per minute.",
            "Neurons transmit electrical signals at speeds up to 120 meters per second.",
            "The liver performs over 500 metabolic functions including detoxification.",
            "Muscles contract when receiving electrical impulses from motor neurons.",
            "The immune system produces antibodies to fight foreign pathogens.",
            "Red blood cells carry oxygen using the protein hemoglobin.",
            "The stomach lining secretes hydrochloric acid for digestion.",
            "Bone tissue regenerates continuously throughout life.",
            "The kidneys filter approximately 180 liters of blood daily.",
            "Insulin regulates blood glucose levels in the bloodstream.",
            "The skin is the largest organ providing protective barrier.",
            "Lungs contain millions of tiny air sacs called alveoli.",
            "The nervous system processes sensory input and coordinates responses.",
            "Hormones released by glands regulate bodily functions.",
            "The pancreas produces digestive enzymes and regulates blood sugar.",
            "White blood cells defend against infection and disease.",
            "The heart consists of four chambers: two atria and two ventricles.",
            "The digestive system breaks down food into absorbable nutrients.",
            "The respiratory system exchanges oxygen and carbon dioxide.",
            "The skeletal system provides structural support and protection."
        ]
    },
    "technological": {
        "expected": ["G4_pfc", "G1_broca"],
        "texts": [
            "The semiconductor chip contains billions of transistors on a single die.",
            "Machine learning algorithms improve through exposure to more data.",
            "Cloud computing provides on-demand access to shared computing resources.",
            "The smartphone integrates camera, processor, and wireless communication.",
            "Blockchain technology enables decentralized transaction verification.",
            "The operating system manages hardware resources and running applications.",
            "Quantum computers leverage superposition to perform parallel calculations.",
            "The internet protocols enable global data communication between computers.",
            "Artificial neural networks are inspired by biological neuron structure.",
            "Computer algorithms solve complex problems through step-by-step procedures.",
            "The database stores and retrieves structured digital information.",
            "Cybersecurity protects systems from unauthorized access and attacks.",
            "The compiler translates high-level code into machine-readable instructions.",
            "Wireless networks use radio waves to transmit data without physical cables.",
            "The graphics processing unit accelerates visual rendering computations.",
            "Data compression reduces file size while preserving essential information.",
            "The microprocessor executes instructions in fetch-decode-execute cycles.",
            "Encryption protects data by converting it into unreadable formats.",
            "The sensor array detects environmental conditions and converts them to signals.",
            "Software debugging identifies and removes errors from program code."
        ]
    },
    "mathematical": {
        "expected": ["G4_pfc", "G1_broca"],
        "texts": [
            "The derivative of x squared equals 2x using power rule differentiation.",
            "The integral of one over x equals the natural logarithm of x.",
            "The eigenvalues of a matrix satisfy its characteristic equation.",
            "Prime numbers are divisible only by one and themselves.",
            "The Pythagorean theorem relates sides of a right triangle: a squared plus b squared equals c squared.",
            "Euler's identity links five fundamental mathematical constants: e to the power of i pi plus one equals zero.",
            "The Gaussian distribution describes natural phenomena through mean and standard deviation.",
            "Calculus relates rates of change through derivatives and integrals.",
            "Group theory studies algebraic structures with binary operations.",
            "Topology examines properties preserved under continuous deformations.",
            "The Fibonacci sequence appears throughout nature in phyllotaxis patterns.",
            "Vector spaces form the foundation of linear algebra.",
            "Probability theory quantifies uncertainty through mathematical models.",
            "Number theory explores properties and relationships of integers.",
            "Graph theory analyzes networks of connected nodes and edges.",
            "The complex plane extends real numbers with imaginary component i.",
            "Set theory provides foundations for all mathematical structures.",
            "The Taylor series approximates functions using infinite polynomial sums.",
            "Differential equations describe relationships between functions and their derivatives.",
            "Mathematical proofs establish truths through logical deduction."
        ]
    },
    "musical": {
        "expected": ["G7_sensory", "G2_wernicke"],
        "texts": [
            "The piano keyboard spans seven octaves plus a minor third from lowest to highest note.",
            "Rhythm creates temporal patterns through organized beats and rests.",
            "Harmony combines simultaneous pitches forming chords and progressions.",
            "The violin produces sound through bow friction on stretched strings.",
            "Dynamics control musical loudness from pianissimo to fortissimo.",
            "The drum kit provides rhythmic foundation through percussion instruments.",
            "Melody combines sequential pitches forming memorable musical lines.",
            "The orchestra sections include strings, woodwinds, brass, and percussion.",
            "Tempo marks indicate speed from largo to presto in beats per minute.",
            "The human voice generates sound through vocal cord vibration.",
            "Counterpoint combines independent melodic lines following voice-leading rules.",
            "The guitar fretboard enables playing melodies and chords through finger placement.",
            "Sonata form structures movements through exposition, development, and recapitulation.",
            "The saxophone uses a single reed to produce warm tonal colors.",
            "Tonal harmony establishes functional relationships between chords.",
            "The trumpet projects bright timbre through brass mouthpiece vibration.",
            "Fugue demonstrates contrapuntal technique through imitative entries.",
            "The flute creates sound through air stream directed across embouchure hole.",
            "Musical notation communicates pitch, rhythm, and expression to performers.",
            "The cello provides rich bass register through bowed string vibration."
        ]
    },
    "visual": {
        "expected": ["G7_sensory", "G3_tpj"],
        "texts": [
            "The crimson sunset painted the sky in shades of orange and pink.",
            "The geometric pattern repeated in perfect symmetry across the mosaic.",
            "Shadows lengthened as the afternoon sun descended toward the horizon.",
            "The opalescent surface shimmered with rainbow refractions.",
            "Perspective lines converged at the vanishing point on the horizon.",
            "The fractal pattern repeated at every scale of magnification.",
            "The iridescent butterfly wings displayed vibrant blue and green hues.",
            "Light refracted through the crystal prism separating white into rainbow colors.",
            "The abstract painting juxtaposed bold geometric shapes with organic curves.",
            "The marbled surface swirled with intertwined veins of color.",
            "The kaleidoscope pattern shifted with each rotation revealing new symmetries.",
            "The chiaroscuro technique contrasted brilliant highlights with deep shadows.",
            "The tessellation covered the plane without gaps or overlaps.",
            "The photograph captured the motion blur of the speeding train.",
            "The gradient transitioned smoothly from deep purple to pale lavender.",
            "The architectural facade featured ornate sculptural decorations.",
            "The pointillist painting built images from tiny dots of pure color.",
            "The stained glass window filtered colored light into the dark interior.",
            "The impressionist brushwork captured fleeting impressions of light.",
            "The optical illusion created depth from flat two-dimensional surface."
        ]
    },
    "olfactory": {
        "expected": ["G7_sensory", "G6_limbic"],
        "texts": [
            "The lavender fields released their sweet floral perfume into the warm air.",
            "Freshly brewed coffee filled the kitchen with its rich roasted aroma.",
            "The pine forest exhaled crisp resiny scent after the morning rain.",
            "The bakery emanated warm vanilla and cinnamon fragrance.",
            "The rose garden perfumed the evening breeze with romantic sweetness.",
            "Fresh cut grass released its distinctive green scent into summer air.",
            "The ocean breeze carried briny seaweed and salty sea spray.",
            "The truffles emitted their distinctive earthy pungent aroma.",
            "The spice market overwhelmed the senses with cinnamon, cumin, and cardamom.",
            "The cedar closet smelled of aromatic wood preservation.",
            "The lemon tree perfumed the patio with citrus blossom fragrance.",
            "The incense smoke curled upward releasing sandalwood and jasmine.",
            "The perfume bottle held concentrated aromatic essential oils.",
            "The damp earth after rain released petrichor, its distinctive scent.",
            "The tobacco pipe emitted sweet aromatic smoke into the study.",
            "The jasmine vine climbed the trellis scenting the entire garden.",
            "The garlic and herbs sizzling in olive oil filled the air.",
            "The vanilla orchid produced beans with intense sweet fragrance.",
            "The mint leaves crushed between fingers released refreshing oil.",
            "The old books emanated distinctive musty paper and leather smell."
        ]
    },
    "gustatory": {
        "expected": ["G7_sensory", "G6_limbic"],
        "texts": [
            "The dark chocolate melted smoothly on the tongue with bitter sweetness.",
            "Fresh strawberries burst with concentrated summer sweetness.",
            "The aged parmesan cheese offered sharp crystalline texture and nutty flavor.",
            "Lemon juice provided bright acidic tang cutting through richness.",
            "The caramelized onions developed sweet complexity through browning.",
            "Fresh basil leaves contributed peppery minty notes to the salad.",
            "The umami-rich mushrooms deepened the savory flavor of the sauce.",
            "The honey drizzle added floral sweetness to the yogurt.",
            "The wasabi burned the sinuses with its sharp horseradish heat.",
            "The ripe peach juices ran down the chin with sticky sweetness.",
            "The espresso shot delivered intense bitter coffee complexity.",
            "The blue cheese crumble provided salty pungent funky notes.",
            "The maple syrup poured thick and amber with brown sugar sweetness.",
            "The fresh ginger provided warm peppery heat to the stir-fry.",
            "The sea salt enhanced the natural sweetness of the caramel.",
            "The truffle oil drizzled over pasta added earthy luxurious aroma.",
            "The red wine exhibited tannins creating drying sensation on the palate.",
            "The hot chili peppers delivered capsaicin burning sensation.",
            "The vanilla bean custard achieved silky smooth texture and floral sweetness.",
            "The fermented kimchi provided sour spicy complex probiotic flavors."
        ]
    },
    "proprioceptive": {
        "expected": ["G9_premotor", "G7_sensory"],
        "texts": [
            "She felt her weight shift onto her back foot as she prepared to throw.",
            "The dancer sensed her spine lengthening as she reached toward the ceiling.",
            "He adjusted his balance feeling the subtle tension in his ankles.",
            "The yoga instructor guided awareness of body position in each pose.",
            "She felt her shoulders drop away from her ears releasing neck tension.",
            "The gymnast calibrated the exact rotation through kinesthetic awareness.",
            "He perceived the subtle rotation of his pelvis during the walking stride.",
            "The pianist developed finger independence through proprioceptive training.",
            "She sensed her core engaging as she lifted the heavy box.",
            "The martial artist felt the weight transfer through each step of the form.",
            "He perceived the stretch receptors firing as he extended his hamstrings.",
            "The surgeon's steady hands reflected refined proprioceptive control.",
            "She balanced on one foot sensing micro-adjustments in her ankle joints.",
            "The athlete visualized muscle contractions during mental rehearsal.",
            "He felt his jaw clench unconsciously under stressful situations.",
            "The climber tested each handhold feeling the friction and stability.",
            "She perceived the subtle warmth spreading through her relaxed muscles.",
            "The pianist's fingers danced across keys with automatic spatial awareness.",
            "He sensed his breathing synchronize with the rhythm of his running stride.",
            "The dancer felt the floor beneath her feet providing grounding support."
        ]
    },
    "spatial_nav": {
        "expected": ["G7_sensory", "G4_pfc"],
        "texts": [
            "The compass needle always points toward magnetic north.",
            "He navigated the maze by remembering left and right turns.",
            "The map showed the trail winding north through dense forest.",
            "She oriented herself using the sun position in the afternoon sky.",
            "The GPS coordinates pinpointed the exact location on Earth.",
            "He estimated distance by counting his footsteps along the path.",
            "The city grid followed a strict rectangular pattern of streets.",
            "She used landmarks to create a mental map of the unfamiliar neighborhood.",
            "The satellite imagery revealed terrain elevation changes across the region.",
            "He walked three blocks east then turned south toward the river.",
            "The constellation Orion served as navigation guide for ancient sailors.",
            "She visualized the building layout from the floor plan diagram.",
            "The airport terminal signs directed travelers to their gates.",
            "He triangulated his position using three visible mountain peaks.",
            "The crossword grid required filling words crossing both directions.",
            "She memorized the route by associating turns with distinctive buildings.",
            "The telescope focused light from distant stars onto the detector.",
            "He charted the course plotting waypoints on the nautical chart.",
            "The maze solver used algorithmic exploration to find the exit.",
            "She sensed direction without visual cues through vestibular awareness."
        ]
    },
    "temporal": {
        "expected": ["G5_dmn", "G4_pfc"],
        "texts": [
            "The pendulum swung back and forth marking seconds in precise rhythm.",
            "She traced the timeline from ancient civilizations to modern era.",
            "The hourglass measured elapsed time through flowing sand grains.",
            "He recalled the sequence of events leading to the fateful decision.",
            "The clock tower chimed the quarter hour throughout the day.",
            "She experienced déjà vu feeling the present moment somehow familiar.",
            "The calendar marked important dates across the passing seasons.",
            "He estimated five minutes elapsed while waiting for the response.",
            "The historical narrative spanned centuries of human civilization.",
            "She anticipated tomorrow's meeting while completing today's tasks.",
            "The metronome established steady tempo for practicing musicians.",
            "He reflected on past experiences to inform present decisions.",
            "The archaeological site revealed layers from different historical periods.",
            "She perceived time moving faster as she grew older.",
            "The sundial shadow moved steadily tracking the sun's apparent motion.",
            "He juggled multiple deadlines competing for his immediate attention.",
            "The film editor arranged scenes in proper chronological sequence.",
            "She experienced flow as hours passed unnoticed during deep focus.",
            "The retrospective examined company achievements over the past decade.",
            "He felt the weight of time pressing on his shoulders."
        ]
    },
    "narrative": {
        "expected": ["G5_dmn", "G2_wernicke"],
        "texts": [
            "The protagonist embarked on a journey seeking hidden treasure.",
            "She discovered the mysterious letter tucked inside the old book.",
            "The hero overcame impossible obstacles through courage and cunning.",
            "He unraveled the conspiracy threatening the peaceful kingdom.",
            "The detective pieced together clues to solve the puzzling case.",
            "She confronted her past to find resolution and peace.",
            "The villain's true motives remained hidden until the dramatic reveal.",
            "He faced the ultimate choice between duty and personal desire.",
            "The story unfolded through multiple perspectives and timelines.",
            "She transformed from naive innocent to wise experienced survivor.",
            "The prophecy foretold the chosen one would save the realm.",
            "He sacrificed everything for the greater good of the community.",
            "The narrative explored themes of love, loss, and redemption.",
            "She outwitted the antagonist using intelligence rather than force.",
            "The tale ended with unexpected twist changing everything.",
            "He embraced his destiny despite fear and uncertainty.",
            "The legend passed down through generations carried wisdom.",
            "She found unexpected ally among former enemies.",
            "The mystery deepened with each new revelation.",
            "He learned the most important lesson through painful experience."
        ]
    },
    "procedural": {
        "expected": ["G4_pfc", "G1_broca"],
        "texts": [
            "First, preheat the oven to three hundred fifty degrees Fahrenheit.",
            "The recipe instructs you to fold the batter until just combined.",
            "To assemble the furniture, insert the dowels into the predrilled holes.",
            "The manual details step-by-step instructions for software installation.",
            "Begin by securing the anchor bolts in the concrete foundation.",
            "Mix the dry ingredients separately before adding to wet mixture.",
            "The tutorial demonstrates how to tie the complex knot correctly.",
            "Set the thermostat to cooling mode and adjust target temperature.",
            "The protocol requires calibrating the equipment before each use.",
            "Insert the SIM card into the slot with gold contacts facing down.",
            "Apply even pressure while smoothing the adhesive surface.",
            "The procedure involves three sequential steps for proper sterilization.",
            "Connect the power cable to the designated input port.",
            "Align the pattern pieces before cutting the fabric.",
            "The algorithm processes input data through defined stages.",
            "Substitute the expired ingredients with fresh alternatives.",
            "Configure the settings according to the specified parameters.",
            "Perform the diagnostic test to verify system functionality.",
            "The instructions recommend resting the dough for one hour.",
            "Allow the primer to dry completely before applying topcoat."
        ]
    },
    "linguistic": {
        "expected": ["G2_wernicke", "G1_broca"],
        "texts": [
            "The verb conjugation changes based on subject person and number.",
            "Syntactic rules govern how words combine into grammatical phrases.",
            "The phoneme represents the smallest distinctive unit of sound.",
            "Semantics studies meaning conveyed through language structures.",
            "The adjective modifies a noun providing descriptive information.",
            "Pragmatics examines how context influences language interpretation.",
            "The syllable comprises vowel sound surrounded by consonant sounds.",
            "Morphology analyzes internal structure of words and their formation.",
            "The declarative sentence makes a statement or assertion.",
            "Discourse analysis studies connected text beyond the sentence.",
            "The preposition indicates spatial or temporal relationship.",
            "Register varies based on formal versus informal context.",
            "The article specifies definiteness as the or introduces new reference.",
            "Paralinguistic features include tone, pitch, and pacing.",
            "The coordinating conjunction links equal grammatical elements.",
            "Code-switching alternates between languages in conversation.",
            "The metaphor transfers meaning through symbolic comparison.",
            "Pidgin languages emerge from contact between different speech communities.",
            "The interrogative pronoun requests information from the listener.",
            "Prosody shapes rhythm and emphasis across the utterance."
        ]
    },
    "social_relational": {
        "expected": ["G3_tpj", "G5_dmn"],
        "texts": [
            "She confided her deepest fears to her trusted best friend.",
            "The family gathered around the table for Sunday dinner.",
            "He navigated the complex political dynamics of the office.",
            "The mentor provided guidance shaping her career trajectory.",
            "She negotiated the terms of the agreement with the supplier.",
            "The colleagues collaborated on the challenging project together.",
            "He mediated the dispute between the conflicting parties.",
            "The friendship deepened through shared experiences and mutual support.",
            "She introduced herself to the unfamiliar neighbors on the street.",
            "The couple celebrated their twentieth wedding anniversary.",
            "He respected the cultural traditions of his wife's family.",
            "The teacher inspired students through passionate dedication.",
            "She apologized sincerely for the misunderstanding and its consequences.",
            "The community came together supporting the struggling family.",
            "He trusted his lawyer to handle the legal proceedings.",
            "The therapist helped the patient explore childhood relationships.",
            "She forgave the betrayal choosing reconciliation over resentment.",
            "The team captain motivated members during the difficult season.",
            "He acknowledged the error and thanked colleagues for feedback.",
            "The support group provided safe space for sharing struggles."
        ]
    },
    "economic": {
        "expected": ["G4_pfc", "G5_dmn"],
        "texts": [
            "The inflation rate rose three percent over the previous year.",
            "Supply and demand curves intersect at equilibrium price point.",
            "The central bank adjusted interest rates to control borrowing.",
            "The entrepreneur secured venture capital funding for expansion.",
            "Fiscal policy influences economic activity through government spending.",
            "The stock market index reached record high closing value.",
            "International trade agreements affect tariff rates and market access.",
            "The budget deficit exceeded projections requiring spending cuts.",
            "Monetary policy manages money supply and credit availability.",
            "The currency exchange rate fluctuated based on economic indicators.",
            "Consumer confidence index predicts spending patterns.",
            "The merger created the largest corporation in the industry.",
            "Unemployment rate declined as job market strengthened.",
            "The trade balance measures imports and exports difference.",
            "Gross domestic product measures total economic output.",
            "The startup valued at billion dollars during funding round.",
            "Supply chain disruptions affected production and delivery timelines.",
            "The recession officially ended after eighteen months.",
            "Foreign investment flowed into emerging markets seeking returns.",
            "The antitrust regulators prevented the monopolistic acquisition."
        ]
    },
    "political": {
        "expected": ["G4_pfc", "G3_tpj"],
        "texts": [
            "The legislative assembly debated the controversial bill for weeks.",
            "The constitution guarantees fundamental rights to all citizens.",
            "The electoral college determined the presidential election outcome.",
            "International diplomacy seeks peaceful resolution to the conflict.",
            "The judicial review upheld the constitutionality of the statute.",
            "Grassroots activism mobilized voters across the district.",
            "The political party platform outlined policy priorities and positions.",
            "The ambassador represented national interests in foreign negotiations.",
            "Campaign finance regulations limit contributions to candidates.",
            "The lobbyist advocated for industry interests before congress.",
            "The ballot initiative allowed direct voter participation in lawmaking.",
            "The impeachment process held officials accountable for misconduct.",
            "Civic engagement encourages citizen participation in democracy.",
            "The census determines congressional representation allocation.",
            "The treaty required Senate ratification to take effect.",
            "The bureaucracy implements policies through administrative procedures.",
            "The referendum gave citizens voice on specific policy question.",
            "Separation of powers divides government into distinct branches.",
            "The public hearing allowed community input on proposed regulation.",
            "Federalism balances power between national and state governments."
        ]
    },
    "artistic": {
        "expected": ["G5_dmn", "G3_tpj"],
        "texts": [
            "The sculptor shaped cold marble into graceful human form.",
            "She composed symphony expressing emotions beyond verbal language.",
            "The photographer captured decisive moment revealing deeper truth.",
            "The painter layered translucent glazes building luminous depth.",
            "He choreographed dance movements expressing joy and sorrow.",
            "The poet crafted verses resonating with universal human experience.",
            "The architect designed sustainable building harmonizing with environment.",
            "She improvised jazz solos spontaneously creating new melodies.",
            "The ceramicist fired clay at high temperature creating durable vessel.",
            "He directed film sequences manipulating audience emotions masterfully.",
            "The calligrapher brushed elegant characters following traditional forms.",
            "She mixed pigments creating custom colors for the mural.",
            "The weaver interlaced threads forming intricate textile pattern.",
            "He performed theater creating empathetic connection with audience.",
            "The designer balanced form and function in product development.",
            "She captured likeness through careful observation in portrait session.",
            "The glassblower shaped molten material into delicate vessel.",
            "He composed electronic music layering synthesized sounds.",
            "The dancer embodied character through precise movement vocabulary.",
            "She restored damaged masterpiece returning it to original beauty."
        ]
    },
    "scientific": {
        "expected": ["G4_pfc", "G7_sensory"],
        "texts": [
            "The hypothesis predicts relationship between variables under investigation.",
            "Peer review ensures published research meets scientific standards.",
            "The experiment controls variables while manipulating independent factor.",
            "Replication verifies findings across different researchers and settings.",
            "The null hypothesis assumes no significant effect or relationship.",
            "Statistical significance indicates results unlikely due to chance alone.",
            "The methodology section details procedures for reproducing the study.",
            "Peer-reviewed journal publishes validated scientific discoveries.",
            "The theory integrates multiple observations into coherent framework.",
            "Experimental controls eliminate alternative explanations for results.",
            "The literature review surveys existing knowledge on the topic.",
            "Reproducibility crisis prompted scrutiny of research practices.",
            "The correlation coefficient measures strength of linear relationship.",
            "Blind experimental design reduces bias in subjective assessment.",
            "The abstract summarizes key methods and major findings concisely.",
            "Meta-analysis combines results from multiple studies statistically.",
            "The researcher formulated predictions based on theoretical framework.",
            "Ethics committee reviewed human subjects protections.",
            "The data analysis employed appropriate statistical tests.",
            "Independent verification confirmed the groundbreaking discovery."
        ]
    },
    "historical": {
        "expected": ["G5_dmn", "G2_wernicke"],
        "texts": [
            "The Roman Empire collapsed in the fifth century CE.",
            "The Renaissance revived classical learning during fifteenth century.",
            "Ancient Egyptians built pyramids as monumental tombs for pharaohs.",
            "The Industrial Revolution transformed agrarian society mechanizing production.",
            "World War II reshaped global political boundaries and alliances.",
            "The French Revolution overthrew monarchy establishing republic.",
            "The Silk Road connected East and West facilitating trade and cultural exchange.",
            "The American Revolution established independent democratic nation.",
            "The Black Death devastated European population in fourteenth century.",
            "The Scientific Revolution challenged traditional authorities and beliefs.",
            "The Cold War divided world between competing superpowers.",
            "The Digital Revolution transformed communication and information access.",
            "The abolition movement ended transatlantic slave trade in nineteenth century.",
            "The Bronze Age preceded Iron Age introducing metal tools and weapons.",
            "The Enlightenment promoted reason and individual rights in eighteenth century.",
            "The exploration age discovered new continents and oceanic routes.",
            "The Reformation broke Christian church unity creating Protestant denominations.",
            "The fall of Constantinople ended Byzantine Empire in fifteenth century.",
            "The civil rights movement achieved legal equality for marginalized groups.",
            "The agricultural revolution preceded urbanization concentrating populations."
        ]
    },
    "religious": {
        "expected": ["G6_limbic", "G5_dmn"],
        "texts": [
            "The pilgrimage to sacred site fulfilled spiritual obligation.",
            "The meditation practice cultivated mindfulness and inner peace.",
            "The congregation gathered for weekly worship and communal prayers.",
            "The scripture contained divine revelations guiding moral conduct.",
            "The monastery sheltered monks pursuing contemplative life.",
            "The baptism ceremony initiated new member into faith community.",
            "The prophet delivered messages from divine realm to humanity.",
            "The funeral ritual honored departed soul journey to afterlife.",
            "The sacred dance expressed devotion through choreographed movement.",
            "The fasting period purified body and spirit during holy season.",
            "The temple architecture reflected cosmic order and divine presence.",
            "The confession absolved sins restoring relationship with divine.",
            "The hymn lifted hearts praising transcendent being.",
            "The wedding ceremony united couple under sacred covenant.",
            "The blessing invoked divine favor upon person or undertaking.",
            "The relic preserved physical remains of saint or holy figure.",
            "The meditation retreat intensified practice through sustained silence.",
            "The ordination ceremony authorized new religious leaders.",
            "The creation myth explained origins of world and humanity.",
            "The resurrection celebrated triumph over death and darkness."
        ]
    }
}

def main():
    output_path = "/home/student/sgp-tribe3/data/expanded_stimulus_bank_v2.json"
    
    with open(output_path, 'w') as f:
        json.dump(NEW_CATEGORIES, f, indent=2)
    
    total = sum(len(cat["texts"]) for cat in NEW_CATEGORIES.values())
    print(f"Generated {len(NEW_CATEGORIES)} new categories with {total} stimuli")
    print(f"Saved to: {output_path}")
    
    for cat_name, cat_data in NEW_CATEGORIES.items():
        print(f"  - {cat_name}: {len(cat_data['texts'])} stimuli, expected nodes: {cat_data['expected']}")

if __name__ == "__main__":
    main()
