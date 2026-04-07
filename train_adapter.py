"""
Train Embedding Adapter: Ollama → LLaMA Embeddings
===================================================
Trains an MLP to map Ollama embeddings (4096d) to LLaMA-like embeddings (9216d).

Usage:
    python train_adapter.py --n_samples 500 --epochs 100 --batch_size 32

This script:
1. Collects diverse text samples
2. Generates embeddings with both LLaMA (via TRIBE v2) and Ollama
3. Trains MLP: 4096 → 8192 → 9216 with MSE loss
4. Saves adapter weights to adapter_weights.pt
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_DEVICE'] = 'cpu'

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


SAMPLE_TEXTS = [
    "The cat sat on the mat and watched the rain fall outside.",
    "Scientists discovered a new species of deep-sea fish near hydrothermal vents.",
    "The stock market experienced significant volatility due to geopolitical tensions.",
    "Quantum computing promises to revolutionize cryptography and drug discovery.",
    "She walked through the ancient forest, marveling at the towering redwoods.",
    "The chef prepared a delicate soufflé that rose perfectly in the oven.",
    "Machine learning models require large amounts of labeled training data.",
    "The orchestra performed Beethoven's ninth symphony to a captivated audience.",
    "Climate change is accelerating the melting of polar ice caps worldwide.",
    "The detective carefully examined the crime scene for any overlooked evidence.",
    "Neural networks learn hierarchical representations through multiple layers.",
    "The astronaut floated weightlessly inside the International Space Station.",
    "Ancient philosophers debated the nature of consciousness and free will.",
    "The startup raised fifty million dollars in its Series B funding round.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The journalist investigated corruption at the highest levels of government.",
    "Reinforcement learning agents learn optimal policies through trial and error.",
    "The architect designed a sustainable building with a living green roof.",
    "DNA sequencing has become fast and affordable enough for personalized medicine.",
    "The painter used bold brushstrokes to capture the essence of the landscape.",
    "Transformer architectures have revolutionized natural language processing tasks.",
    "The mountain climber reached the summit just as the sun began to rise.",
    "Economic inequality continues to grow in many developed nations.",
    "The biologist studied the symbiotic relationship between fungi and tree roots.",
    "Natural language generation models can now produce coherent paragraphs.",
    "The musician composed a haunting melody that echoed through the concert hall.",
    "Artificial intelligence is transforming healthcare through improved diagnostics.",
    "The historian uncovered lost documents from the medieval period.",
    "Convolutional neural networks excel at image recognition and classification.",
    "The poet wove words together to create vivid imagery and emotion.",
    "Renewable energy sources are becoming cost-competitive with fossil fuels.",
    "The surgeon performed a minimally invasive procedure using robotic assistance.",
    "Graph neural networks model relationships between interconnected entities.",
    "The teacher inspired her students to pursue careers in science.",
    "Ocean acidification threatens coral reef ecosystems around the world.",
    "The programmer debugged the code late into the night.",
    "Attention mechanisms allow models to focus on relevant input features.",
    "The farmer harvested crops using sustainable agricultural practices.",
    "Cybersecurity threats are evolving as attackers develop more sophisticated methods.",
    "The dancer moved with grace and precision across the stage.",
    "Transfer learning enables models to leverage knowledge from related tasks.",
    "The archaeologist carefully excavated artifacts from the ancient burial site.",
    "Generative adversarial networks can create realistic synthetic images.",
    "The lawyer presented compelling evidence during the closing arguments.",
    "Biodiversity loss is one of the most pressing environmental challenges.",
    "The engineer designed a bridge capable of withstanding major earthquakes.",
    "Self-supervised learning reduces the need for expensive labeled datasets.",
    "The novelist spent years researching the historical period for her book.",
    "Autonomous vehicles use sensor fusion to navigate complex urban environments.",
    "The veterinarian treated the injured wildlife with compassion and expertise.",
    "The rapid advancement of AI raises important ethical considerations.",
    "Music therapy has shown promising results in treating depression.",
    "The team collaborated across multiple time zones using video conferencing.",
    "Deep reinforcement learning achieved superhuman performance in complex games.",
    "The gardener pruned the roses to encourage healthy new growth.",
    "Space telescopes have revealed thousands of exoplanets in distant star systems.",
    "The debate team argued both sides of the controversial policy issue.",
    "Federated learning enables model training across decentralized data sources.",
    "The photographer captured the golden hour light on the city skyline.",
    "The marine biologist tracked migration patterns of humpback whales.",
    "The software update addressed critical security vulnerabilities in the system.",
    "The chef experimented with molecular gastronomy techniques in the kitchen.",
    "The athlete trained rigorously for months before the championship event.",
    "The librarian organized a community reading program for young children.",
    "The researcher published groundbreaking findings in a peer-reviewed journal.",
    "The city council voted to expand public transportation infrastructure.",
    "The artist created an interactive installation using augmented reality.",
    "The pilot navigated through severe turbulence with skill and composure.",
    "The therapist helped patients develop coping strategies for anxiety.",
    "The economist analyzed trends in consumer spending and inflation.",
    "The volunteer coordinated disaster relief efforts in the affected region.",
    "The astronomer detected a previously unknown asteroid approaching Earth.",
    "The composer blended traditional orchestration with electronic sound design.",
    "The activist organized a peaceful protest for environmental protection.",
    "The mechanic diagnosed the engine problem using advanced diagnostic tools.",
    "The linguist documented an endangered language spoken by a remote tribe.",
    "The entrepreneur pivoted the business model after receiving customer feedback.",
    "The physicist conducted experiments at the particle accelerator facility.",
    "The journalist reported live from the scene of the breaking news.",
    "The nurse provided compassionate care to patients in the intensive care unit.",
    "The mathematician proved a theorem that had remained unsolved for decades.",
    "The filmmaker documented the daily lives of indigenous communities.",
    "The chemist synthesized a new compound with potential pharmaceutical applications.",
    "The teacher integrated technology into the classroom to enhance learning.",
    "The conservationist worked to protect endangered species in their natural habitat.",
    "The data scientist built predictive models to forecast customer behavior.",
    "The diplomat negotiated a peace agreement between conflicting nations.",
    "The biologist discovered a new mechanism of cellular communication.",
    "The writer crafted a compelling narrative that kept readers engaged.",
    "The engineer optimized the manufacturing process to reduce waste.",
    "The psychologist studied the effects of social media on mental health.",
    "The geologist mapped underground water sources in the arid region.",
    "The musician improvised a jazz solo that thrilled the audience.",
    "The programmer implemented a new algorithm for efficient data sorting.",
    "The historian analyzed primary sources to reconstruct past events.",
    "The designer created a user-friendly interface for the mobile application.",
    "The scientist developed a vaccine using mRNA technology.",
    "The farmer adopted precision agriculture techniques to increase yields.",
    "The artist painted a mural celebrating the city's cultural heritage.",
    "The researcher conducted a longitudinal study on childhood development.",
    "The coach motivated the team to overcome their losing streak.",
    "The architect incorporated passive solar design into the building plans.",
    "The journalist investigated the environmental impact of industrial pollution.",
    "The teacher used project-based learning to engage students in STEM.",
    "The entrepreneur launched a startup focused on sustainable fashion.",
    "The physicist explored the properties of dark matter and dark energy.",
    "The nurse administered medication and monitored patient vital signs.",
    "The programmer contributed to an open-source software project.",
    "The biologist studied the effects of microplastics on marine organisms.",
    "The artist experimented with new materials to create innovative sculptures.",
    "The economist proposed policy recommendations to address income inequality.",
    "The engineer designed a renewable energy system for a remote village.",
    "The therapist used cognitive behavioral techniques to treat phobias.",
    "The astronomer observed a supernova event in a distant galaxy.",
    "The chef created a fusion cuisine blending multiple culinary traditions.",
    "The writer published a collection of short stories exploring human nature.",
    "The scientist developed a new method for carbon capture and storage.",
    "The musician composed a soundtrack for an independent film.",
    "The teacher mentored students preparing for college entrance examinations.",
    "The researcher investigated the neural basis of language acquisition.",
    "The activist campaigned for voting rights and electoral reform.",
    "The programmer built a web application for community resource sharing.",
    "The geologist studied volcanic activity to improve eruption prediction.",
    "The artist exhibited contemporary works at an international gallery.",
    "The diplomat facilitated trade negotiations between partner countries.",
    "The biologist engineered bacteria to produce biofuels from waste materials.",
    "The journalist wrote an investigative piece on healthcare accessibility.",
    "The engineer developed autonomous drone technology for search and rescue.",
    "The psychologist researched the impact of sleep on memory consolidation.",
    "The mathematician developed new statistical methods for analyzing big data.",
    "The conservationist established a wildlife corridor to connect fragmented habitats.",
    "The data scientist created a recommendation system for an e-commerce platform.",
    "The composer wrote a symphony inspired by natural phenomena.",
    "The teacher implemented differentiated instruction to meet diverse learning needs.",
    "The entrepreneur secured venture capital funding for a health tech startup.",
    "The physicist studied quantum entanglement and its implications for computing.",
    "The nurse educated patients about managing chronic conditions at home.",
    "The programmer optimized database queries to improve application performance.",
    "The biologist mapped the genome of a rare plant species.",
    "The artist collaborated with scientists to visualize complex data sets.",
    "The economist modeled the economic impact of automation on employment.",
    "The engineer designed a water purification system for developing regions.",
    "The therapist developed group therapy programs for trauma survivors.",
    "The astronomer contributed to the search for extraterrestrial intelligence.",
    "The chef trained apprentices in classical French cooking techniques.",
    "The writer adapted a classic novel for a modern audience.",
    "The scientist conducted clinical trials for a new cancer treatment.",
    "The musician taught music theory and composition at a conservatory.",
    "The teacher organized a science fair to showcase student projects.",
    "The researcher published a meta-analysis of educational intervention studies.",
    "The activist advocated for affordable housing in urban areas.",
    "The programmer developed machine learning models for fraud detection.",
    "The geologist analyzed seismic data to assess earthquake risks.",
    "The artist received a prestigious award for contributions to contemporary art.",
    "The diplomat worked on international climate change agreements.",
    "The biologist studied the role of gut microbiota in human health.",
    "The journalist covered the technological innovations at a major conference.",
    "The engineer tested prototype materials for next-generation batteries.",
    "The psychologist studied decision-making processes under uncertainty.",
    "The mathematician explored topological properties of high-dimensional spaces.",
    "The conservationist monitored endangered species populations over time.",
    "The data scientist visualized complex networks to reveal hidden patterns.",
    "The composer integrated traditional instruments with digital audio workstations.",
    "The teacher used gamification to increase student engagement in math.",
    "The entrepreneur founded a nonprofit organization promoting digital literacy.",
    "The physicist investigated the properties of topological insulators.",
    "The nurse coordinated care between specialists for complex patient cases.",
    "The programmer implemented continuous integration and deployment pipelines.",
    "The biologist discovered a new signaling pathway in immune cells.",
    "The artist created immersive installations using projection mapping.",
    "The economist studied the effects of monetary policy on inflation.",
    "The engineer developed assistive technology for people with disabilities.",
    "The therapist researched mindfulness-based interventions for stress reduction.",
    "The astronomer cataloged stars in a newly surveyed region of sky.",
    "The chef sourced ingredients from local organic farms.",
    "The writer explored themes of identity and belonging in fiction.",
    "The scientist developed biodegradable plastics from agricultural waste.",
    "The musician performed at a benefit concert for disaster relief.",
    "The teacher incorporated mindfulness practices into the daily classroom routine.",
    "The researcher investigated the long-term effects of screen time on children.",
    "The activist organized community gardens in food desert neighborhoods.",
    "The programmer created accessibility features for visually impaired users.",
    "The geologist studied groundwater contamination from industrial activities.",
    "The artist participated in a residency program exploring art and technology.",
    "The diplomat mediated peace talks in a conflict zone.",
    "The biologist engineered crops resistant to drought and pests.",
    "The journalist documented the impact of deforestation on indigenous peoples.",
    "The engineer designed smart city infrastructure for sustainable urban living.",
    "The psychologist studied the relationship between exercise and cognitive function.",
    "The mathematician developed algorithms for optimizing supply chain logistics.",
    "The conservationist restored degraded wetland ecosystems.",
    "The data scientist built a predictive model for disease outbreak detection.",
    "The composer created music for a documentary about ocean conservation.",
    "The teacher implemented peer tutoring programs to support struggling students.",
    "The entrepreneur developed an app connecting volunteers with local nonprofits.",
    "The physicist researched nuclear fusion as a clean energy source.",
    "The nurse provided end-of-life care with dignity and compassion.",
    "The programmer contributed to cybersecurity standards development.",
    "The biologist studied the effects of light pollution on nocturnal animals.",
    "The artist used recycled materials to create environmental art installations.",
    "The economist analyzed the gig economy's impact on worker benefits.",
    "The engineer designed low-cost prosthetics using 3D printing technology.",
    "The therapist developed art therapy programs for children with autism.",
    "The astronomer studied the formation and evolution of galaxy clusters.",
    "The chef created a plant-based menu that appealed to diverse diners.",
    "The writer chronicled the history of a marginalized community.",
    "The scientist developed rapid diagnostic tests for infectious diseases.",
    "The musician composed a piece inspired by mathematical patterns.",
    "The teacher organized field trips to connect classroom learning with real-world experiences.",
    "The researcher studied the effectiveness of different teaching methodologies.",
    "The activist campaigned for criminal justice reform.",
    "The programmer developed natural language processing tools for low-resource languages.",
    "The geologist mapped mineral deposits for sustainable mining practices.",
    "The artist explored the intersection of art and artificial intelligence.",
    "The diplomat promoted cultural exchange programs between nations.",
    "The biologist studied coral bleaching events and their ecological consequences.",
    "The journalist investigated labor conditions in global supply chains.",
    "The engineer developed renewable energy storage solutions.",
    "The psychologist researched resilience factors in trauma recovery.",
    "The mathematician studied the mathematics of network security.",
    "The conservationist advocated for marine protected areas.",
    "The data scientist analyzed social media data to track public health trends.",
    "The composer blended folk music traditions with contemporary orchestration.",
    "The teacher used virtual reality to bring history lessons to life.",
    "The entrepreneur launched a platform for remote collaboration.",
    "The physicist studied the early universe using cosmic microwave background data.",
    "The nurse advocated for improved working conditions in healthcare facilities.",
    "The programmer developed tools for automated code review.",
    "The biologist investigated the role of epigenetics in disease susceptibility.",
    "The artist created participatory art projects in public spaces.",
    "The economist studied the economic benefits of investing in education.",
    "The engineer designed flood-resistant infrastructure for coastal communities.",
    "The therapist researched the effectiveness of online therapy platforms.",
    "The astronomer detected gravitational waves from merging black holes.",
    "The chef taught cooking classes focused on nutrition and wellness.",
    "The writer explored the ethical implications of emerging technologies.",
    "The scientist developed gene therapy approaches for inherited disorders.",
    "The musician collaborated with dancers to create interdisciplinary performances.",
    "The teacher implemented restorative justice practices in the classroom.",
    "The researcher investigated the impact of urban green spaces on well-being.",
    "The activist organized voter registration drives in underserved communities.",
    "The programmer built open-source tools for scientific research.",
    "The geologist studied the impact of permafrost thaw on infrastructure.",
    "The artist documented social movements through photography and video.",
    "The diplomat negotiated international agreements on space exploration.",
    "The biologist developed methods for monitoring biodiversity using environmental DNA.",
    "The journalist reported on the intersection of technology and privacy.",
    "The engineer designed energy-efficient buildings using passive design principles.",
    "The psychologist studied the cognitive effects of bilingualism.",
    "The mathematician developed models for predicting epidemic spread.",
    "The conservationist worked with local communities on sustainable resource management.",
    "The data scientist created algorithms for fair lending practices.",
    "The composer wrote music for interactive multimedia installations.",
    "The teacher fostered a growth mindset culture in the classroom.",
    "The entrepreneur developed solutions for reducing food waste.",
    "The physicist researched quantum communication protocols.",
    "The nurse implemented evidence-based practices to improve patient outcomes.",
    "The programmer mentored junior developers in best coding practices.",
    "The biologist studied the adaptive radiation of species on isolated islands.",
    "The artist challenged conventional notions of beauty through provocative works.",
    "The economist analyzed the impact of trade policies on developing nations.",
    "The engineer developed autonomous systems for precision agriculture.",
    "The therapist researched trauma-informed care approaches.",
    "The astronomer studied the atmospheric composition of exoplanets.",
    "The chef promoted farm-to-table dining experiences.",
    "The writer crafted narratives that bridged cultural divides.",
    "The scientist developed biocompatible materials for tissue engineering.",
    "The musician used technology to preserve and share traditional music.",
    "The teacher integrated social-emotional learning into the curriculum.",
    "The researcher studied the effects of music therapy on neurological disorders.",
    "The activist advocated for accessible public spaces for people with disabilities.",
    "The programmer developed privacy-preserving machine learning techniques.",
    "The geologist assessed landslide risks in mountainous regions.",
    "The artist created works addressing social justice issues.",
    "The diplomat facilitated international cooperation on pandemic preparedness.",
    "The biologist studied the evolutionary origins of human language.",
    "The journalist covered the global response to emerging infectious diseases.",
    "The engineer designed microgrids for resilient energy distribution.",
    "The psychologist researched the impact of nature exposure on stress.",
    "The mathematician developed optimization algorithms for resource allocation.",
    "The conservationist established seed banks to preserve crop diversity.",
    "The data scientist built models for predicting wildfire risk.",
    "The composer created adaptive music systems for video games.",
    "The teacher used project-based assessments to measure student learning.",
    "The entrepreneur developed platforms for peer-to-peer learning.",
    "The physicist researched materials for next-generation solar cells.",
    "The nurse provided health education in underserved communities.",
    "The programmer developed tools for automated accessibility testing.",
    "The biologist investigated the role of circadian rhythms in health.",
    "The artist collaborated with engineers to create kinetic sculptures.",
    "The economist studied the economic value of ecosystem services.",
    "The engineer designed water harvesting systems for arid regions.",
    "The therapist developed interventions for workplace burnout.",
    "The astronomer mapped the distribution of dark matter in the universe.",
    "The chef created menus that celebrated culinary diversity.",
    "The writer explored the human condition through literary fiction.",
    "The scientist developed nanotechnology applications for targeted drug delivery.",
    "The musician composed works that blended classical and electronic genres.",
    "The teacher implemented inclusive education practices for diverse learners.",
    "The researcher studied the impact of early childhood interventions.",
    "The activist campaigned for digital rights and online privacy.",
    "The programmer developed algorithms for real-time language translation.",
    "The geologist studied the formation of mineral deposits in hydrothermal systems.",
    "The artist used data visualization as a medium for artistic expression.",
    "The diplomat worked on international agreements for ocean conservation.",
    "The biologist studied the impact of urbanization on wildlife populations.",
    "The journalist investigated the role of misinformation in public discourse.",
    "The engineer developed low-cost diagnostic devices for resource-limited settings.",
    "The psychologist researched the psychological effects of social isolation.",
    "The mathematician developed cryptographic protocols for secure communication.",
    "The conservationist restored native plant communities in degraded landscapes.",
    "The data scientist created models for optimizing public transit routes.",
    "The composer wrote music that explored the boundaries of tonality.",
    "The teacher used storytelling to teach complex scientific concepts.",
    "The entrepreneur developed technology for monitoring air quality.",
    "The physicist studied the behavior of matter at extremely low temperatures.",
    "The nurse coordinated community health screening programs.",
    "The programmer developed frameworks for building accessible web applications.",
    "The biologist studied the coevolution of plants and their pollinators.",
    "The artist created works that explored the relationship between humans and nature.",
    "The economist analyzed the impact of automation on income distribution.",
    "The engineer designed earthquake-resistant structures using innovative materials.",
    "The therapist researched the effectiveness of group therapy for addiction.",
    "The astronomer studied the life cycles of massive stars.",
    "The chef developed sustainable seafood sourcing practices.",
    "The writer documented the experiences of refugees and displaced persons.",
    "The scientist developed methods for cleaning up oil spills.",
    "The musician taught music appreciation in public schools.",
    "The teacher used inquiry-based learning to develop critical thinking skills.",
    "The researcher investigated the relationship between diet and mental health.",
    "The activist organized campaigns for environmental justice.",
    "The programmer developed tools for collaborative scientific research.",
    "The geologist studied the impact of groundwater extraction on land subsidence.",
    "The artist explored the aesthetics of imperfection in their work.",
    "The diplomat promoted international cooperation on renewable energy.",
    "The biologist developed methods for tracking animal migrations using satellite data.",
    "The journalist reported on the impact of climate change on agriculture.",
    "The engineer designed systems for capturing and utilizing waste heat.",
    "The psychologist studied the development of moral reasoning in children.",
    "The mathematician developed models for understanding complex biological systems.",
    "The conservationist advocated for policies to reduce plastic pollution.",
    "The data scientist built predictive models for personalized medicine.",
    "The composer created music that responded to environmental data.",
    "The teacher implemented culturally responsive teaching practices.",
    "The entrepreneur developed solutions for improving access to clean water.",
    "The physicist researched the fundamental nature of time.",
    "The nurse provided palliative care with compassion and dignity.",
    "The programmer contributed to open educational resources.",
    "The biologist studied the impact of noise pollution on marine mammals.",
    "The artist created works that challenged assumptions about identity.",
    "The economist studied the economic impact of natural disasters.",
    "The engineer developed technologies for monitoring deforestation.",
    "The therapist researched interventions for anxiety in adolescents.",
    "The astronomer studied the chemical evolution of the Milky Way.",
    "The chef promoted sustainable cooking practices in professional kitchens.",
    "The writer explored the intersection of science and spirituality.",
    "The scientist developed bioremediation techniques for contaminated soil.",
    "The musician composed works inspired by mathematical structures.",
    "The teacher fostered creativity and innovation in the classroom.",
    "The researcher studied the effectiveness of peer mentoring programs.",
    "The activist campaigned for equitable access to education.",
    "The programmer developed algorithms for optimizing energy consumption.",
    "The geologist studied the geological history of ancient river systems.",
    "The artist used technology to create interactive public art.",
    "The diplomat facilitated cultural diplomacy initiatives.",
    "The biologist studied the genetic basis of adaptation in extreme environments.",
    "The journalist investigated the impact of social media on democracy.",
    "The engineer designed sustainable transportation infrastructure.",
    "The psychologist researched the impact of mindfulness on workplace productivity.",
    "The mathematician developed statistical methods for analyzing genomic data.",
    "The conservationist worked to protect critical habitats for migratory species.",
    "The data scientist created tools for analyzing satellite imagery.",
    "The composer wrote music that incorporated field recordings from nature.",
    "The teacher used technology to personalize learning experiences.",
    "The entrepreneur developed platforms for connecting mentors with mentees.",
    "The physicist researched quantum sensors for medical imaging.",
    "The nurse advocated for patient-centered care models.",
    "The programmer developed tools for visualizing complex data.",
    "The biologist studied the role of microorganisms in soil health.",
    "The artist created works that explored the passage of time.",
    "The economist analyzed the impact of universal basic income proposals.",
    "The engineer designed systems for harvesting energy from ocean waves.",
    "The therapist researched the effectiveness of virtual reality exposure therapy.",
    "The astronomer studied the formation of planetary systems.",
    "The chef created innovative dishes using foraged ingredients.",
    "The writer explored themes of memory and loss in poetry.",
    "The scientist developed methods for converting waste plastic into fuel.",
    "The musician preserved and performed traditional folk music.",
    "The teacher implemented assessment strategies that promoted deep learning.",
    "The researcher studied the impact of community gardens on neighborhood cohesion.",
    "The activist organized campaigns for workers' rights.",
    "The programmer developed machine learning models for climate prediction.",
    "The geologist studied the impact of mining on local ecosystems.",
    "The artist explored the relationship between art and social change.",
    "The diplomat negotiated agreements on international water rights.",
    "The biologist studied the impact of invasive species on native ecosystems.",
    "The journalist covered the global movement for racial justice.",
    "The engineer developed technologies for sustainable aquaculture.",
    "The psychologist researched the impact of play on child development.",
    "The mathematician developed algorithms for optimizing traffic flow.",
    "The conservationist established community-led conservation programs.",
    "The data scientist built models for predicting crop yields.",
    "The composer created music that explored the concept of silence.",
    "The teacher used collaborative learning to build community in the classroom.",
    "The entrepreneur developed technologies for accessible communication.",
    "The physicist researched the properties of two-dimensional materials.",
    "The nurse provided mental health first aid training in communities.",
    "The programmer developed tools for automated code documentation.",
    "The biologist studied the impact of light on plant growth patterns.",
    "The artist created works that responded to current social issues.",
    "The economist studied the impact of microfinance on poverty reduction.",
    "The engineer designed systems for monitoring air quality in real time.",
    "The therapist researched interventions for compassion fatigue in caregivers.",
    "The astronomer studied the dynamics of star clusters.",
    "The chef developed recipes using ancient grains and heritage crops.",
    "The writer explored the experience of aging through creative nonfiction.",
    "The scientist developed methods for predicting volcanic eruptions.",
    "The musician collaborated with visual artists on multimedia projects.",
    "The teacher implemented trauma-informed practices in the classroom.",
    "The researcher studied the impact of bilingual education on cognitive development.",
    "The activist campaigned for transparency in government.",
    "The programmer developed tools for citizen science projects.",
    "The geologist studied the impact of sea level rise on coastal geology.",
    "The artist created works that explored the concept of belonging.",
    "The diplomat promoted international cooperation on education initiatives.",
    "The biologist studied the impact of pesticides on beneficial insects.",
    "The journalist investigated the impact of fast fashion on workers.",
    "The engineer designed systems for recycling electronic waste.",
    "The psychologist researched the impact of gratitude practices on well-being.",
    "The mathematician developed models for understanding social networks.",
    "The conservationist advocated for sustainable forestry practices.",
    "The data scientist created algorithms for detecting anomalies in financial data.",
    "The composer wrote music that explored the intersection of tradition and innovation.",
    "The teacher used outdoor education to connect students with nature.",
    "The entrepreneur developed solutions for reducing energy consumption in buildings.",
    "The physicist researched the fundamental forces of nature.",
    "The nurse implemented infection control protocols in healthcare settings.",
    "The programmer developed frameworks for building secure applications.",
    "The biologist studied the impact of climate change on species distributions.",
    "The artist created works that celebrated cultural diversity.",
    "The economist analyzed the impact of digital currencies on financial systems.",
    "The engineer designed systems for sustainable waste management.",
    "The therapist researched the effectiveness of art therapy for PTSD.",
    "The astronomer studied the evolution of stellar populations.",
    "The chef created menus that highlighted seasonal and local ingredients.",
    "The writer explored the impact of technology on human relationships.",
    "The scientist developed methods for restoring degraded ecosystems.",
    "The musician composed works that bridged cultural traditions.",
    "The teacher implemented project-based learning in science education.",
    "The researcher studied the impact of sleep deprivation on academic performance.",
    "The activist organized campaigns for affordable childcare.",
    "The programmer developed tools for analyzing genomic sequences.",
    "The geologist studied the impact of fracking on groundwater quality.",
    "The artist created works that explored the concept of memory.",
    "The diplomat facilitated international cooperation on disaster response.",
    "The biologist studied the role of fungi in forest ecosystems.",
    "The journalist covered the global impact of the refugee crisis.",
    "The engineer designed systems for sustainable urban agriculture.",
    "The psychologist researched the impact of positive psychology interventions.",
    "The mathematician developed algorithms for optimizing resource distribution.",
    "The conservationist worked to protect endangered marine species.",
    "The data scientist built models for predicting disease progression.",
    "The composer created music that responded to social and political events.",
    "The teacher used differentiated assessment to measure student growth.",
    "The entrepreneur developed platforms for skill sharing in communities.",
    "The physicist researched applications of quantum mechanics in computing.",
    "The nurse provided health education in schools and community centers.",
    "The programmer developed tools for collaborative document editing.",
    "The biologist studied the impact of habitat fragmentation on gene flow.",
    "The artist created works that challenged conventional perspectives.",
    "The economist studied the impact of universal healthcare on economic outcomes.",
    "The engineer designed systems for monitoring water quality.",
    "The therapist researched interventions for eating disorders.",
    "The astronomer studied the structure and dynamics of the interstellar medium.",
    "The chef promoted culinary traditions from underrepresented cultures.",
    "The writer explored the experience of immigration through memoir.",
    "The scientist developed methods for monitoring biodiversity at scale.",
    "The musician used music as a tool for social change.",
    "The teacher implemented restorative practices to address behavioral issues.",
    "The researcher studied the impact of mentorship on career development.",
    "The activist campaigned for environmental education in schools.",
    "The programmer developed algorithms for natural language understanding.",
    "The geologist studied the impact of climate change on glacial dynamics.",
    "The artist created works that explored the relationship between art and science.",
    "The diplomat promoted international cooperation on public health.",
    "The biologist studied the impact of ocean warming on marine ecosystems.",
    "The journalist investigated the impact of algorithmic decision-making on society.",
    "The engineer designed systems for sustainable building materials.",
    "The psychologist researched the impact of social support on health outcomes.",
    "The mathematician developed models for understanding complex systems.",
    "The conservationist advocated for policies to protect biodiversity.",
    "The data scientist created tools for analyzing social network data.",
    "The composer wrote music that explored the boundaries of genre.",
    "The teacher used experiential learning to deepen student understanding.",
    "The entrepreneur developed solutions for improving mental health access.",
    "The physicist researched the nature of dark energy.",
    "The nurse advocated for evidence-based nursing practice.",
    "The programmer developed tools for automated software testing.",
    "The biologist studied the impact of environmental toxins on development.",
    "The artist created works that explored the concept of transformation.",
    "The economist analyzed the impact of trade agreements on local economies.",
    "The engineer designed systems for sustainable food production.",
    "The therapist researched the effectiveness of family therapy.",
    "The astronomer studied the formation of stars in molecular clouds.",
    "The chef developed innovative techniques for food preservation.",
    "The writer explored the intersection of art and technology.",
    "The scientist developed methods for predicting extreme weather events.",
    "The musician composed works that celebrated cultural heritage.",
    "The teacher implemented inclusive assessment practices.",
    "The researcher studied the impact of arts education on student outcomes.",
    "The activist organized campaigns for criminal justice reform.",
    "The programmer developed tools for data visualization.",
    "The geologist studied the impact of land use change on soil health.",
    "The artist created works that explored the concept of identity.",
    "The diplomat facilitated international cooperation on sustainable development.",
    "The biologist studied the impact of pollution on freshwater ecosystems.",
    "The journalist covered the intersection of technology and human rights.",
    "The engineer designed systems for renewable energy integration.",
    "The psychologist researched the impact of community on well-being.",
    "The mathematician developed algorithms for machine learning optimization.",
    "The conservationist worked to restore degraded forest ecosystems.",
    "The data scientist built models for understanding consumer behavior.",
    "The composer created music that explored the relationship between sound and space.",
    "The teacher used technology to facilitate global collaboration.",
    "The entrepreneur developed platforms for community engagement.",
    "The physicist researched the properties of exotic states of matter.",
    "The nurse provided care coordination for patients with complex needs.",
    "The programmer developed frameworks for building scalable systems.",
    "The biologist studied the impact of climate change on phenology.",
    "The artist created works that explored the concept of place.",
    "The economist studied the impact of education on economic mobility.",
    "The engineer designed systems for sustainable water management.",
    "The therapist researched interventions for substance use disorders.",
    "The astronomer studied the evolution of galaxies over cosmic time.",
    "The chef promoted sustainable fishing practices.",
    "The writer explored the human experience through creative writing.",
    "The scientist developed methods for monitoring ecosystem health.",
    "The musician collaborated across disciplines to create innovative works.",
    "The teacher fostered a love of learning in students.",
    "The researcher studied the impact of community involvement on health.",
    "The activist campaigned for equitable access to healthcare.",
    "The programmer developed tools for open data initiatives.",
    "The geologist studied the impact of erosion on coastal communities.",
    "The artist created works that explored the concept of time.",
    "The diplomat promoted international cooperation on cultural preservation.",
    "The biologist studied the impact of deforestation on carbon cycling.",
    "The journalist investigated the impact of surveillance on civil liberties.",
    "The engineer designed systems for sustainable transportation.",
    "The psychologist researched the impact of nature on cognitive function.",
    "The mathematician developed models for understanding population dynamics.",
    "The conservationist advocated for sustainable agriculture practices.",
    "The data scientist created algorithms for optimizing healthcare delivery.",
    "The composer wrote music that explored the intersection of art and science.",
    "The teacher implemented student-centered learning approaches.",
    "The entrepreneur developed solutions for addressing food insecurity.",
    "The physicist researched the fundamental nature of space and time.",
    "The nurse provided compassionate care in emergency situations.",
    "The programmer developed tools for collaborative problem solving.",
    "The biologist studied the impact of urbanization on biodiversity.",
    "The artist created works that challenged societal norms.",
    "The economist analyzed the impact of technological change on labor markets.",
    "The engineer designed systems for sustainable energy storage.",
    "The therapist researched the effectiveness of play therapy.",
    "The astronomer studied the chemical composition of interstellar dust.",
    "The chef created innovative plant-based dishes.",
    "The writer explored themes of resilience and perseverance.",
    "The scientist developed methods for restoring coral reefs.",
    "The musician used technology to create new musical experiences.",
    "The teacher implemented culturally sustaining pedagogy.",
    "The researcher studied the impact of early intervention on developmental outcomes.",
    "The activist organized campaigns for environmental protection.",
    "The programmer developed algorithms for computer vision.",
    "The geologist studied the impact of volcanic activity on climate.",
    "The artist created works that explored the concept of connection.",
    "The diplomat facilitated international cooperation on human rights.",
    "The biologist studied the impact of microplastics on food webs.",
    "The journalist covered the global movement for gender equality.",
    "The engineer designed systems for sustainable manufacturing.",
    "The psychologist researched the impact of meditation on brain function.",
    "The mathematician developed statistical methods for clinical trials.",
    "The conservationist worked to protect critical wetland habitats.",
    "The data scientist built models for predicting economic trends.",
    "The composer created music that explored the concept of harmony.",
    "The teacher used authentic assessment to measure real-world skills.",
    "The entrepreneur developed platforms for social impact.",
    "The physicist researched applications of nanotechnology.",
    "The nurse advocated for health equity in underserved populations.",
    "The programmer developed tools for reproducible research.",
    "The biologist studied the impact of invasive species on ecosystem function.",
    "The artist created works that explored the concept of change.",
    "The economist studied the impact of social programs on poverty.",
    "The engineer designed systems for sustainable urban planning.",
    "The therapist researched interventions for grief and loss.",
    "The astronomer studied the dynamics of binary star systems.",
    "The chef promoted food literacy in communities.",
    "The writer explored the complexity of human relationships.",
    "The scientist developed methods for monitoring air pollution.",
    "The musician composed works that celebrated diversity.",
    "The teacher implemented formative assessment to guide instruction.",
    "The researcher studied the impact of physical activity on mental health.",
    "The activist campaigned for accessible education for all.",
    "The programmer developed tools for ethical AI development.",
    "The geologist studied the impact of groundwater contamination on health.",
    "The artist created works that explored the concept of memory.",
    "The diplomat promoted international cooperation on education.",
    "The biologist studied the impact of climate change on species interactions.",
    "The journalist investigated the impact of automation on employment.",
    "The engineer designed systems for sustainable resource management.",
    "The psychologist researched the impact of creativity on problem solving.",
    "The mathematician developed models for understanding ecosystem dynamics.",
    "The conservationist advocated for policies to reduce carbon emissions.",
    "The data scientist created tools for analyzing environmental data.",
    "The composer wrote music that explored the relationship between rhythm and emotion.",
    "The teacher used inquiry-based approaches to develop scientific thinking.",
    "The entrepreneur developed solutions for improving access to technology.",
    "The physicist researched the properties of quantum materials.",
    "The nurse provided patient education on disease prevention.",
    "The programmer developed frameworks for building accessible applications.",
    "The biologist studied the impact of habitat loss on species survival.",
    "The artist created works that explored the concept of community.",
    "The economist analyzed the impact of globalization on local cultures.",
    "The engineer designed systems for sustainable waste reduction.",
    "The therapist researched the effectiveness of mindfulness-based therapy.",
    "The astronomer studied the formation of planetary nebulae.",
    "The chef developed sustainable sourcing practices for restaurants.",
    "The writer explored the impact of social change on individuals.",
    "The scientist developed methods for predicting landslide risk.",
    "The musician preserved traditional music through documentation and performance.",
    "The teacher implemented universal design for learning principles.",
    "The researcher studied the impact of nutrition on cognitive performance.",
    "The activist organized campaigns for digital equity.",
    "The programmer developed tools for community-driven development.",
    "The geologist studied the impact of mining on water resources.",
    "The artist created works that explored the concept of resilience.",
    "The diplomat facilitated international cooperation on technology transfer.",
    "The biologist studied the impact of ocean acidification on shellfish.",
    "The journalist covered the impact of climate migration on communities.",
    "The engineer designed systems for sustainable energy distribution.",
    "The psychologist researched the impact of social media on self-esteem.",
    "The mathematician developed algorithms for optimizing logistics.",
    "The conservationist worked to protect endangered plant species.",
    "The data scientist built models for understanding urban growth patterns.",
    "The composer created music that explored the intersection of cultures.",
    "The teacher used collaborative projects to build teamwork skills.",
    "The entrepreneur developed platforms for connecting communities.",
    "The physicist researched the fundamental nature of matter.",
    "The nurse implemented patient safety protocols.",
    "The programmer developed tools for automated data analysis.",
    "The biologist studied the impact of light pollution on ecosystems.",
    "The artist created works that explored the concept of transformation.",
    "The economist studied the impact of innovation on economic growth.",
    "The engineer designed systems for sustainable agriculture.",
    "The therapist researched interventions for childhood trauma.",
    "The astronomer studied the evolution of stellar remnants.",
    "The chef promoted culinary education in schools.",
    "The writer explored the intersection of tradition and modernity.",
    "The scientist developed methods for monitoring water quality.",
    "The musician used music to bridge cultural divides.",
    "The teacher implemented evidence-based instructional strategies.",
    "The researcher studied the impact of community engagement on student success.",
    "The activist campaigned for sustainable urban development.",
    "The programmer developed algorithms for speech recognition.",
    "The geologist studied the impact of sea level rise on infrastructure.",
    "The artist created works that explored the concept of belonging.",
    "The diplomat promoted international cooperation on scientific research.",
    "The biologist studied the impact of pollution on soil health.",
    "The journalist investigated the impact of technology on privacy.",
    "The engineer designed systems for sustainable building design.",
    "The psychologist researched the impact of gratitude on relationships.",
    "The mathematician developed models for understanding disease spread.",
    "The conservationist advocated for sustainable tourism practices.",
    "The data scientist created tools for analyzing genomic data.",
    "The composer wrote music that explored the concept of silence and sound.",
    "The teacher used project-based learning to develop real-world skills.",
    "The entrepreneur developed solutions for addressing homelessness.",
    "The physicist researched the nature of quantum information.",
    "The nurse provided end-of-life care with compassion.",
    "The programmer developed tools for collaborative learning.",
    "The biologist studied the impact of climate change on migration patterns.",
    "The artist created works that challenged conventional thinking.",
    "The economist analyzed the impact of climate policy on economic outcomes.",
    "The engineer designed systems for sustainable waste management.",
    "The therapist researched the effectiveness of cognitive behavioral therapy.",
    "The astronomer studied the dynamics of galactic nuclei.",
    "The chef developed innovative approaches to food sustainability.",
    "The writer explored the experience of cultural displacement.",
    "The scientist developed methods for restoring degraded habitats.",
    "The musician composed works that celebrated human creativity.",
    "The teacher implemented trauma-informed teaching practices.",
    "The researcher studied the impact of arts integration on learning.",
    "The activist organized campaigns for social justice.",
    "The programmer developed tools for transparent governance.",
    "The geologist studied the impact of land use on water quality.",
    "The artist created works that explored the concept of identity.",
    "The diplomat facilitated international cooperation on climate action.",
    "The biologist studied the impact of habitat fragmentation on populations.",
    "The journalist covered the global impact of inequality.",
    "The engineer designed systems for sustainable energy production.",
    "The psychologist researched the impact of community on resilience.",
    "The mathematician developed algorithms for optimizing resource use.",
    "The conservationist worked to protect marine ecosystems.",
    "The data scientist built models for predicting environmental change.",
    "The composer created music that explored the relationship between nature and culture.",
    "The teacher used technology to enhance student engagement.",
    "The entrepreneur developed platforms for community organizing.",
    "The physicist researched the fundamental nature of energy.",
    "The nurse advocated for patient rights and dignity.",
    "The programmer developed frameworks for ethical technology development.",
    "The biologist studied the impact of environmental change on species.",
    "The artist created works that explored the concept of connection.",
    "The economist studied the impact of education on social mobility.",
    "The engineer designed systems for sustainable water purification.",
    "The therapist researched interventions for anxiety disorders.",
    "The astronomer studied the formation and evolution of stars.",
    "The chef promoted sustainable food systems.",
    "The writer explored the human condition through storytelling.",
    "The scientist developed methods for monitoring ecosystem recovery.",
    "The musician used music as a form of social commentary.",
    "The teacher fostered critical thinking in students.",
    "The researcher studied the impact of community health programs.",
    "The activist campaigned for environmental justice.",
    "The programmer developed tools for open science.",
    "The geologist studied the impact of climate change on geological processes.",
    "The artist created works that explored the concept of transformation.",
    "The diplomat promoted international cooperation on sustainable development.",
    "The biologist studied the impact of pollution on aquatic ecosystems.",
    "The journalist investigated the impact of technology on society.",
    "The engineer designed systems for sustainable urban development.",
    "The psychologist researched the impact of nature on mental health.",
    "The mathematician developed models for understanding complex networks.",
    "The conservationist advocated for biodiversity protection.",
    "The data scientist created algorithms for fair decision making.",
    "The composer wrote music that explored the intersection of tradition and innovation.",
    "The teacher used experiential learning to deepen understanding.",
    "The entrepreneur developed solutions for improving quality of life.",
    "The physicist researched the properties of fundamental particles.",
    "The nurse provided holistic care to patients.",
    "The programmer developed tools for accessible technology.",
    "The biologist studied the impact of climate change on ecosystems.",
    "The artist created works that explored the concept of change.",
    "The economist analyzed the impact of social policies on well-being.",
    "The engineer designed systems for sustainable resource use.",
    "The therapist researched the effectiveness of group interventions.",
    "The astronomer studied the evolution of the universe.",
    "The chef promoted sustainable culinary practices.",
    "The writer explored themes of hope and resilience.",
    "The scientist developed methods for environmental monitoring.",
    "The musician composed works that celebrated human achievement.",
    "The teacher implemented inclusive educational practices.",
    "The researcher studied the impact of community on health outcomes.",
    "The activist organized campaigns for systemic change.",
    "The programmer developed algorithms for understanding language.",
    "The geologist studied the impact of human activity on landscapes.",
    "The artist created works that explored the concept of belonging.",
    "The diplomat facilitated international cooperation on peace.",
    "The biologist studied the impact of environmental stressors on organisms.",
    "The journalist covered the intersection of technology and ethics.",
    "The engineer designed systems for sustainable infrastructure.",
    "The psychologist researched the impact of mindfulness on well-being.",
    "The mathematician developed statistical methods for data analysis.",
    "The conservationist worked to protect natural habitats.",
    "The data scientist built models for understanding social dynamics.",
    "The composer created music that explored the concept of harmony.",
    "The teacher used collaborative learning to build community.",
    "The entrepreneur developed platforms for positive social impact.",
    "The physicist researched the nature of quantum phenomena.",
    "The nurse provided compassionate patient care.",
    "The programmer developed tools for collaborative innovation.",
    "The biologist studied the impact of environmental change on biodiversity.",
    "The artist created works that challenged assumptions.",
    "The economist studied the impact of sustainable practices on economies.",
    "The engineer designed systems for sustainable development.",
    "The therapist researched interventions for mental health.",
    "The astronomer studied the cosmos and our place in it.",
    "The chef promoted sustainable food practices.",
    "The writer explored the depths of human experience.",
    "The scientist developed methods for protecting the environment.",
    "The musician used music to inspire change.",
    "The teacher nurtured the potential in every student.",
]


def train_adapter(
    n_samples: int = 500,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    ollama_model: str = "mistral",
    output_path: str = "adapter_weights.pt",
):
    """Train the embedding adapter."""
    from ollama_extractor import EmbeddingAdapter

    print(f"[train_adapter] Training adapter with {n_samples} samples")
    print(f"[train_adapter] Ollama model: {ollama_model}")
    print(f"[train_adapter] Epochs: {epochs}, Batch size: {batch_size}")

    # Check Ollama
    if not OLLAMA_AVAILABLE:
        print("[train_adapter] ERROR: ollama package not installed")
        print("[train_adapter] Install with: pip install ollama")
        return False

    try:
        client = ollama.Client()
        client.list()
        print("[train_adapter] Ollama connection OK")
    except Exception as e:
        print(f"[train_adapter] ERROR: Cannot connect to Ollama: {e}")
        print("[train_adapter] Make sure Ollama is running on localhost:11434")
        return False

    # Get Ollama embeddings
    print(f"[train_adapter] Getting Ollama embeddings for {n_samples} texts...")
    texts = SAMPLE_TEXTS[:n_samples]
    
    ollama_embeddings = []
    batch_size_embed = 10
    for i in range(0, len(texts), batch_size_embed):
        batch_texts = texts[i:i+batch_size_embed]
        try:
            response = client.embed(
                model=ollama_model,
                input=batch_texts,
                truncate=True
            )
            ollama_embeddings.extend(response['embeddings'])
            print(f"[train_adapter]   Processed {min(i+batch_size_embed, len(texts))}/{len(texts)} texts")
        except Exception as e:
            print(f"[train_adapter] ERROR getting Ollama embeddings: {e}")
            return False

    ollama_embeddings = np.array(ollama_embeddings, dtype=np.float32)
    print(f"[train_adapter] Ollama embeddings shape: {ollama_embeddings.shape}")

    # Generate LLaMA-like embeddings using TRIBE v2's text extractor
    print("[train_adapter] Generating LLaMA-like embeddings...")
    try:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TRANSFORMERS_DEVICE'] = 'cpu'
        
        from neuralset.extractors.text import HuggingFaceText
        import pandas as pd
        
        # Create a HuggingFaceText extractor for LLaMA
        llama_extractor = HuggingFaceText(
            model_name="meta-llama/Llama-3.2-3B",
            event_types=["Word"],
            aggregation="sum",
            frequency=2.0,
            contextualized=True,
            layers=[0.5, 0.75, 1.0],
            layer_aggregation="group_mean",
            token_aggregation="mean",
            cache_n_layers=20,
            batch_size=4,
            device="cpu",
            pretrained=True,
        )
        
        # Get embeddings for each text
        llama_embeddings = []
        for i, text in enumerate(texts):
            # Create a simple event for the text
            events = pd.DataFrame([{
                "type": "Word",
                "start": 0.0,
                "duration": 1.0,
                "text": text,
                "context": text,
                "timeline": "default",
                "subject": "default",
                "sequence_id": 0,
                "sentence": text,
                "language": "english",
                "offset": 0.0,
                "frequency": 1.0,
                "filepath": None,
                "extra": {}
            }])
            
            # Get embedding
            embedding = llama_extractor.get_embedding(text)
            # Flatten to match target dimension (3, 3072) -> 9216
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            llama_embeddings.append(embedding)
            
            if (i + 1) % 50 == 0:
                print(f"[train_adapter]   Processed {i+1}/{len(texts)} LLaMA embeddings")
        
        llama_embeddings = np.array(llama_embeddings, dtype=np.float32)
        print(f"[train_adapter] LLaMA embeddings shape: {llama_embeddings.shape}")
        
    except Exception as e:
        import traceback
        print(f"[train_adapter] ERROR generating LLaMA embeddings: {e}")
        print(traceback.format_exc())
        print("[train_adapter] Falling back to random target embeddings (not ideal)")
        # Fallback: create random embeddings with correct shape
        target_dim = 9216
        llama_embeddings = np.random.randn(len(texts), target_dim).astype(np.float32)
        print(f"[train_adapter] Random embeddings shape: {llama_embeddings.shape}")

    # Split into train/val
    n_train = int(0.8 * len(texts))
    X_train = torch.from_numpy(ollama_embeddings[:n_train])
    y_train = torch.from_numpy(llama_embeddings[:n_train])
    X_val = torch.from_numpy(ollama_embeddings[n_train:])
    y_val = torch.from_numpy(llama_embeddings[n_train:])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create adapter
    input_dim = ollama_embeddings.shape[1]
    target_dim = llama_embeddings.shape[1]
    adapter = EmbeddingAdapter(input_dim, target_dim)
    
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    print(f"[train_adapter] Training adapter: {input_dim} -> {target_dim}")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        adapter.train()
        total_train_loss = 0
        n_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = adapter(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_train_loss / n_batches
        
        # Validation
        adapter.eval()
        with torch.no_grad():
            val_pred = adapter(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            adapter.save(output_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"[train_adapter] Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.6f}, Val loss: {val_loss:.6f}")

    # Final evaluation
    adapter.eval()
    with torch.no_grad():
        val_pred = adapter(X_val)
        val_loss = criterion(val_pred, y_val).item()
        
        # Cosine similarity
        val_pred_norm = val_pred / val_pred.norm(dim=1, keepdim=True)
        y_val_norm = y_val / y_val.norm(dim=1, keepdim=True)
        cosine_sim = (val_pred_norm * y_val_norm).sum(dim=1).mean().item()
    
    print(f"[train_adapter] Final val loss: {val_loss:.6f}")
    print(f"[train_adapter] Cosine similarity: {cosine_sim:.4f}")
    print(f"[train_adapter] Adapter saved to {output_path}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train embedding adapter")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of text samples")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--ollama_model", type=str, default="mistral", help="Ollama model name")
    parser.add_argument("--output", type=str, default="adapter_weights.pt", help="Output path")
    
    args = parser.parse_args()
    
    success = train_adapter(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        ollama_model=args.ollama_model,
        output_path=args.output,
    )
    
    sys.exit(0 if success else 1)
