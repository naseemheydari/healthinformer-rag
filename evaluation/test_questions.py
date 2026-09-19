"""Curated test questions spanning all 36 topic categories.

These are our "gold" test set for RAGAS evaluation. Each question includes
an optional ground_truth answer sketch (used by context_recall) and the
expected topic category for analysis breakdowns.

Coverage: ~50 questions across all 36 categories from config/constants.py.
Question style: plain-language questions a general-public user would ask.
Mix of factual, comparative, and lifestyle question types.
"""

# Each entry: (question, category, ground_truth_sketch)
# ground_truth_sketch is a brief expected answer — not exhaustive, just
# enough for RAGAS context_recall to check if retrieval covered the key facts.
TEST_QUESTIONS: list[tuple[str, str, str]] = [
    # ── Diabetes ────────────────────────────────────────────────────────
    (
        "What are the warning signs of type 2 diabetes?",
        "Diabetes",
        "Common warning signs include increased thirst, frequent urination, "
        "unexplained weight loss, fatigue, blurred vision, and slow-healing wounds.",
    ),
    (
        "How is type 2 diabetes typically managed?",
        "Diabetes",
        "Management includes lifestyle changes (diet, exercise), blood sugar monitoring, "
        "oral medications like metformin, and sometimes insulin therapy.",
    ),

    # ── Hypertension ────────────────────────────────────────────────────
    (
        "What lifestyle changes help lower blood pressure?",
        "Hypertension",
        "Reducing sodium intake, regular exercise, maintaining healthy weight, "
        "limiting alcohol, managing stress, and eating a DASH-style diet.",
    ),

    # ── Asthma ──────────────────────────────────────────────────────────
    (
        "What triggers can worsen asthma symptoms in adults?",
        "Asthma",
        "Common triggers include allergens, air pollution, respiratory infections, "
        "exercise, cold air, smoke, and stress.",
    ),

    # ── Heart Failure ───────────────────────────────────────────────────
    (
        "What are the early symptoms of heart failure?",
        "Heart Failure",
        "Early symptoms include shortness of breath, fatigue, swelling in legs and ankles, "
        "rapid or irregular heartbeat, and reduced exercise tolerance.",
    ),

    # ── Kidney Disease ──────────────────────────────────────────────────
    (
        "How can chronic kidney disease be prevented?",
        "Kidney Disease",
        "Prevention includes controlling blood pressure and blood sugar, maintaining healthy weight, "
        "avoiding excessive NSAID use, staying hydrated, and not smoking.",
    ),

    # ── Cancer Screening ────────────────────────────────────────────────
    (
        "At what age should breast cancer screening begin?",
        "Cancer Screening",
        "Guidelines generally recommend mammography screening starting at age 40-50, "
        "depending on risk factors and the guideline organization.",
    ),

    # ── Cancer Prevention ───────────────────────────────────────────────
    (
        "What are the recommended screenings for colorectal cancer?",
        "Cancer Prevention",
        "Recommended screenings include colonoscopy, stool-based tests (FIT, gFOBT), "
        "and CT colonography, typically starting at age 45.",
    ),

    # ── Cancer Risk ─────────────────────────────────────────────────────
    (
        "How does smoking increase the risk of lung cancer?",
        "Cancer Risk",
        "Smoking damages lung cells through carcinogens in tobacco smoke, causing DNA mutations "
        "that accumulate over time and lead to uncontrolled cell growth.",
    ),

    # ── Mental Health ───────────────────────────────────────────────────
    (
        "What are evidence-based treatments for depression?",
        "Mental Health",
        "Evidence-based treatments include cognitive behavioral therapy (CBT), "
        "antidepressant medications (SSRIs, SNRIs), exercise, and combination therapy.",
    ),
    (
        "How effective is cognitive behavioral therapy for anxiety?",
        "Mental Health",
        "CBT is considered a first-line treatment for anxiety disorders with strong evidence "
        "for its effectiveness, often comparable to or better than medication alone.",
    ),

    # ── Sleep Health ────────────────────────────────────────────────────
    (
        "What are recommended treatments for chronic insomnia?",
        "Sleep Health",
        "Recommended treatments include cognitive behavioral therapy for insomnia (CBT-I), "
        "sleep hygiene practices, and in some cases short-term medication.",
    ),

    # ── Nutrition ───────────────────────────────────────────────────────
    (
        "What dietary patterns are best for heart health?",
        "Nutrition",
        "Heart-healthy diets include the Mediterranean diet, DASH diet, and plant-based diets "
        "rich in fruits, vegetables, whole grains, lean protein, and healthy fats.",
    ),
    (
        "Is intermittent fasting safe and effective for weight loss?",
        "Nutrition",
        "Evidence suggests intermittent fasting can aid weight loss and improve metabolic markers, "
        "but results are comparable to standard caloric restriction. Not recommended for everyone.",
    ),

    # ── Exercise ────────────────────────────────────────────────────────
    (
        "How much physical activity do adults need per week?",
        "Exercise",
        "Adults should get at least 150 minutes of moderate-intensity or 75 minutes of "
        "vigorous-intensity aerobic activity per week, plus muscle-strengthening activities.",
    ),
    (
        "What's the difference between cardio and strength training benefits?",
        "Exercise",
        "Cardio improves cardiovascular fitness, endurance, and calorie burn. Strength training "
        "builds muscle mass, bone density, and metabolic rate. Both are recommended for overall health.",
    ),

    # ── Weight Management ───────────────────────────────────────────────
    (
        "What interventions are effective for obesity management?",
        "Weight Management",
        "Effective interventions include caloric restriction, increased physical activity, "
        "behavioral counseling, and in some cases medication or bariatric surgery.",
    ),

    # ── Vaccines ────────────────────────────────────────────────────────
    (
        "How effective is the annual influenza vaccine?",
        "Vaccines",
        "Influenza vaccine effectiveness varies by season (typically 40-60%), "
        "but vaccination reduces severity of illness and risk of hospitalization.",
    ),
    (
        "What vaccines are recommended for adults over 65?",
        "Vaccines",
        "Recommended vaccines include influenza (high-dose), pneumococcal, shingles (Shingrix), "
        "Tdap booster, and COVID-19 boosters.",
    ),

    # ── COVID-19 ────────────────────────────────────────────────────────
    (
        "What are the long-term effects of COVID-19?",
        "COVID-19",
        "Long COVID symptoms include fatigue, brain fog, shortness of breath, "
        "chest pain, joint pain, and cardiovascular complications.",
    ),

    # ── Antibiotics ─────────────────────────────────────────────────────
    (
        "Why is antibiotic resistance a growing concern?",
        "Antibiotics",
        "Overuse and misuse of antibiotics drive resistance, making infections harder to treat, "
        "increasing healthcare costs, and raising mortality risk.",
    ),

    # ── Prenatal Health ─────────────────────────────────────────────────
    (
        "What are the key components of prenatal care?",
        "Prenatal Health",
        "Key components include regular check-ups, nutritional guidance, folic acid supplementation, "
        "screening tests, monitoring fetal development, and managing risk factors.",
    ),

    # ── Women's Health ──────────────────────────────────────────────────
    (
        "What are common symptoms of menopause and how are they managed?",
        "Women's Health",
        "Common symptoms include hot flashes, night sweats, mood changes, and sleep disturbances. "
        "Management options include hormone replacement therapy, non-hormonal medications, "
        "and lifestyle modifications.",
    ),

    # ── Bone Health ─────────────────────────────────────────────────────
    (
        "How can osteoporosis be prevented?",
        "Bone Health",
        "Prevention includes adequate calcium and vitamin D intake, weight-bearing exercise, "
        "avoiding smoking and excessive alcohol, and bone density screening.",
    ),

    # ── Geriatrics ──────────────────────────────────────────────────────
    (
        "What strategies help prevent falls in elderly adults?",
        "Geriatrics",
        "Strategies include exercise programs for balance and strength, home safety modifications, "
        "medication review, vision checks, and assistive devices.",
    ),

    # ── Sexual Health ───────────────────────────────────────────────────
    (
        "What are the most effective ways to prevent sexually transmitted infections?",
        "Sexual Health",
        "Prevention includes consistent condom use, regular STI screening, vaccination (HPV, hepatitis B), "
        "limiting number of partners, and open communication with partners.",
    ),
    (
        "What treatment options are available for erectile dysfunction?",
        "Sexual Health",
        "Treatment options include PDE5 inhibitors (sildenafil, tadalafil), lifestyle changes, "
        "testosterone therapy if deficient, vacuum devices, and psychological counseling.",
    ),

    # ── Dermatology ─────────────────────────────────────────────────────
    (
        "What are the best ways to protect skin from sun damage?",
        "Dermatology",
        "Sun protection includes broad-spectrum sunscreen (SPF 30+), protective clothing, "
        "seeking shade during peak hours, avoiding tanning beds, and regular skin checks.",
    ),
    (
        "What treatments are available for eczema?",
        "Dermatology",
        "Treatments include moisturizers, topical corticosteroids, calcineurin inhibitors, "
        "antihistamines for itching, and avoiding known triggers like harsh soaps and allergens.",
    ),

    # ── Allergies ───────────────────────────────────────────────────────
    (
        "How are seasonal allergies treated?",
        "Allergies",
        "Treatment includes antihistamines, nasal corticosteroid sprays, decongestants, "
        "allergen avoidance, and immunotherapy (allergy shots) for severe cases.",
    ),
    (
        "What should someone do during a severe allergic reaction?",
        "Allergies",
        "Use an epinephrine auto-injector immediately, call emergency services, lie down with legs elevated, "
        "and seek immediate medical attention. Anaphylaxis can be life-threatening.",
    ),

    # ── Substance Use ───────────────────────────────────────────────────
    (
        "What are the health risks of vaping and e-cigarettes?",
        "Substance Use",
        "Health risks include lung damage (EVALI), nicotine addiction, cardiovascular effects, "
        "exposure to harmful chemicals, and potential gateway to traditional cigarettes in youth.",
    ),
    (
        "What are the most effective methods to quit smoking?",
        "Substance Use",
        "Effective methods include nicotine replacement therapy (patches, gum), prescription medications "
        "(varenicline, bupropion), behavioral counseling, and combination approaches.",
    ),

    # ── Pain Management ─────────────────────────────────────────────────
    (
        "What are non-drug treatments for chronic back pain?",
        "Pain Management",
        "Non-drug treatments include physical therapy, exercise, yoga, acupuncture, "
        "cognitive behavioral therapy, massage, and spinal manipulation.",
    ),
    (
        "What causes migraines and how can they be prevented?",
        "Pain Management",
        "Migraines are caused by neurological changes involving blood vessels and neurotransmitters. "
        "Prevention includes identifying triggers, stress management, regular sleep, "
        "and preventive medications like beta-blockers or anti-seizure drugs.",
    ),

    # ── Oral Health ─────────────────────────────────────────────────────
    (
        "How does oral health affect overall body health?",
        "Oral Health",
        "Poor oral health is linked to cardiovascular disease, diabetes, respiratory infections, "
        "and adverse pregnancy outcomes. Periodontal bacteria can enter the bloodstream.",
    ),

    # ── Eye Health ──────────────────────────────────────────────────────
    (
        "How can I reduce eye strain from screens?",
        "Eye Health",
        "Follow the 20-20-20 rule (every 20 minutes look at something 20 feet away for 20 seconds), "
        "adjust screen brightness, use proper lighting, blink frequently, and take regular breaks.",
    ),

    # ── Ear Health ──────────────────────────────────────────────────────
    (
        "How can hearing loss be prevented?",
        "Ear Health",
        "Prevention includes limiting exposure to loud noises, wearing ear protection, "
        "keeping volume low on personal audio devices, and getting regular hearing checks.",
    ),

    # ── Digestive Health ────────────────────────────────────────────────
    (
        "What helps manage irritable bowel syndrome symptoms?",
        "Digestive Health",
        "Management includes dietary changes (low-FODMAP diet), stress management, regular exercise, "
        "adequate hydration, fiber supplementation, and medications for specific symptoms.",
    ),
    (
        "Are probiotics actually beneficial for gut health?",
        "Digestive Health",
        "Evidence supports probiotics for specific conditions like antibiotic-associated diarrhea "
        "and IBS, but benefits are strain-specific. General gut health claims lack strong evidence.",
    ),

    # ── Heart Health ────────────────────────────────────────────────────
    (
        "What are the warning signs of a stroke?",
        "Heart Health",
        "Warning signs follow the FAST acronym: Face drooping, Arm weakness, Speech difficulty, "
        "Time to call emergency services. Other signs include sudden confusion, vision problems, "
        "severe headache, and dizziness.",
    ),
    (
        "How can cholesterol levels be managed without medication?",
        "Heart Health",
        "Lifestyle changes include eating soluble fiber, reducing saturated and trans fats, "
        "regular exercise, maintaining healthy weight, and limiting alcohol.",
    ),

    # ── Respiratory ─────────────────────────────────────────────────────
    (
        "What are the treatment options for sleep apnea?",
        "Respiratory",
        "Treatment includes CPAP therapy, oral appliances, weight loss, positional therapy, "
        "lifestyle changes, and in severe cases surgery.",
    ),

    # ── Endocrine ───────────────────────────────────────────────────────
    (
        "What are the symptoms of thyroid disorders?",
        "Endocrine",
        "Hypothyroidism symptoms include fatigue, weight gain, cold intolerance, and depression. "
        "Hyperthyroidism symptoms include weight loss, rapid heartbeat, anxiety, and heat intolerance.",
    ),

    # ── Aging ───────────────────────────────────────────────────────────
    (
        "What lifestyle habits are associated with healthy aging?",
        "Aging",
        "Healthy aging habits include regular physical activity, social engagement, mental stimulation, "
        "balanced diet, adequate sleep, stress management, and preventive healthcare.",
    ),
    (
        "What can be done to reduce the risk of dementia?",
        "Aging",
        "Risk reduction includes regular exercise, cognitive stimulation, social engagement, "
        "managing cardiovascular risk factors, healthy diet, and adequate sleep.",
    ),

    # ── Pediatrics ──────────────────────────────────────────────────────
    (
        "What vaccines are recommended for children in their first year?",
        "Pediatrics",
        "First-year vaccines typically include hepatitis B, DTaP, IPV (polio), Hib, PCV13, "
        "rotavirus, and influenza, following the CDC recommended schedule.",
    ),
    (
        "How can childhood obesity be prevented?",
        "Pediatrics",
        "Prevention includes encouraging physical activity, limiting screen time, providing "
        "nutritious meals, reducing sugary drinks, and modeling healthy behaviors.",
    ),

    # ── Travel Health ───────────────────────────────────────────────────
    (
        "What health precautions should travelers take before international trips?",
        "Travel Health",
        "Precautions include checking required vaccinations, carrying medications, "
        "travel health insurance, food and water safety awareness, and consulting a travel medicine clinic.",
    ),

    # ── Environmental Health ────────────────────────────────────────────
    (
        "How does air pollution affect health?",
        "Environmental Health",
        "Air pollution increases risk of respiratory diseases, cardiovascular disease, lung cancer, "
        "and premature death. Fine particulate matter (PM2.5) is especially harmful.",
    ),

    # ── Cross-cutting questions ─────────────────────────────────────────
    (
        "What is the relationship between obesity and type 2 diabetes?",
        "Diabetes",
        "Obesity is a major risk factor for type 2 diabetes due to insulin resistance. "
        "Excess body fat, especially visceral fat, impairs insulin signaling.",
    ),
    (
        "How does exercise benefit mental health?",
        "Mental Health",
        "Exercise reduces symptoms of depression and anxiety through endorphin release, "
        "improved sleep, stress reduction, and increased self-efficacy.",
    ),
    (
        "How does chronic stress affect physical health?",
        "Mental Health",
        "Chronic stress increases risk of cardiovascular disease, weakens immune function, "
        "disrupts sleep, raises blood pressure, and can worsen chronic conditions.",
    ),
]
