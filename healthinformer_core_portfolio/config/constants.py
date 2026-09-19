"""PubMed search queries and topic categories that define the corpus scope.

Each query maps to a human-readable category used for metadata tagging.
Comprehensive coverage of health/wellness topics the general public asks about.
~130 queries × 200 results each → ~20K+ unique articles.
"""

# Maps a PubMed search query → a short topic category label.
# The label travels with every chunk for filtering and citation display.
PUBMED_QUERIES: dict[str, str] = {

    # ── Chronic Diseases ────────────────────────────────────────────────
    "type 2 diabetes management": "Diabetes",
    "hypertension treatment guidelines": "Hypertension",
    "asthma management adults": "Asthma",
    "heart failure patient education": "Heart Failure",
    "chronic kidney disease prevention": "Kidney Disease",

    # ── Cancer Screening & Prevention ───────────────────────────────────
    "breast cancer screening recommendations": "Cancer Screening",
    "colorectal cancer prevention": "Cancer Prevention",
    "lung cancer risk factors smoking": "Cancer Risk",

    # ── Diet & Nutrition ────────────────────────────────────────────────
    "healthy diet cardiovascular health": "Nutrition",
    "Mediterranean diet health benefits": "Nutrition",
    "intermittent fasting evidence": "Nutrition",
    "vitamin D supplementation": "Nutrition",
    "gut microbiome diet": "Nutrition",
    "sugar intake health effects": "Nutrition",
    "plant based diet health outcomes": "Nutrition",
    "omega 3 fatty acids benefits": "Nutrition",
    "hydration water intake health": "Nutrition",
    "caffeine health effects": "Nutrition",
    "food allergies management": "Nutrition",
    "gluten intolerance celiac disease": "Nutrition",
    "iron deficiency anemia diet": "Nutrition",
    "calcium intake bone health": "Nutrition",
    "protein intake muscle health": "Nutrition",
    "ultra processed food health risks": "Nutrition",

    # ── Fitness & Exercise ──────────────────────────────────────────────
    "physical activity guidelines adults": "Exercise",
    "strength training health benefits": "Exercise",
    "cardio vs resistance training": "Exercise",
    "exercise mental health benefits": "Exercise",
    "stretching flexibility injury prevention": "Exercise",
    "yoga health benefits evidence": "Exercise",
    "walking health benefits": "Exercise",
    "high intensity interval training": "Exercise",
    "exercise during pregnancy": "Exercise",
    "physical activity older adults": "Exercise",
    "sports injury prevention": "Exercise",
    "sedentary behavior health risks": "Exercise",

    # ── Weight Management ───────────────────────────────────────────────
    "obesity weight management interventions": "Weight Management",

    # ── Mental Health & Psychology ──────────────────────────────────────
    "depression treatment options": "Mental Health",
    "anxiety disorders cognitive behavioral therapy": "Mental Health",
    "stress management techniques evidence": "Mental Health",
    "mindfulness meditation health": "Mental Health",
    "burnout prevention strategies": "Mental Health",
    "grief coping mechanisms": "Mental Health",
    "ADHD adult management": "Mental Health",
    "PTSD treatment evidence": "Mental Health",
    "social isolation loneliness health effects": "Mental Health",
    "therapy psychotherapy effectiveness": "Mental Health",
    "postpartum depression treatment": "Mental Health",
    "bipolar disorder management": "Mental Health",
    "eating disorders treatment": "Mental Health",
    "substance abuse treatment options": "Mental Health",
    "self care mental health": "Mental Health",
    "screen time mental health effects": "Mental Health",

    # ── Sleep Health ────────────────────────────────────────────────────
    "sleep disorders insomnia management": "Sleep Health",

    # ── Sexual & Reproductive Health ────────────────────────────────────
    "prenatal care guidelines": "Prenatal Health",
    "menopause symptom management": "Women's Health",
    "STI prevention screening": "Sexual Health",
    "contraception methods effectiveness": "Sexual Health",
    "erectile dysfunction treatment": "Sexual Health",
    "fertility optimization": "Sexual Health",
    "polycystic ovary syndrome management": "Women's Health",
    "endometriosis treatment": "Women's Health",
    "testosterone levels health": "Sexual Health",

    # ── Skin & Dermatology ──────────────────────────────────────────────
    "sunscreen skin cancer prevention": "Dermatology",
    "acne treatment options": "Dermatology",
    "eczema management": "Dermatology",
    "psoriasis treatment": "Dermatology",
    "hair loss treatment options": "Dermatology",
    "wound healing nutrition": "Dermatology",

    # ── Allergies & Immune System ───────────────────────────────────────
    "seasonal allergy treatment": "Allergies",
    "immune system strengthening evidence": "Allergies",
    "autoimmune disease management": "Allergies",
    "anaphylaxis prevention management": "Allergies",

    # ── Infectious Disease & Vaccines ───────────────────────────────────
    "influenza vaccination effectiveness": "Vaccines",
    "COVID-19 long term effects": "COVID-19",
    "antibiotic resistance patient education": "Antibiotics",

    # ── Substance Use ───────────────────────────────────────────────────
    "alcohol health effects moderate drinking": "Substance Use",
    "smoking cessation methods effectiveness": "Substance Use",
    "cannabis health effects evidence": "Substance Use",
    "vaping e-cigarette health risks": "Substance Use",

    # ── Common Symptoms & Pain ──────────────────────────────────────────
    "headache migraine treatment": "Pain Management",
    "back pain management evidence": "Pain Management",
    "fatigue causes treatment": "Pain Management",
    "joint pain arthritis management": "Pain Management",
    "chronic pain non opioid treatment": "Pain Management",
    "nausea vomiting management": "Pain Management",
    "dizziness vertigo treatment": "Pain Management",
    "constipation treatment": "Pain Management",
    "acid reflux GERD management": "Pain Management",

    # ── Dental & Oral Health ────────────────────────────────────────────
    "oral health systemic disease connection": "Oral Health",
    "periodontal disease prevention": "Oral Health",
    "teeth grinding bruxism treatment": "Oral Health",

    # ── Eye & Ear Health ────────────────────────────────────────────────
    "screen time eye health digital eye strain": "Eye Health",
    "hearing loss prevention": "Ear Health",
    "dry eye treatment": "Eye Health",

    # ── Gut & Digestive Health ──────────────────────────────────────────
    "irritable bowel syndrome management": "Digestive Health",
    "probiotics health evidence": "Digestive Health",
    "inflammatory bowel disease treatment": "Digestive Health",
    "celiac disease management": "Digestive Health",
    "liver health fatty liver disease": "Digestive Health",

    # ── Heart & Vascular ────────────────────────────────────────────────
    "blood pressure natural management": "Heart Health",
    "atrial fibrillation patient education": "Heart Health",
    "peripheral artery disease": "Heart Health",
    "stroke prevention risk factors": "Heart Health",
    "cholesterol management lifestyle": "Heart Health",

    # ── Respiratory ─────────────────────────────────────────────────────
    "COPD management patient education": "Respiratory",
    "sleep apnea treatment options": "Respiratory",
    "allergic rhinitis treatment": "Respiratory",

    # ── Endocrine ───────────────────────────────────────────────────────
    "thyroid disorders management": "Endocrine",
    "adrenal fatigue evidence": "Endocrine",
    "insulin resistance prevention": "Endocrine",

    # ── Aging & Longevity ───────────────────────────────────────────────
    "osteoporosis prevention treatment": "Bone Health",
    "fall prevention elderly": "Geriatrics",
    "healthy aging strategies evidence": "Aging",
    "dementia prevention risk factors": "Aging",
    "sarcopenia muscle loss aging": "Aging",
    "cognitive decline prevention": "Aging",

    # ── Children & Adolescents ──────────────────────────────────────────
    "childhood vaccination schedule": "Pediatrics",
    "adolescent mental health": "Pediatrics",
    "childhood obesity prevention": "Pediatrics",
    "pediatric asthma management": "Pediatrics",

    # ── Travel & Environmental ──────────────────────────────────────────
    "travel health vaccination": "Travel Health",
    "air pollution health effects": "Environmental Health",
    "heat stroke prevention treatment": "Environmental Health",
}

# All unique topic categories (auto-derived for reference)
TOPIC_CATEGORIES: list[str] = sorted(set(PUBMED_QUERIES.values()))
