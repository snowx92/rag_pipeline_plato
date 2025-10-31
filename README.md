# RAG Resume Evaluation System - Plato Assignment

Hi! This is my solution for the Plato RAG Engineering challenge. I built a system that scores job candidates by matching their resumes to job descriptions using a RAG pipeline.

The core idea: take a resume and a job posting, extract the relevant bits using vector search, and produce a detailed JSON evaluation with scores and explanations. The system needs to be consistent (same input = same output every time) and reliable (always returns valid JSON).

---

## Getting Started

### Install Dependencies

```bash
pip install -r requirment.txt
```

You'll need: `openai`, `chromadb`, `sentence-transformers`, `jsonschema`, `numpy`, `tqdm`

### Run It

**With OpenAI (smarter evaluations):**
```bash
set OPENAI_API_KEY=your-key-here
python -m main --jd-txt jd.txt --resume resume.txt --out result.json
```

**Without API key (rule-based scoring):**
```bash
python -m main --jd-txt jd.txt --resume resume.txt --mode rules --out result.json
```

The output is a JSON file with scores, strengths, gaps, and detailed breakdowns.

---

## Pipeline Architecture

Here's the complete flow of how the system processes a candidate:


```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INPUT                             │
│                                                                 │
│  ┌──────────────────┐              ┌─────────────────────┐    │
│  │  Job Description │              │   Resume (CV)       │    │
│  │  ─────────────── │              │   ────────────      │    │
│  │  • Title         │              │   • Work history    │    │
│  │  • Sector        │              │   • Skills          │    │
│  │  • Requirements  │              │   • Dates           │    │
│  │  • Description   │              │   • Education       │    │
│  └──────────────────┘              └─────────────────────┘    │
│           │                                    │                │
└───────────┼────────────────────────────────────┼────────────────┘
            │                                    │
            ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MAIN.PY (Entry Point)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Parse CLI arguments                                   │  │
│  │  • Load JD and Resume files                              │  │
│  │  • Route to Rules or LLM mode                            │  │
│  │  • Validate final output                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE.PY (Orchestrator)                   │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Step 1: PARSE RESUME (parse_resume.py)                  │  │
│   │ ──────────────────────────────────────────────────────  │  │
│   │ Input:  Resume text + JD requirements                   │  │
│   │ Output: {                                               │  │
│   │   "skills": ["Python", "Django", "AWS"],                │  │
│   │   "experience_years": 5.2,                              │  │
│   │   "evidence_lines": [                                   │  │
│   │     "Built Python apps 2018-01 to 2020-06",             │  │
│   │     "Led team of 5 engineers using AWS"                 │  │
│   │   ]                                                     │  │
│   │ }                                                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Step 2: EMBED & RETRIEVE (retrieve.py)                  │  │
│   │ ──────────────────────────────────────────────────────  │  │
│   │                                                         │  │
│   │  ┌────────────────────────────────────────────────┐    │  │
│   │  │ ChromaDB (In-Memory Vector Database)          │    │  │
│   │  │ ───────────────────────────────────────────   │    │  │
│   │  │ • Each resume line = 1 document               │    │  │
│   │  │ • Embedded using sentence-transformers        │    │  │
│   │  │ • Cosine similarity search                    │    │  │
│   │  └────────────────────────────────────────────────┘    │  │
│   │                                                         │  │
│   │  For each "Proficiency in X" requirement:              │  │
│   │  ┌─────────────────────────────────────────────┐       │  │
│   │  │ Query: "Python"                             │       │  │
│   │  │ → Retrieve top-3 most similar resume lines  │       │  │
│   │  │ → Return with similarity distances          │       │  │
│   │  └─────────────────────────────────────────────┘       │  │
│   │                                                         │  │
│   │ Output: {                                               │  │
│   │   "Proficiency in Python": [                            │  │
│   │     {id: "res-0001", text: "Built...", dist: 0.12},     │  │
│   │     {id: "res-0005", text: "Led...", dist: 0.23}        │  │
│   │   ],                                                    │  │
│   │   "Proficiency in AWS": [...]                           │  │
│   │ }                                                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Step 3: SCORING (Two Paths)                             │  │
│   └─────────────────────────────────────────────────────────┘  │
│            │                                                    │
│            ├──────────────────┬─────────────────────────────┐  │
│            ▼                  ▼                             │  │
│   ┌────────────────┐  ┌──────────────────────────────────┐ │  │
│   │  RULES MODE    │  │         LLM MODE                 │ │  │
│   │  (scorer.py)   │  │  (prompt.py + llm_evaluator.py) │ │  │
│   │ ──────────────  │  │  ────────────────────────────── │ │  │
│   │                │  │                                  │ │  │
│   │ Deterministic: │  │  1. Build Prompt (prompt.py)    │ │  │
│   │                │  │     ┌───────────────────────┐   │ │  │
│   │ Tech score =   │  │     │ SYSTEM: Instructions │   │ │  │
│   │  matches/total │  │     │ SCHEMA: JSON Schema  │   │ │  │
│   │  × 100         │  │     │ JOB: Requirements    │   │ │  │
│   │                │  │     │ PARSED_RESUME: Data  │   │ │  │
│   │ Exp score =    │  │     │ RETRIEVAL: Hits      │   │ │  │
│   │  f(years)      │  │     │ TASK: Instructions   │   │ │  │
│   │                │  │     └───────────────────────┘   │ │  │
│   │ Cultural =     │  │                                  │ │  │
│   │  soft signals  │  │  2. Call OpenAI API              │ │  │
│   │                │  │     • model: gpt-4o-mini         │ │  │
│   │ Overall =      │  │     • temperature: 0             │ │  │
│   │  0.4T + 0.4E   │  │     • seed: 42                   │ │  │
│   │  + 0.2C        │  │     • response_format: json      │ │  │
│   │                │  │                                  │ │  │
│   │ Generate       │  │  3. Parse JSON Response          │ │  │
│   │ breakdowns,    │  │                                  │ │  │
│   │ strengths,     │  │  4. Validate Schema              │ │  │
│   │ gaps           │  │     ├─ Valid? → Return           │ │  │
│   │                │  │     └─ Invalid? ↓                │ │  │
│   │                │  │                                  │ │  │
│   │                │  │  5. Repair Attempt               │ │  │
│   │                │  │     ├─ Valid? → Return           │ │  │
│   │                │  │     └─ Invalid? → Fallback       │ │  │
│   │                │  │                    ↓             │ │  │
│   │                │  │          ┌──────────────────┐   │ │  │
│   │ ◄──────────────┼──┼──────────┤ Use Rules Mode   │   │ │  │
│   │                │  │          └──────────────────┘   │ │  │
│   └────────────────┘  └──────────────────────────────────┘ │  │
│            │                           │                    │  │
│            └─────────┬─────────────────┘                    │  │
│                      ▼                                       │  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Step 4: VALIDATE (schema.py)                            │  │
│   │ ──────────────────────────────────────────────────────  │  │
│   │ • Check against JSON Schema (Draft-07)                  │  │
│   │ • Ensure all required fields present                    │  │
│   │ • Verify data types and ranges                          │  │
│   │ • No additional properties allowed                      │  │
│   │ • Raise error if invalid                                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                      ↓                                       │  │
└──────────────────────┼───────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VALIDATED OUTPUT                         │
│                                                                 │
│  {                                                              │
│    "overallScore": 85,                                          │
│    "technicalSkillsScore": 80,                                  │
│    "experienceScore": 90,                                       │
│    "culturalFitScore": 75,                                      │
│    "matchSummary": "Candidate meets most requirements...",      │
│    "strengthsHighlights": ["Strong Python", "10 years exp"],    │
│    "improvementAreas": ["No Docker", "Missing AWS cert"],       │
│    "detailedBreakdown": {                                       │
│      "technicalSkills": [...],                                  │
│      "experience": [...],                                       │
│      "educationAndCertifications": [...],                       │
│      "culturalFitAndSoftSkills": [...]                          │
│    },                                                           │
│    "redFlags": []                                               │
│  }                                                              │
│                                                                 │
│  ✅ Schema-valid JSON                                           │
│  ✅ Deterministic (same input = same output)                    │
│  ✅ Evidence-grounded explanations                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## How the Pipeline Works

### Stage 1: Parse the Resume

The parser (`src/parse_resume.py`) extracts:
- **Skills**: Matches against job requirements like "Proficiency in Python"
- **Experience years**: Calculates from date ranges (e.g., 2018-03 to 2021-06 = 3.25 years)
- **Evidence lines**: Sentences that mention the matched skills

This gives us structured data to work with.

### Stage 2: Embed & Retrieve (The RAG Part)

For each skill requirement in the job description:
1. Embed the requirement (e.g., "Python")
2. Embed all resume lines
3. Store in ChromaDB (in-memory vector database)
4. Query: "Which resume lines are most similar to this requirement?"
5. Get top-3 matches with distance scores

This is the retrieval-augmented part - we're finding evidence in the resume that supports (or doesn't support) each requirement.

Example:
```
Requirement: "Proficiency in Python"
Retrieved lines:
1. [distance=0.42] "Built Python microservices using Django and FastAPI..."
2. [distance=0.68] "Led team of 5 engineers on backend projects..."
3. [distance=0.89] "Improved system performance by 30%..."
```

Lower distance = better match. The first line clearly mentions Python, so it scores well.

### Stage 3: Score & Explain

I implemented two scoring approaches:

**Rules Mode (deterministic)**
- Technical score = (skills matched / skills required) × 100
- Experience score = based on years vs requirement
- Cultural fit = looks for collaboration/ownership keywords
- Overall = 40% tech + 40% exp + 20% cultural

This is pure math - same inputs always give the same outputs.

**LLM Mode (with safety net)**
1. Build a structured prompt with the job, resume, and retrieval results
2. Include the JSON schema in the prompt
3. Call OpenAI GPT-4o-mini with `temperature=0` and `seed=42`
4. Parse the JSON response
5. Validate against schema
6. If invalid → try to repair it
7. If still broken → fall back to rules mode

The LLM mode is smarter but the rules mode guarantees we always get output.

### Stage 4: Validate

Every output goes through JSON Schema validation (Draft-07). This ensures:
- All required fields are present
- Scores are integers 0-100
- Arrays don't exceed size limits
- Data types are correct

If validation fails, the system raises an error with details.

---

## The Consistency Guarantee

The assignment requires <1% score variance across repeated runs. I exceeded this - the system produces **identical** outputs for the same input.

**How I achieved this:**

**Rules Mode:**
- Pure deterministic logic - no randomness at all
- Tested: Ran 10 times, got identical results every time (same hash)

**LLM Mode:**
- `temperature=0` - no sampling randomness
- `seed=42` - OpenAI's reproducibility parameter
- All data sorted alphabetically before processing
- Distance scores rounded to 8 decimals
- JSON serialized with `sort_keys=True`
- Tested: Ran 3 times, got identical results (same hash)

You can verify this yourself in `evaluation.ipynb` - run the cells and see the hash values match.

---

## Project Structure

```
cv_project/
├── main.py              # CLI interface
├── src/
│   ├── pipeline.py      # Main orchestrator
│   ├── parse_resume.py  # Extract skills, years, evidence
│   ├── retrieve.py      # Vector DB + retrieval logic
│   ├── scorer.py        # Rule-based scoring
│   ├── llm_evaluator.py # OpenAI integration
│   ├── prompt.py        # Build LLM prompts
│   └── schema.py        # JSON validation
├── tests/               # Unit tests for everything
├── evaluation.ipynb     # Consistency verification
├── jd.txt              # Sample job description
├── resume.txt          # Sample resume
└── result.json         # Example output
```

---

## Example Output

Using the provided sample files (Growth Marketer job + resume):

```json
{
  "overallScore": 85,
  "technicalSkillsScore": 80,
  "experienceScore": 90,
  "culturalFitScore": 75,
  "matchSummary": "The candidate meets most technical requirements...",
  "strengthsHighlights": [
    "Strong experience in Attribution and CRM",
    "6.9 years of relevant experience",
    "Proven track record in Copywriting"
  ],
  "improvementAreas": [
    "Enhance skills in Google Ads",
    "Gain proficiency in WordPress"
  ],
  "detailedBreakdown": {
    "technicalSkills": [
      {
        "requirement": "Proficiency in Attribution",
        "present": true,
        "evidence": "Delivered 5 projects using Copywriting, SEO, Attribution...",
        "gapPercentage": 0,
        "missingDetail": ""
      },
      {
        "requirement": "Proficiency in Google Ads",
        "present": false,
        "evidence": "",
        "gapPercentage": 100,
        "missingDetail": "No evidence of proficiency in Google Ads."
      }
    ],
    "experience": [...],
    "educationAndCertifications": [...],
    "culturalFitAndSoftSkills": [...]
  },
  "redFlags": []
}
```

The candidate scored 85 overall because they have strong experience (6.9 years vs 3 required) and most key skills, but are missing Google Ads and WordPress.

---

## Testing

**Run unit tests:**
```bash
pytest
```

**Verify consistency:**
```bash
jupyter notebook evaluation.ipynb
# Run all cells - should show identical hashes across runs
```

**Manual test:**
```bash
python -m main --jd-txt jd.txt --resume resume.txt --mode rules > out1.json
python -m main --jd-txt jd.txt --resume resume.txt --mode rules > out2.json
fc out1.json out2.json  # Should be identical
```

All tests pass and consistency is verified.

---

## Scaling to the Full Dataset

The assignment provides 500 jobs × 10 applicants = 5,000 CVs in `universal_jobs_shard_01.jsonl.gz`. Here's how I'd scale this system to handle them:

### 1. Batch Processing
Process all 5,000 candidate-job pairs:

```python
import gzip, json

with gzip.open('universal_jobs_shard_01.jsonl.gz', 'rt') as f:
    for line in f:
        job_record = json.loads(line)
        job = job_record['job']

        for applicant in job_record['applicants']:
            resume = format_resume(applicant['candidate'])
            score = evaluate(job, resume)
            save_to_db(score)
```

### 2. Build a Candidate Vector Database

Instead of processing one-by-one, build a persistent database:

```python
# Store all 5,000 CVs with embeddings
db = build_persistent_chromadb()
for job_record in all_jobs:
    for applicant in job_record['applicants']:
        db.add(
            documents=[format_resume(applicant['candidate'])],
            metadatas=[{
                'job_id': job['id'],
                'name': applicant['candidate']['name'],
                'sector': job['sector']
            }],
            ids=[f"{job['id']}-{applicant['candidate']['email']}"]
        )
```

### 3. Reverse Search: Job → Best Candidates

Given a new job, find the best matches from 5,000 CVs:

```python
def find_top_candidates(job_description, n=10):
    # Embed job requirements
    job_embedding = embed(job_description)

    # Search database
    results = db.query(
        query_embeddings=[job_embedding],
        n_results=n,
        where={"sector": job["sector"]}  # Optional filter
    )

    # Score each candidate
    candidates = []
    for cv in results:
        score = evaluate(job, cv)
        candidates.append((cv, score))

    return sorted(candidates, key=lambda x: x[1]['overallScore'], reverse=True)
```

This enables:
- **Talent search**: "Find all Senior Backend Engineers with AWS experience"
- **Batch hiring**: Score all 5,000 candidates for a new job in minutes
- **Analytics**: "Which sector has the most qualified candidates?"

### 4. Performance Optimizations

- **Cache embeddings**: Don't re-embed the same resume twice
- **Batch API calls**: Process 10 candidates per OpenAI request (if supported)
- **Parallel processing**: Use async/threading for I/O-bound operations
- **Persistent storage**: Use Pinecone or Weaviate for production scale
- **Results database**: Store scores in PostgreSQL for querying/filtering

### 5. Simple Web Interface

Build a Flask/FastAPI dashboard:
- Upload new jobs/resumes
- View scoring results with explanations
- Filter candidates by score thresholds
- Export to CSV for hiring managers

---

## The LLM Prompt Structure

When using LLM mode, the system builds a carefully structured prompt with 6 sections. Here's what gets sent to OpenAI:

```
┌─────────────────────────────────────────────────────────────────┐
│                    📝 PROMPT STRUCTURE                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SECTION 1: SYSTEM                                              │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  You are an evaluation service.                                 │
│  Return a SINGLE JSON object ONLY that strictly validates       │
│  against the provided JSON_SCHEMA.                              │
│  Do not include explanations, markdown, or any extra text.      │
│  If uncertain, make the best deterministic judgment using       │
│  only the provided evidence.                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 2: JSON_SCHEMA                                         │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  {                                                              │
│    "$schema": "http://json-schema.org/draft-07/schema#",        │
│    "properties": {                                              │
│      "overallScore": {"type": "integer", "min": 0, "max": 100}, │
│      "technicalSkillsScore": {...},                             │
│      "experienceScore": {...},                                  │
│      "culturalFitScore": {...},                                 │
│      "matchSummary": {"type": "string"},                        │
│      "strengthsHighlights": {"type": "array", "maxItems": 3},   │
│      "improvementAreas": {"type": "array", "maxItems": 3},      │
│      "detailedBreakdown": {                                     │
│        "technicalSkills": [...],                                │
│        "experience": [...],                                     │
│        "educationAndCertifications": [...],                     │
│        "culturalFitAndSoftSkills": [...]                        │
│      },                                                         │
│      "redFlags": [...]                                          │
│    },                                                           │
│    "required": ["overallScore", "technicalSkillsScore", ...]    │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 3: JOB                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  {                                                              │
│    "title": "Growth Marketer",                                  │
│    "sector": "Marketing",                                       │
│    "location": "Cairo, Egypt",                                  │
│    "description": "We are hiring a Growth Marketer...",         │
│    "requirements": [                                            │
│      "3+ years of relevant experience",                         │
│      "Bachelor's degree or equivalent experience",              │
│      "Proficiency in Attribution",                              │
│      "Proficiency in Copywriting",                              │
│      "Proficiency in CRM",                                      │
│      "Proficiency in Google Ads",                               │
│      "Proficiency in Meta Ads",                                 │
│      "Proficiency in WordPress"                                 │
│    ]                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 4: PARSED_RESUME                                       │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  {                                                              │
│    "skills": ["Attribution", "CRM", "Meta Ads"],                │
│    "experience_years": 6.9,                                     │
│    "evidence_lines": [                                          │
│      "Delivered 5 projects using Copywriting, SEO,              │
│       Attribution with measurable KPIs.",                       │
│      "Delivered 2 projects using Content Strategy,              │
│       HubSpot, CRM with measurable KPIs.",                      │
│      "Delivered 5 projects using Content Strategy, SEM,         │
│       Meta Ads with measurable KPIs."                           │
│    ]                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 5: RETRIEVAL (The RAG Evidence)                        │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  {                                                              │
│    "Proficiency in Attribution": [                              │
│      {                                                          │
│        "distance": 0.6081,                                      │
│        "id": "res-0000",                                        │
│        "text": "Delivered 5 projects using Copywriting,         │
│                 SEO, Attribution with measurable KPIs."         │
│      },                                                         │
│      {"distance": 0.8081, "text": "..."},                       │
│      {"distance": 0.8835, "text": "..."}                        │
│    ],                                                           │
│    "Proficiency in Google Ads": [                               │
│      {                                                          │
│        "distance": 0.5885,                                      │
│        "text": "Delivered 5 projects using Content Strategy,    │
│                 SEM, Meta Ads with measurable KPIs."            │
│      },                                                         │
│      {"distance": 0.7741, "text": "..."},                       │
│      {"distance": 0.9129, "text": "..."}                        │
│    ],                                                           │
│    ... (one entry per requirement)                              │
│  }                                                              │
│                                                                 │
│  Note: Lower distance = better match                            │
│  Distance < 0.5  = Excellent match                              │
│  Distance 0.5-0.7 = Good match                                  │
│  Distance > 0.7  = Weak match                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SECTION 6: TASK                                                │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Using JOB, PARSED_RESUME, and RETRIEVAL, produce scores and    │
│  explanations that match JSON_SCHEMA.                           │
│                                                                 │
│  - Be faithful to the evidence                                  │
│  - Scores are integers 0..100                                   │
│  - All arrays and fields required by the schema must be present │
│  - If an item is missing, explain the gap in missingDetail      │
│  - No extra keys                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                  🤖 Sent to OpenAI GPT-4o-mini
                     (temperature=0, seed=42)
                              ↓
                  📊 Returns validated JSON
```

### Why This Prompt Works

1. **Schema-First Design**: The LLM sees the exact JSON structure required - no guessing
2. **Evidence-Grounded**: The RETRIEVAL section provides actual resume lines with similarity scores
3. **Clear Instructions**: "SINGLE JSON object ONLY" prevents the LLM from adding explanatory text
4. **Deterministic**: All data is sorted alphabetically, distances rounded - same prompt every time
5. **Fail-Safe**: If LLM output doesn't match schema → repair attempt → fallback to rules

### Example Prompt Output

For the sample Growth Marketer job and resume, the prompt is ~7,700 characters with:
- Full JSON Schema (2,500 chars)
- Job requirements (sorted)
- 3 matched skills + 6.9 years experience
- 18 retrieval hits (6 requirements × 3 hits each)

You can see the actual prompt by running:
```bash
python test_prompt_run.py
```

This generates `generated_prompt.txt` with the exact text sent to the LLM.

---

## Design Decisions

### Why Two Modes?

**Rules mode** is:
- Free (no API costs)
- Fast (~500ms per evaluation)
- 100% deterministic
- Transparent (you can see exactly how scores are calculated)
- Works offline

**LLM mode** is:
- Smarter (better at nuanced evaluations)
- More flexible (handles edge cases better)
- Natural language explanations
- But costs money and requires internet

Having both means the system is always reliable - if OpenAI is down or you hit rate limits, it falls back to rules automatically.

### Why Per-Requirement Retrieval?

Instead of one big "find similar resume content" query, I query once per requirement:
- "Proficiency in Python" → get evidence for Python
- "Proficiency in AWS" → get evidence for AWS
- etc.

This makes it easy to trace which resume lines support which requirements. The detailed breakdown can point to specific evidence or say "no evidence found."

### Why Embed the Schema in the Prompt?

The LLM gets the full JSON Schema in the prompt. This:
- Reduces hallucination (LLM knows exact structure required)
- Enables self-correction (it can match its output to the spec)
- Works with OpenAI's `response_format="json_object"` mode

---

## Current Limitations

I made some tradeoffs to ship a working system:

**1. Resume Parsing**
- Uses simple regex-based extraction
- Doesn't handle PDF files (would need PyPDF2 or similar)
- Doesn't extract education/certifications (placeholder logic only)
- Works well for structured text resumes

**2. Skill Matching**
- Exact string matching only
- Doesn't understand synonyms ("JavaScript" ≠ "JS")
- Could be improved with a skill taxonomy or fuzzy matching

**3. Scalability**
- In-memory ChromaDB doesn't persist across runs
- For 5,000+ CVs, would need persistent storage
- Single-threaded processing (could parallelize)

**4. Context Limits**
- Very short resumes (<3 lines) might not have enough evidence
- Very long resumes might need chunking

These are all solvable - I prioritized getting the core RAG pipeline working correctly first.

---

## Technical Choices

**Vector DB**: ChromaDB
- Easy to use, good for prototyping
- Works in-memory for single evaluations
- Would switch to Pinecone/Weaviate for production

**Embedding Model**: sentence-transformers (default)
- Good quality, runs locally
- Free (no API costs)
- Could upgrade to OpenAI embeddings for better semantic understanding

**LLM**: GPT-4o-mini
- Good balance of quality and cost (~$0.001 per evaluation)
- Supports JSON mode
- Fast enough for real-time use
- `temperature=0` + `seed=42` for consistency

**Validation**: jsonschema (Draft-07)
- Industry standard
- Clear error messages
- Easy to extend with new requirements

---

## Files Included

**Code:**
- `main.py` - CLI interface
- `src/` - All pipeline components (7 modules)
- `tests/` - Unit tests for each module

**Sample Data:**
- `jd.txt` - Plain text job description
- `jd.json` - JSON format job description
- `resume.txt` - Sample resume
- `result.json` - Example evaluation output

**Testing:**
- `evaluation.ipynb` - Consistency verification (10 runs rules, 3 runs LLM - all identical)
- `pytest.ini` - Test configuration

**Documentation:**
- `README.md` - This file
- `requirment.txt` - Dependencies

---

## Assignment Requirements Met

✅ **Embed and retrieve job + resume context** - ChromaDB with sentence-transformers
✅ **Integrate with LLM** - OpenAI GPT-4o-mini with JSON mode
✅ **Ensure scoring consistency** - <1% margin achieved (actually 0% - identical outputs)
✅ **Apply custom parsing rules** - Resume parser extracts skills/years/evidence
✅ **Produce exact JSON format** - Validated against Draft-07 schema
✅ **Technical correctness** - All components unit tested
✅ **Code clarity** - Documented code + comprehensive README

---

## Running the System

**Basic usage:**
```bash
# With your own files
python -m main --jd-txt myjob.txt --resume myresume.txt --out score.json

# Or use JSON format for JD
python -m main --jd myjob.json --resume myresume.txt --out score.json

# Debug mode (see what's happening)
python -m main --jd-txt jd.txt --resume resume.txt --debug --print-prompt
```

**Available options:**
- `--mode` - `llm` (default) or `rules`
- `--k` - Top-k retrieval hits per requirement (default 3)
- `--model` - OpenAI model (default `gpt-4o-mini`)
- `--seed` - Random seed for determinism (default 42)
- `--out` - Output file path (default: print to console)
- `--debug` - Verbose logging
- `--print-prompt` - Show the LLM prompt

---

## Summary

This system evaluates candidates by:
1. Parsing their resume to extract skills and experience
2. Using vector search to find relevant evidence for each requirement
3. Scoring them with either deterministic rules or an LLM
4. Producing validated JSON with detailed explanations

It's consistent (0% variance), reliable (fallback logic), and production-ready (error handling, tests, validation).

The RAG approach makes evaluations evidence-based - every score can be traced back to specific resume content. This is more transparent than a black-box classifier and easier to explain to hiring managers.

For scaling to 5,000 CVs, the next steps would be building a persistent vector database, adding batch processing, and creating a simple UI for talent search and candidate ranking.

Thanks for reviewing my submission!

---

**Assignment**: Plato RAG Engineering Challenge
**Submission Date**: [Add date]
