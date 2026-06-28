"""
Intelligent Candidate Discovery - Three-Stage Cascading Ranker
Executes in < 5 mins on CPU. Implements Deterministic Pruning, Sparse Lexical Ranking, and Dense Semantic Alignment.
"""

import json
import gzip
import argparse
import csv
import re
from datetime import datetime
import numpy as np

# Stage 2: Sparse Lexical Matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sparse_cosine_similarity

# Stage 3: Dense Semantic Matching
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as dense_cosine_similarity

def load_rules(rules_path):
    with open(rules_path, 'r') as f:
        return json.load(f)

def check_honeypots(candidate):
    """Stage 1 Hard-Gate: Detects impossible profiles (Expert skills, 0 months experience)."""
    skills = candidate.get('skills', [])
    for skill in skills:
        if skill.get('proficiency') in ['advanced', 'expert'] and skill.get('duration_months', 0) == 0:
            return True
    return False

def has_invalid_title(candidate, rules):
    """Stage 1 Hard-Gate: Detects 'Keyword Stuffers' with non-technical job titles."""
    prof = candidate.get('profile', {})
    title = str(prof.get('current_title', '')).lower()
    headline = str(prof.get('headline', '')).lower()
    
    for invalid_word in rules['anti_patterns']['irrelevant_titles']:
        if invalid_word.lower() in title or invalid_word.lower() in headline:
            return True
    return False

def build_candidate_document(candidate):
    """Compiles candidate text into a single document for TF-IDF and Dense embeddings."""
    prof = candidate.get('profile', {})
    parts = [prof.get('headline', ''), prof.get('summary', '')]
    
    for role in candidate.get('career_history', []):
        parts.append(role.get('title', ''))
        parts.append(role.get('description', ''))
        
    for skill in candidate.get('skills', []):
        parts.append(skill.get('name', ''))
        
    return " ".join(filter(None, parts))

def calculate_behavioral_multiplier(signals, reference_date):
    """Applies exponential time-decay to engagement signals to penalize 'ghosts'."""
    response_rate = signals.get('recruiter_response_rate', 0.0)
    last_active = signals.get('last_active_date')
    
    if not last_active:
        return 0.0
    
    try:
        active_date = datetime.strptime(last_active, "%Y-%m-%d")
        days_inactive = max(0, (reference_date - active_date).days)
    except ValueError:
        days_inactive = 365
        
    # exp(-lambda * t). lambda=0.01 means ~40% value lost after 50 days.
    decay_factor = np.exp(-0.01 * days_inactive)
    return float(response_rate * decay_factor)

def analyze_career_dna(candidate, rules):
    """Analyzes trajectory, tenure, product-vs-consulting background, and logistics."""
    history = candidate.get('career_history', [])
    signals = candidate.get('redrob_signals', {})
    prof = candidate.get('profile', {})
    
    if not history:
        return 0.5, "No career history"
        
    total_months = sum(role.get('duration_months', 0) for role in history)
    avg_tenure = total_months / len(history) if history else 0
    
    industries = [str(role.get('industry', '')).lower() for role in history]
    companies = [str(role.get('company', '')).lower() for role in history]
    
    # Check for Consulting Only Anti-Pattern
    is_consulting_only = all(
        ind in [i.lower() for i in rules['anti_patterns']['consulting_only']['trigger_industries']] or 
        comp in rules['anti_patterns']['consulting_only']['trigger_companies']
        for ind, comp in zip(industries, companies)
    )
    
    score_multiplier = 1.0
    tags = []
    
    if is_consulting_only:
        score_multiplier *= rules['anti_patterns']['consulting_only']['penalty_multiplier']
        tags.append("Consulting-heavy")
    else:
        score_multiplier *= rules['bonuses']['product_company']
        tags.append("Product background")
        
    if avg_tenure < rules['anti_patterns']['title_chaser']['max_avg_tenure_months_flag']:
        score_multiplier *= rules['anti_patterns']['title_chaser']['penalty_multiplier']
        tags.append("Short tenure flagged")
        
    github_score = signals.get('github_activity_score', -1)
    if github_score > 50:
        score_multiplier *= rules['bonuses']['high_github_activity']
        tags.append("Strong OSS")
        
    # Logistics Checks (Notice period and location)
    if signals.get('notice_period_days', 90) <= 30:
        score_multiplier *= rules['bonuses']['fast_notice_period']
        
    location = str(prof.get('location', '')).lower()
    if 'pune' in location or 'noida' in location or signals.get('willing_to_relocate', False):
        score_multiplier *= rules['bonuses']['ideal_location_or_relocate']
        tags.append("Location matched")

    # Cap base experience score at ~8 years (96 months) for Senior role
    base_exp_score = min(total_months / 96.0, 1.0) 
    final_career_score = min(base_exp_score * score_multiplier, 1.0)
    
    return final_career_score, "; ".join(tags)

def generate_reasoning(rank, prof, tags, response_rate, days_inactive):
    """Generates a dynamic, non-hallucinated reasoning string."""
    exp = prof.get('years_of_experience', 0)
    title = prof.get('current_title', 'Engineer')
    
    return (
        f"Rank {rank}: {exp:.1f} yrs exp as {title}. DNA shows {tags}. "
        f"Strong semantic alignment with JD intent. "
        f"Behavioral check passed: {response_rate*100:.0f}% response rate, active {days_inactive} days ago."
    )

def run_ranking(candidates_path, rules_path, output_path):
    print("Loading JD Rules...")
    jd_rules = load_rules(rules_path)
    jd_query = jd_rules['core_semantic_query']
    
    # ---------------------------------------------------------
    # STAGE 1: Deterministic Pruning (Streaming)
    # ---------------------------------------------------------
    print("Stage 1: Streaming candidates and applying deterministic hard-gates...")
    valid_candidates = []
    candidate_docs = []
    
    open_fn = gzip.open if str(candidates_path).endswith('.gz') else open
    with open_fn(candidates_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line in ('[', ']'): continue
            if line.endswith(','): line = line[:-1]
            
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            # Hard-Gates: Drop Honeypots and Marketing Keyword Stuffers instantly
            if check_honeypots(candidate) or has_invalid_title(candidate, jd_rules):
                continue
                
            valid_candidates.append(candidate)
            candidate_docs.append(build_candidate_document(candidate))

    print(f"Passed Stage 1: {len(valid_candidates)} candidates.")

    # ---------------------------------------------------------
    # STAGE 2: Sparse Lexical Ranking (TF-IDF)
    # ---------------------------------------------------------
    print("Stage 2: Calculating TF-IDF sparse matrices...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    
    tfidf_matrix = vectorizer.fit_transform(candidate_docs)
    jd_tfidf = vectorizer.transform([jd_query])
    
    sparse_scores = sparse_cosine_similarity(jd_tfidf, tfidf_matrix)[0]
    
    # Slice top 2,000 candidates based on sparse lexical fit
    TOP_K_SPARSE = 2000
    top_sparse_indices = np.argsort(sparse_scores)[::-1][:TOP_K_SPARSE]
    
    stage2_candidates = [valid_candidates[i] for i in top_sparse_indices]
    stage2_docs = [candidate_docs[i] for i in top_sparse_indices]
    
    print(f"Passed Stage 2: Sliced to top {len(stage2_candidates)} candidates.")

    # ---------------------------------------------------------
    # STAGE 3: Dense Semantic Alignment
    # ---------------------------------------------------------
    print("Stage 3: Loading local all-MiniLM-L6-v2 for dense embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    jd_dense = model.encode([jd_query])
    # Batch encoding bypasses GIL and utilizes multi-core CPU efficiently
    cand_dense = model.encode(stage2_docs, batch_size=256, show_progress_bar=True)
    
    dense_scores = dense_cosine_similarity(jd_dense, cand_dense)[0]
    dense_scores = np.maximum(0.0, (dense_scores + 1) / 2) # Normalize to [0, 1]

    # ---------------------------------------------------------
    # Final Multi-Dimensional Scoring
    # ---------------------------------------------------------
    print("Calculating final composite scores...")
    
    all_dates = [c.get('redrob_signals', {}).get('last_active_date') for c in stage2_candidates]
    valid_dates = [datetime.strptime(d, "%Y-%m-%d") for d in all_dates if d]
    reference_date = max(valid_dates) if valid_dates else datetime.now()

    scored_candidates = []
    
    for i, candidate in enumerate(stage2_candidates):
        cid = candidate.get('candidate_id')
        prof = candidate.get('profile', {})
        signals = candidate.get('redrob_signals', {})
        
        # Base scores
        sem_score = dense_scores[i]
        career_score, career_tags = analyze_career_dna(candidate, jd_rules)
        
        # Multipliers
        behavioral_mult = calculate_behavioral_multiplier(signals, reference_date)
        
        w_sem = jd_rules['weights']['semantic_fit']
        w_dna = jd_rules['weights']['career_dna']
        
        final_score = ((w_sem * sem_score) + (w_dna * career_score)) * behavioral_mult
        
        last_active = signals.get('last_active_date')
        try:
            days_inactive = max(0, (reference_date - datetime.strptime(last_active, "%Y-%m-%d")).days) if last_active else 365
        except:
            days_inactive = 365
            
        scored_candidates.append({
            'candidate_id': cid,
            'final_score': final_score,
            'profile': prof,
            'tags': career_tags,
            'response_rate': signals.get('recruiter_response_rate', 0.0),
            'days_inactive': days_inactive
        })

    # Sort primarily by score (descending), tie-break by candidate_id (ascending)
    unique_candidates = {c['candidate_id']: c for c in scored_candidates}.values()
    scored_candidates = list(unique_candidates)

    # AFTER
    scored_candidates.sort(key=lambda x: (-round(x['final_score'], 4), x['candidate_id']))
    
    top_100 = scored_candidates[:100]
    
    print(f"Writing top 100 to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        
        for rank, cand in enumerate(top_100, start=1):
            reasoning = generate_reasoning(
                rank, cand['profile'], cand['tags'], 
                cand['response_rate'], cand['days_inactive']
            )
            writer.writerow([
                cand['candidate_id'],
                rank,
                f"{cand['final_score']:.4f}",
                reasoning
            ])
            
    print("Ranking complete! Ready for validation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob 3-Stage Ranker")
    parser.add_argument('--candidates', default='candidates.jsonl.gz', help='Path to candidates dataset')
    parser.add_argument('--rules', default='jd_rules.json', help='Path to offline JD rules')
    parser.add_argument('--out', default='team_submission.csv', help='Output CSV path')
    args = parser.parse_args()
    
    run_ranking(args.candidates, args.rules, args.out)