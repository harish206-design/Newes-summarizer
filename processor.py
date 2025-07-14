import fitz  # PyMuPDF
import re
import spacy
import spacy.cli
from transformers import pipeline
from keybert import KeyBERT

# Download the spaCy model first
spacy.cli.download("en_core_web_sm")

# --- Load all models once ---
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
kw_model = KeyBERT()

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_articles(text):
    raw_articles = re.split(r'\n(?=[A-Z][A-Z\s]{5,100})\n', text)
    return [a.strip() for a in raw_articles if len(a.strip()) > 300]

def score_article(article):
    doc = nlp(article)
    named_entities = len(doc.ents)
    length = len(article.split())
    return named_entities * 2 + length / 40

def summarize_article(article):
    try:
        summary = summarizer(article[:1024], max_length=60, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        return article[:200] + "..."

def extract_keywords(article):
    try:
        keywords_with_scores = kw_model.extract_keywords(article[:500], keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        return [kw[0] for kw in keywords_with_scores]
    except Exception:
        return []

def process_pdf(file):
    text = extract_text_from_pdf(file)
    articles = split_articles(text)

    processed = []
    for article in articles:
        score = score_article(article)
        summary = summarize_article(article)
        keywords = extract_keywords(article)
        headline = article.split('\n')[0][:100]

        processed.append({
            "headline": headline,
            "summary": summary,
            "keywords": keywords,
            "score": score
        })

    # Sort by score and return the top 5
    processed.sort(key=lambda x: x['score'], reverse=True)
    return processed[:5]