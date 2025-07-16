import fitz  # PyMuPDF
import re
import spacy
from transformers import pipeline
from keybert import KeyBERT

# --- Download spaCy model if it doesn't exist ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# --- Load all models once ---
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
kw_model = KeyBERT()



def extract_text_from_pdf(file):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_articles(text):
    """Splits the text into articles based on a pattern."""
    raw_articles = re.split(r'\n(?=[A-Z][A-Z\s]{5,100})\n', text)
    return [a.strip() for a in raw_articles if len(a.strip()) > 300]


def score_article(article):
    """Scores an article based on the number of named entities and length."""
    doc = nlp(article)
    named_entities = len(doc.ents)
    length = len(article.split())
    return named_entities * 2 + length / 40


def summarize_article(article):
    """Summarizes an article using the transformers pipeline."""
    try:
        summary = summarizer(article[:1024], max_length=60, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return article[:200] + "..."


def extract_keywords(article):
    """Extracts keywords from an article using KeyBERT."""
    try:
        keywords_with_scores = kw_model.extract_keywords(article[:500], keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        return [kw[0] for kw in keywords_with_scores]
    except Exception as e:
        print(f"Error during keyword extraction: {e}")
        return []


def process_pdf(file):
    """Processes a PDF file and returns a list of processed articles."""
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


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Replace 'your_pdf_file.pdf' with the actual path to your PDF file
    pdf_file = "your_pdf_file.pdf"  
    results = process_pdf(pdf_file)

    for result in results:
        print(f"Headline: {result['headline']}")
        print(f"Summary: {result['summary']}")
        print(f"Keywords: {result['keywords']}")
        print(f"Score: {result['score']}")
        print("-" * 20)
