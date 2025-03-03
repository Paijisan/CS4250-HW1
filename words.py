from bs4 import BeautifulSoup
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

LANGUAGE_MAPPING = {
    "en": "english",
    "zh-cn": "chinese",
    "tl": "tagalog",
    "ko": "korean",
    "ja": "japanese"
}

SYMBOLS = {"|", "-", ",", ".", "+", ":", "?", "!", '"', "'", "(", ")", "[", "]", "...", "$"}

def tokenize_document(filename: str, lang: str) -> list[str]:
    # Read file and parse with BeautifulSoup
    with open(filename, "r") as f:
        contents = f.read()
        soup = BeautifulSoup(contents, "html.parser")

    # Remove unnecessary elements
    for element in soup.find_all(["script", "style", ""]):
        element.decompose()

    # TODO: support other languages besides English
    text = soup.get_text()
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text, language="english")
    filtered_tokens = list(filter(lambda x: x.lower() not in stop_words and x not in SYMBOLS, tokens))
    return filtered_tokens
    
    
def stem_tokens(tokens: list[str]) -> list[str]:
    # stuff here
    return tokens


def nltk_download():
    print("Checking for tokenizers/punkt data...")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("tokenizers/punkt")
    
    print("Checking for corpora/stopwords data...")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("corpora/stopwords")


if __name__ == "__main__":
    try:
        report = open("report.csv", "r")
        report_csv = csv.reader(report)
    except Exception as e:
        print(f"Could not open report.csv: {e}")
        exit(1)
    nltk_download()

    report_header = next(report_csv)
    sites = {}
    for url, outlinks, lang, filename in report_csv:
        # Tokenize and stem document to get words
        print(f"Tokenizing {filename} ({lang}): ", end="")
        tokens = tokenize_document(filename, lang)
        stems = stem_tokens(tokens)

        # Write words to file
        words_filename = filename.replace(".html", "-words.txt")
        with open(words_filename, "w") as f:
            f.write("\n".join(stems))
            print(len(stems))

        # Add words to accumulator
        site = "/".join(filename.split("/")[:-1])
        if site not in sites:
            sites[site] = []
        sites[site] += stems
    
    # Write words from each site to file
    for site, tokens in sites.items():
        print(f"{site} word count: {len(tokens)}")
        words_filename = f"{site}/words.txt"
        with open(words_filename, "w") as f:
            f.write("\n".join(tokens))
    
    print("Done!")

