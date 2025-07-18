from os.path import dirname, realpath, join


ROOT_DIR = dirname(realpath(__file__))
DATASET_DIR = join(ROOT_DIR, "resources")

SUMMARIZE_PROMPT = 'Summarize clinical text'

SUMMARIZE_PROMPT_LOCALE = {
    "en": SUMMARIZE_PROMPT,
    "pt": "Resumir texto clínico",
    "es": "Resumir el texto clínico",
    "fr": "Résumer le texte clinique"
}


