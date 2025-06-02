from os.path import dirname, realpath, join


ROOT_DIR = dirname(realpath(__file__))
DATASET_DIR = join(ROOT_DIR, "resources")

SUMMARIZE_PROMPT = 'Summarize clinical text'