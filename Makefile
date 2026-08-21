########################################################################
# Karma Edge - convenience targets. Assumes an activated virtualenv:
#   python3 -m venv .venv && source .venv/bin/activate
########################################################################
.PHONY: help setup seed reseed test demo chat ui ledger metrics clean

help:
	@echo "make setup    - install requirements"
	@echo "make seed     - build the demo warehouse (18 months of data)"
	@echo "make reseed   - delete and rebuild the databases"
	@echo "make test     - run the offline test suite (no API key needed)"
	@echo "make demo     - one scripted question, offline"
	@echo "make chat     - interactive CLI chatbot"
	@echo "make ui       - Streamlit chat UI"
	@echo "make ledger   - dump the accountability ledger"
	@echo "make metrics  - list the semantic metric layer"

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Ready. MODEL_PROVIDER=fake works with zero API keys."

seed:
	python -m data.seed

reseed:
	rm -f karma_edge.db karma_ledger.db
	python -m data.seed

test:
	python -m pytest -q

demo:
	MODEL_PROVIDER=fake python -m app.main ask "Which category is losing the most gross margin and who owns it?"

chat:
	python -m app.main

ui:
	streamlit run app/ui.py

ledger:
	python -m app.main ledger

metrics:
	python -m app.main metrics

clean:
	rm -rf **/__pycache__ __pycache__ .pytest_cache
	rm -f karma_edge.db karma_ledger.db
