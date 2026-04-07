.PHONY: data build run demo

# Récupère, nettoie les données et construit l'index vectoriel
data:
	python scripts/fetch_events.py
	python scripts/clean_events.py
	python scripts/build_index.py

# Build l'image Docker
build:
	docker build -t puls-events-rag .

# Lance le conteneur (nécessite data/ et index/ générés)
run:
	docker run -p 8000:8000 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/index:/app/index \
		--env-file .env \
		puls-events-rag

# Flow complet : données + build + run
demo: data build run
