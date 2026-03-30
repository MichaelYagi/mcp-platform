---
name: ml_recommendations
description: >
  ML-powered movie and TV show recommendations based on your Plex viewing history.
  Automatically imports your watch history, trains a machine learning model on your
  preferences, and provides personalized recommendations. Use this skill when the
  user wants personalized content suggestions, viewing history analysis, or to set
  up automated recommendations.
tags:
  - plex
  - recommendations
  - ml
  - machine-learning
  - personalization
  - viewing-history
tools:
  - auto_train_from_plex
  - import_plex_history
  - train_recommender
  - record_viewing
  - recommend_content
  - get_recommender_stats
  - reset_recommender
  - auto_recommend_from_plex
---

# ML Recommendations Skill

## Tool Routing

| User asks... | Tool |
|---|---|
| "Set up / auto-train recommendations" | `auto_train_from_plex` |
| "Import my Plex watch history" | `import_plex_history` |
| "Train the model" | `train_recommender` |
| "Record that I watched X" | `record_viewing` |
| "Which of these should I watch?" | `recommend_content` |
| "Show recommender stats / is it trained?" | `get_recommender_stats` |
| "What should I watch tonight from Plex?" | `auto_recommend_from_plex` |
| "Reset / start fresh" | `reset_recommender` |

## Key Workflows

**One-click setup (preferred):**
```
auto_train_from_plex(50)   # imports history + trains in one step
```

**Manual setup:**
```
import_plex_history(50)    # import first
train_recommender()         # then train (needs 20+ events)
```

**Rank specific options:**
```
recommend_content([
    {"title": "Dune", "genre": "SciFi", "year": 2021, "rating": 8.0, "runtime": 155},
    {"title": "Knives Out", "genre": "Mystery", "year": 2019, "rating": 7.9, "runtime": 130}
])
```

**Auto-recommend from unwatched Plex content:**
```
auto_recommend_from_plex(limit=20, genre_filter="SciFi", min_rating=7.0)
```

## Common Patterns

- User asks "what should I watch?" → check `get_recommender_stats` first;
  if not trained → `auto_train_from_plex`; if trained → `auto_recommend_from_plex`
- User mentions watching something → `record_viewing(..., finished=True/False)`
- User provides a list of options → `recommend_content([...])`

## Notes

- Minimum 20 viewing events required to train; 50+ recommended
- `auto_train_from_plex` skips music content automatically
- `record_viewing` requires: title, genre, year, rating, runtime, finished
- Model persists between sessions