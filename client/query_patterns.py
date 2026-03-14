"""
Query Patterns — Single Routing Authority
==========================================
This is the ONE place where query intent is decided.

All other modules (langgraph.py, distributed_skills_manager.py, client.py)
import QueryIntent and call classify() — they do NO independent pattern matching.

Usage:
    from client.query_patterns import classify, QueryIntent

    intent = classify("what's the weather this week?")
    intent.category          # "weather"
    intent.tools             # ["get_location_tool", "get_weather_tool"]
    intent.needs_web_search  # False
    intent.needs_skills      # False
    intent.is_conversational # False

Adding a new category:
    Add one entry to INTENT_CATALOG. That's it.
    The rest of the system picks it up automatically.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# INTENT CATALOG
# Single definition of every category, its trigger pattern,
# which tools it needs, and its priority (lower = checked first).
#
# Rules:
#   - pattern:         regex that identifies this intent
#   - tools:           tool names to bind (supports "prefix*" wildcards)
#   - priority:        1 = highest (checked first), 3 = lowest fallback
#   - web_search:      True if web search should augment the response
#   - skills:          True if skill injection should be considered
#   - exclude_pattern: optional regex that vetoes a match
# ═══════════════════════════════════════════════════════════════════

INTENT_CATALOG = [
    {
        "name": "github_review",
        "pattern": (
            r'github\.com/'
            r'|\breview\b.*github\.com|github\.com.*\breview\b'
            r'|\b(clone|analyze|audit)\b.*github\.com|github\.com.*(clone|analyze|audit)\b'
            r'|\breview.*(repo|repository)'
            r'|\banalyze.*(repo|repository)'
            r'|\bcheck.*(repo|repository)'
            r'|\breview.*https?://'
            r'|\banalyze.*https?://'
            r'|\bgithub\s+(repo|repository|project)'
            r'|\bfrom\s+github'
        ),
        "tools": [
            "github_clone_repo", "github_list_files", "github_get_file_content",
            "github_cleanup_repo", "analyze_project", "analyze_code_file",
            "review_code", "scan_project_structure"
        ],
        "examples": {
            "github_clone_repo": "use github_clone_repo to review ",
            "github_list_files": "use github_list_files to list files in ",
            "github_get_file_content": "use github_get_file_content to read file ",
            "github_cleanup_repo": "use github_cleanup_repo to clean up the repo",
            "analyze_project": "use analyze_project to analyze project at ",
            "analyze_code_file": "use analyze_code_file to analyze ",
            "review_code": "use review_code to review ",
            "scan_project_structure": "use scan_project_structure to scan "
        },
        "priority": 2,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "file_analyst",
        "pattern": (
            r'\b(/mnt/[a-zA-Z]/|/home/|/tmp/|/var/|~/)'
            r'|[A-Za-z]:[/\\]'
            r'|\b(open|read|look\s+at|load|analys[ei]|analyze|review|inspect|check|examine|show)\b.*\.(csv|tsv|txt|md|json|yaml|yml|toml|xml|log|py|js|ts|ini|cfg|conf|sh|bash)\b'
            r'|\S+\.(csv|tsv|txt|md|json|yaml|yml|toml|xml|log|ini|cfg|conf)\b'
            r'|\b(budget|spreadsheet|expenses?|transactions?|ledger|invoice)\b'
            r'|\b(insights?\s+into|analys[ei]\s+my|analyze\s+my|summarize\s+my)\b.*\bfile\b'
        ),
        "tools": ["read_file_tool_handler"],
        "examples": {
            "read_file_tool_handler": "use read_file_tool_handler to analyze "
        },
        "priority": 2,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "code_assistant",
        "pattern": (
            r'\btech\s+stack\b|\btechnology\s+stack\b|\bwhat.*tech\b|\bwhat.*stack\b'
            r'|\bwhat.*technologies\b|\bwhat.*languages\b|\bwhat.*frameworks?\b'
            r'|\bwhat.*dependencies\b|\bproject\s+structure\b|\banalyze.*project\b'
            r'|\bscan.*project\b|\bshow.*structure\b|\blist.*dependencies\b'
            r'|\bnode\.?js\s+(packages?|dependencies|modules)\b|\bnpm\s+(packages?|dependencies|modules)\b'
            r'|\bpackage\.json\b|\bnode\s+(packages?|dependencies|modules)\b'
            r'|\b(about|more|explain).*node\.?js\s+(packages?|dependencies)\b'
            r'|\btell.*about.*(node|packages?|dependencies)\b'
            r'|\bmore.*about.*(node|packages?|dependencies)\b'
            r'|\bgo\s+into\s+(depth|detail)\b'
            r'|\bmore\s+detail.*about.*(packages?|dependencies|modules)\b'
            r'|\bin[\s-]?depth.*about.*(packages?|dependencies|modules)\b'
            r'|\belaborate.*on.*(packages?|dependencies|modules)\b'
            r'|\bexpand.*on.*(packages?|dependencies|modules)\b'
            r'|\bwhat.*(do|are).*(packages?|dependencies|modules)\b'
            r'|\bexplain.*(packages?|dependencies|modules)\b'
            r'|\bwhat.*do\s+they\s+do\b|\bwhat.*are\s+(they|those)\s+(for|used\s+for)\b'
            r'|\bwhat.*they.*used\s+for\b'
            r'|\banalyze.*code\b|\bcheck.*code\b|\blint\b'
            r'|\banalyze.*\.(py|js|jsx|ts|tsx|rs|go|java|kt)\b'
            r'|\bcheck.*\.(py|js|jsx|ts|tsx|rs|go|java|kt)\b'
            r'|\bfix.*bug\b|\bfix.*error\b|\bfix.*issue\b|\bfix.*code\b'
            r'|\bgenerate.*code\b|\bcreate.*(function|class|module|component)\b'
            r'|\bwrite.*(function|class)\b'
        ),
        "tools": [
            "analyze_project", "get_project_dependencies", "scan_project_structure",
            "analyze_code_file", "fix_code_file", "suggest_improvements",
            "explain_code", "generate_tests", "refactor_code", "generate_code"
        ],
        "examples": {
            "analyze_project": "use analyze_project to analyze project at ",
            "get_project_dependencies": "use get_project_dependencies to list dependencies for ",
            "scan_project_structure": "use scan_project_structure to scan ",
            "analyze_code_file": "use analyze_code_file to analyze ",
            "fix_code_file": "use fix_code_file to fix the bug in ",
            "suggest_improvements": "use suggest_improvements to improve ",
            "explain_code": "use explain_code to explain ",
            "generate_tests": "use generate_tests to generate tests for ",
            "refactor_code": "use refactor_code to refactor ",
            "generate_code": "use generate_code to write a function that "
        },
        "priority": 2,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "analyze_image",
        "pattern": (
            r'\banalyze\b.*https?://'
            r'|\bdescribe\b.*https?://'
            r'|\bwhat.*https?://.*\b(image|photo|picture|pic)\b'
            r'|\b(image|photo|picture|pic)\b.*https?://'
            r'|\banalyze\s+(this\s+)?(image|photo|picture)\b'
            r'|\blook\s+at\s+(this\s+)?(image|photo|picture)\b'
            r'|\banalyze\b.*\.(jpg|jpeg|png|gif|webp|bmp|heic)\b'
            r'|\bdescribe\b.*\.(jpg|jpeg|png|gif|webp|bmp|heic)\b'
            r'|\b(image|photo|picture)\b.*\.(jpg|jpeg|png|gif|webp|bmp|heic)\b'
            r'|\bdescribe\b.*(image|photo|picture)\b.*(/mnt/|/home/|/tmp/|https?://)'
            r'|\banalyze\b.*(image|photo|picture)\b.*(/mnt/|/home/|/tmp/|https?://)'
        ),
        "tools": ["analyze_image_tool"],
        "examples": {
            "analyze_image_tool": "use analyze_image_tool to describe this image: "
        },
        "priority": 1,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "shashin_random",
        "pattern": (
            r'\brandom\b.*(photo|image|picture|pic)\b'
            r'|\b(photo|image|picture|pic)\b.*\brandom\b'
            r'|\bsurprise\s+me\b'
            r'|\bshow\s+me\s+a\s+random\b'
            r'|\brandom\s+shashin\b|\bshashin.*random\b'
        ),
        "tools": ["shashin_random_tool"],
        "examples": {
            "shashin_random_tool": "use shashin_random_tool to show me a random photo"
        },
        "priority": 1,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "web_image_search",
        "pattern": (
            r'\bshow\s+me\s+a\s+(picture|photo|image)\s+of\b'
            r'|\bwhat\s+does\b.+\blook\s+like\b'
            r'|\b(picture|photo|image)\s+of\b.+\b(actor|person|place|building|animal)\b'
            r'|\bweb\s+(image|photo|picture)\s+(of|search)\b'
            r'|\bfind\s+(me\s+)?(a\s+)?(picture|photo|image)\s+of\b'
        ),
        "exclude_pattern": r'\bmy\b|\bshashin\b|\bphoto\s+of\s+[a-f0-9\-]{36}\b',
        "tools": ["web_image_search_tool"],
        "examples": {
            "web_image_search_tool": "use web_image_search_tool to show me a picture of "
        },
        "priority": 1,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "shashin_analyze",
        "pattern": (
            r'\banalyze\b.*(photo|image|picture|pic)\b'
            r'|\b(photo|image|picture|pic)\b.*\banalyze\b'
            r'|\bdescribe\b.*(photo|image|picture|pic)\b'
            r'|\b(photo|image|picture|pic)\b.*\bdescribe\b'
            r'|\bwhat.*(photo|image|picture|pic).*(show|in|of)\b'
            r'|\bvision\b.*(photo|image|picture|pic)\b'
            r'|\bshashin.*analyz\b|\banalyz.*shashin\b'
        ),
        "exclude_pattern": r'(/mnt/|/home/|/tmp/|https?://|\.(jpg|jpeg|png|gif|webp|bmp|heic)\b)',
        "tools": ["shashin_search_tool", "shashin_analyze_tool"],
        "examples": {
            "shashin_search_tool": "use shashin_search_tool to find photos of ",
            "shashin_analyze_tool": "use shashin_analyze_tool to describe the photo of "
        },
        "priority": 2,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "shashin_search",
        "pattern": (
            r'\b(find|search|show|look\s+for)\b.*(photo|image|picture|pic)\b'
            r'|\bmy\s+(photos?|images?|pictures?|gallery)\b'
            r'|\bshashin\b'
            r'|\bphotos?\s+(of|with|by|from|taken|featuring)\b'
            r'|\b(browse|list)\b.*(photos?|images?|gallery)\b'
        ),
        "tools": ["shashin_search_tool"],
        "examples": {
            "shashin_search_tool": "use shashin_search_tool to find photos of "
        },
        "priority": 2,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "plex_search",
        "pattern": (
            r'\b(find|search|look\s+for|show\s+me)\s+.*\b(movie|film|show|media|series)\b'
            r'|\bmovies?\s+(about|where|with|featuring|in\s+which)\b'
            r'|\bfilms?\s+(about|where|with|featuring|in\s+which)\b'
            r'|\bwhere\s+.*\s+(wins?|loses?|dies|survives|happens|occurs|escapes)\b'
            r'|\bsearch\s+(plex|library|my\s+library|my\s+movies)\b'
            r'|\bfind\s+.*\s+in\s+(plex|library|my\s+library)\b'
            r'|\bscene\s+(where|with|from)\b|\bfind\s+scene\b|\blocate\s+scene\b'
            r'|\bbrowse\s+my\b|\blist\s+.*\s+(movies|films|shows)\b'
        ),
        "tools": [
            "rag_search_tool", "semantic_media_search_text",
            "scene_locator_tool", "find_scene_by_title"
        ],
        "examples": {
            "semantic_media_search_text": "use semantic_media_search_text to find movies about ",
            "scene_locator_tool": "use scene_locator_tool to find the scene where ",
            "find_scene_by_title": "use find_scene_by_title to find a scene in "
        },
        "priority": 2,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "rag",
        "pattern": (
            r'\bhow\s+many\s+.*(ingested|in\s+rag)\b'
            r'|\bwhat\s+(has|was)\s+been\s+ingested\b'
            r'|\bitems?\s+(have\s+been|were)\s+ingested\b'
            r'|\bcount\s+.*(items?|in\s+rag)\b|\btotal\s+.*(items?|in\s+rag)\b'
            r'|\b(show|list|display)\s+rag\b'
            r'|\brag\s+(status|contents?|info|summary|overview|report|stats)\b'
            r'|\bwhat.s\s+in\s+(the\s+|my\s+)?rag\b'
            r'|\bgive\s+me\s+rag\s+(stats|status|info|details)\b'
            r'|\bsearch\s+(the\s+)?rag\b|\bfind\s+in\s+rag\b'
            r'|\blook\s+up\s+in\s+rag\b|\brag\s+search\b|\bquery\s+(the\s+)?rag\b'
            r'|\bdo\s+you\s+have\s+.*\s+in\s+rag\b'
            r'|\bbrowse\s+(the\s+)?rag\b'
            r'|\bshow\s+rag\s+(content|documents|entries|sources)\b'
            r'|\blist\s+rag\s+(sources|documents|content)\b'
            r'|\bwhat\s+sources\s+.*(in\s+)?rag\b'
            r'|\brag\s+(database|storage|data)\b'
            r'|\bwhat\s+do\s+you\s+know\s+about\b'
            r'|\bwhat\s+is\s+.+\s+in\s+my\s+(rag|knowledge|database)\b'
        ),
        "tools": [
            "rag_search_tool", "rag_status_tool", "rag_list_sources_tool",
            "rag_browse_tool", "rag_diagnose_tool", "rag_add_tool"
        ],
        "examples": {
            "rag_search_tool": "use rag_search_tool to search for ",
            "rag_status_tool": "use rag_status_tool to show RAG stats",
            "rag_list_sources_tool": "use rag_list_sources_tool to list RAG sources",
            "rag_browse_tool": "use rag_browse_tool to browse RAG content",
            "rag_diagnose_tool": "use rag_diagnose_tool to diagnose the RAG database",
            "rag_add_tool": "use rag_add_tool to add this to RAG: "
        },
        "priority": 2,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "trilium",
        "pattern": (
            r'\btrilium\b|\bnotes?\s+(in\s+)?trilium\b|\bmy\s+notes?\b'
            r'|\bsearch\s+(my\s+)?notes?\b|\bfind\s+(in\s+)?(my\s+)?notes?\b'
            r'|\blook\s+up\s+(in\s+)?notes?\b'
            r'|\bcreate\s+(a\s+)?note\b|\badd\s+(a\s+)?note\b'
            r'|\bupdate\s+(my\s+)?note\b|\bdelete\s+(my\s+)?note\b'
            r'|\bnotes?\s+tagged\b|\bnotes?\s+with\s+label\b'
            r'|\badd\s+(label|tag)\s+to\s+note\b'
            r'|\brecent\s+notes?\b|\blatest\s+notes?\b|\bchild\s+notes?\b'
        ),
        "tools": [
            "search_notes", "search_by_label", "get_note_by_id",
            "create_note", "update_note_content", "update_note_title",
            "delete_note", "add_label_to_note", "get_note_labels",
            "get_note_children", "get_recent_notes"
        ],
        "examples": {
            "search_notes": "use search_notes to search my notes for ",
            "search_by_label": "use search_by_label to find notes tagged ",
            "get_note_by_id": "use get_note_by_id to get note ",
            "create_note": "use create_note to create a note titled ",
            "update_note_content": "use update_note_content to update note ",
            "update_note_title": "use update_note_title to rename note ",
            "delete_note": "use delete_note to delete note ",
            "add_label_to_note": "use add_label_to_note to tag note ",
            "get_note_labels": "use get_note_labels to list labels for note ",
            "get_note_children": "use get_note_children to list children of note ",
            "get_recent_notes": "use get_recent_notes to show my recent notes"
        },
        "priority": 2,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "weather",
        "pattern": (
            r'\bweather\b|\btemperature\b|\bforecast\b'
            r'|\brain\b|\bsnow\b|\bwind\b|\bconditions\b'
        ),
        "tools": ["get_location_tool", "get_weather_tool"],
        "examples": {
            "get_weather_tool": "use get_weather_tool to get the weather in "
        },
        "priority": 3,
        "web_search": False,   # MCP tools handle this — never web search
        "skills": False,
    },
    {
        "name": "location",
        "pattern": (
            r'\b(my|what\'?s?\s+my)\s+location\b'
            r'|\bwhere\s+am\s+i\b|\bcurrent\s+location\b|\bwhere\s+do\s+i\s+live\b'
        ),
        "tools": ["get_location_tool"],
        "examples": {
            "get_location_tool": "use get_location_tool to show my current location"
        },
        "priority": 3,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "time",
        "pattern": (
            r'\bwhat\s+time\b|\bwhat\s+date\b|\bcurrent\s+time\b'
            r'|\bcurrent\s+date\b|\btime\s+now\b|\btime\s+is\s+it\b'
        ),
        "tools": ["get_time_tool"],
        "examples": {
            "get_time_tool": "use get_time_tool to show the current time"
        },
        "priority": 3,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "system",
        "pattern": (
            r'\bsystem\s+info\b|\bhardware\b|\b(cpu|gpu|ram)\b'
            r'|\bspecs?\b|\bprocesses?\b|\bperformance\b'
            r'|\butilization\b|\bmemory\s+usage\b'
        ),
        "tools": [
            "get_hardware_specs_tool", "get_system_info",
            "list_system_processes", "terminate_process"
        ],
        "examples": {
            "get_hardware_specs_tool": "use get_hardware_specs_tool to show my hardware specs",
            "get_system_info": "use get_system_info to show system info",
            "list_system_processes": "use list_system_processes to list running processes",
            "terminate_process": "use terminate_process to terminate process "
        },
        "priority": 3,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "ml_recommendation",
        "pattern": (
            r'\brecommend(ation)?s?\b|\bsuggest(ion)?s?\b'
            r'|\bml\s+(model|train|recommendation)\b'
            r'|\btrain\s+(model|recommender|recommendation)\b|\bauto.?train\b'
            r'|\bimport\s+.*\s*history\b|\bviewing\s+history\b|\bwatch\s+history\b'
            r'|\brecord\s+(viewing|that\s+i\s+watched)\b'
            r'|\bwhat\s+should\s+i\s+watch\b|\brank\s+(these|movies|shows)\b'
            r'|\bmy\s+best\s+unwatched\b|\bunwatched\s+(recommendations|suggestions)\b'
            r'|\brecommender\s+stats\b'
        ),
        "tools": [
            "record_viewing", "train_recommender", "recommend_content",
            "get_recommender_stats", "import_plex_history", "auto_train_from_plex",
            "reset_recommender", "auto_recommend_from_plex"
        ],
        "examples": {
            "record_viewing": "use record_viewing to record that I watched ",
            "train_recommender": "use train_recommender to train the recommendation model",
            "recommend_content": "use recommend_content to recommend something to watch",
            "get_recommender_stats": "use get_recommender_stats to show recommender stats",
            "import_plex_history": "use import_plex_history to import my Plex viewing history",
            "auto_train_from_plex": "use auto_train_from_plex to train from my Plex history",
            "reset_recommender": "use reset_recommender to reset the recommendation model",
            "auto_recommend_from_plex": "use auto_recommend_from_plex to recommend based on my Plex history"
        },
        "priority": 3,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "code",
        "pattern": (
            r'\bcode\b|\bscan\s+code\b|\bdebug\b|\breview\s+code\b'
            r'|\bsummarize\s+code\b|\bfix\s+this\s+code\b|\bexplain\s+this\s+code\b'
        ),
        "tools": [
            "review_code", "summarize_code_file", "search_code_in_directory",
            "scan_code_directory", "summarize_code", "debug_fix"
        ],
        "examples": {
            "review_code": "use review_code to review ",
            "summarize_code_file": "use summarize_code_file to summarize ",
            "search_code_in_directory": "use search_code_in_directory to search code for ",
            "scan_code_directory": "use scan_code_directory to scan code in ",
            "summarize_code": "use summarize_code to summarize this code: ",
            "debug_fix": "use debug_fix to fix this error: "
        },
        "priority": 3,
        "web_search": False,
        "skills": True,
    },
    {
        "name": "text",
        "pattern": r'\b(summarize|summary|explain|simplify|break\s+down)\b',
        "exclude_pattern": r'\bcode\b',
        "tools": [
            "summarize_text_tool", "concept_contextualizer_tool",
            "summarize_direct_tool", "explain_simplified_tool"
        ],
        "examples": {
            "summarize_text_tool": "use summarize_text_tool to summarize: ",
            "concept_contextualizer_tool": "use concept_contextualizer_tool to explain ",
            "summarize_direct_tool": "use summarize_direct_tool to summarize: ",
            "explain_simplified_tool": "use explain_simplified_tool to simplify: "
        },
        "priority": 3,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "todo",
        "pattern": (
            r'\btodo\b|\btask\b|\bremind\s+me\b|\bmy\s+todos?\b'
            r'|\bmy\s+tasks?\b|\badd\s+to\s+my\s+list\b|\btask\s+list\b'
        ),
        "tools": [
            "add_todo_item", "list_todo_items", "search_todo_items",
            "update_todo_item", "delete_todo_item", "delete_all_todo_items"
        ],
        "examples": {
            "add_todo_item": "use add_todo_item to add task: ",
            "list_todo_items": "use list_todo_items to show my todos",
            "search_todo_items": "use search_todo_items to search todos for ",
            "update_todo_item": "use update_todo_item to mark done: ",
            "delete_todo_item": "use delete_todo_item to delete todo: ",
            "delete_all_todo_items": "use delete_all_todo_items to clear all my todos"
        },
        "priority": 3,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "knowledge",
        "pattern": (
            r'\bremember\b|\bsave\s+this\b|\bmake\s+a\s+note\b|\bknowledge\s+base\b'
            r'|\bsearch\s+my\s+notes?\b|\badd\s+entry\b|\bnote\s+this\b|\bstore\s+this\b'
        ),
        "tools": [
            "add_entry", "list_entries", "get_entry", "search_entries",
            "search_by_tag", "search_semantic", "update_entry", "delete_entry"
        ],
        "examples": {
            "add_entry": "use add_entry to remember: ",
            "list_entries": "use list_entries to show all knowledge entries",
            "get_entry": "use get_entry to retrieve entry ",
            "search_entries": "use search_entries to search my knowledge for ",
            "search_by_tag": "use search_by_tag to find entries tagged ",
            "search_semantic": "use search_semantic to semantically search for ",
            "update_entry": "use update_entry to update entry ",
            "delete_entry": "use delete_entry to delete entry "
        },
        "priority": 3,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "ingest",
        "pattern": (
            r'\bingest\s+(now|movies?|items?|\d+|batch)\b'
            r'|\bstart\s+ingesting\b|\badd\s+to\s+(rag|knowledge)\b'
            r'|\bprocess\s+subtitles?\b|\bextract\s+subtitles?\b'
        ),
        "tools": ["ingest_movies", "ingest_batch_tool"],
        "examples": {
            "ingest_movies": "use ingest_movies to ingest movies now",
            "ingest_batch_tool": "use ingest_batch_tool to ingest a batch"
        },
        "priority": 3,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "a2a",
        "pattern": (
            r'\ba2a\b|\bremote\s+(agents?|tools?)\b|\bdiscover\s+(agents?|tools?)\b'
            r'|\bsend\s+to\s+remote\b|\bcall\s+remote\s+tool\b'
            r'|\buse\s+remote\s+agent\b|\bconnect\s+to\s+agent\b'
        ),
        "tools": ["send_a2a*", "discover_a2a"],
        "examples": {
            "send_a2a*": "use send_a2a to send to remote agent: ",
            "discover_a2a": "use discover_a2a to discover remote agents"
        },
        "priority": 3,
        "web_search": False,
        "skills": False,
    },
    {
        "name": "current_events",
        "pattern": (
            r'\b(latest|breaking)\s+(news|story|stories|update)\b'
            r'|\bwhat\'?s\s+(happening|going\s+on)\b'
            r'|\bin\s+the\s+news\b'
        ),
        "tools": [],
        "examples": {},
        "priority": 3,
        "web_search": True,
        "skills": False,
    },
    {
        "name": "stock_price",
        "pattern": r'\b(stock|share)\s+price\b|\btrading\s+at\b|\bmarket\s+cap\b',
        "tools": [],
        "examples": {},
        "priority": 3,
        "web_search": True,
        "skills": False,
    },
]

# Pre-compile all patterns for performance
for _entry in INTENT_CATALOG:
    _entry["_compiled"] = re.compile(_entry["pattern"], re.IGNORECASE)
    if "exclude_pattern" in _entry:
        _entry["_compiled_exclude"] = re.compile(_entry["exclude_pattern"], re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════
# CONVERSATIONAL EXCLUSIONS
# Queries matching these patterns are always conversational —
# no tools, no web search, no skills.
# Checked before INTENT_CATALOG.
# ═══════════════════════════════════════════════════════════════════

_CONVERSATIONAL_PATTERNS = re.compile(
    r'^(my |i |i\'m |i am )'                           # personal statements
    r'|^(acknowledge|confirm|please note|note that)'   # memory instructions
    r'|^(yes|no|ok|okay|sure|thanks|thank you|hello|hi\b)'  # filler
    r'|\b(favourite|favorite|i like|i love|i hate)\b'  # preferences (i prefer removed — conflicts with knowledge)
    r'|^(create|write|generate|make|draft|compose)\b'  # creative tasks
    r'|\b(you just|i just|i told you|i said|i mentioned|i gave you)\b'  # recall
    r'|^(what did i|what were the|what was the|do you remember|can you recall)\b'
    # Pronoun follow-ups — context-dependent, never need tools
    # Requires he/she/they/him/her/them/his/her/their to avoid catching
    # broad questions like "what is the weather" or "what has the news been"
    r'|^(what (are|were|is|was) (his|her|their|its)\b)'
    r'|^(what (did|does|do|has|have|had) (he|she|they|it)\b)'
    r'|^(tell me (more about|about) (him|her|them|it)\b)'
    r'|^(how (did|does|do|has|have) (he|she|they|it)\b)'
    r'|^(why (did|does|do|is|was|were|has|have) (he|she|they|it)\b)'
    r'|^(when (did|does|do|has|have|had|is|was|were) (he|she|they|it)\b)',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════
# QueryIntent — the single result object consumed by all modules
# ═══════════════════════════════════════════════════════════════════

@dataclass
class QueryIntent:
    category: str                    # matched category name, or "conversational" / "general"
    tools: list = field(default_factory=list)
    needs_web_search: bool = False
    needs_skills: bool = False
    is_conversational: bool = False
    priority: int = 3


# ═══════════════════════════════════════════════════════════════════
# classify() — THE single entry point
# ═══════════════════════════════════════════════════════════════════

def classify(query: str, available_tool_names: list = None) -> QueryIntent:
    """
    Classify a query and return a QueryIntent.

    Args:
        query:                Natural language query from the user
        available_tool_names: Optional list of tool names currently available.
                              If provided, explicit tool-name detection is enabled.

    Returns:
        QueryIntent with all routing decisions pre-computed.
    """
    msg = query.strip()

    # ── Step 1: Conversational check — exits immediately ─────────
    if _CONVERSATIONAL_PATTERNS.search(msg):
        return QueryIntent(category="conversational", is_conversational=True)

    # ── Step 2: Explicit tool name in query ───────────────────────
    if available_tool_names:
        msg_lower = msg.lower()
        for tool_name in available_tool_names:
            if tool_name.lower() in msg_lower:
                return QueryIntent(
                    category="explicit_tool",
                    tools=[tool_name],
                    needs_web_search=False,
                    needs_skills=False,
                )

    # ── Step 3: Match against INTENT_CATALOG (priority order) ────
    sorted_catalog = sorted(INTENT_CATALOG, key=lambda x: x["priority"])

    for entry in sorted_catalog:
        if not entry["_compiled"].search(msg):
            continue
        if "_compiled_exclude" in entry and entry["_compiled_exclude"].search(msg):
            continue

        return QueryIntent(
            category=entry["name"],
            tools=entry["tools"],
            needs_web_search=entry["web_search"],
            needs_skills=entry["skills"],
            priority=entry["priority"],
        )

    # ── Step 4: No match — check if current info needed ──────────
    _WEB_SEARCH_PATTERN = re.compile(
        r'\b(current|latest|recent|today|right now|as of)\b'
        r'|\b(news|stock|price|score|update)\b'
        r'|\bwho\s+is\s+the\s+(current|new)\b'
        r'|\bwhat\'?s\s+the\s+(current|latest)\b',
        re.IGNORECASE
    )

    return QueryIntent(
        category="general",
        tools=[],
        needs_web_search=bool(_WEB_SEARCH_PATTERN.search(msg)),
        needs_skills=False,
    )


# ═══════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY SHIMS
# Keep existing imports working while langgraph.py and
# distributed_skills_manager.py migrate to classify().
# Remove once migration is complete.
# ═══════════════════════════════════════════════════════════════════

def needs_tools(query: str) -> bool:
    """Legacy shim. Use classify() instead."""
    intent = classify(query)
    return not intent.is_conversational and bool(intent.tools)


def is_general_knowledge(query: str) -> bool:
    """Legacy shim. Use classify() instead."""
    intent = classify(query)
    return intent.category == "general" and not intent.needs_web_search


# Router pattern constants kept for backward compatibility with langgraph.py
ROUTER_INGEST_COMMAND = re.compile(
    r'\bingest\s+(now|movies?|items?|\d+|batch)\b'
    r'|\bstart\s+ingesting\b|\badd\s+to\s+(rag|knowledge)\b'
    r'|\bprocess\s+subtitles?\b',
    re.IGNORECASE
)
ROUTER_STATUS_QUERY = re.compile(
    r'\bhow\s+many\s+.*(ingested|in\s+rag)\b'
    r'|\bwhat\s+(has|was)\s+been\s+ingested\b'
    r'|\bitems?\s+(have\s+been|were)\s+ingested\b'
    r'|\bcount\s+.*(items?|in\s+rag)\b|\btotal\s+.*(items?|in\s+rag)\b'
    r'|\b(show|list|display)\s+rag\b',
    re.IGNORECASE
)
ROUTER_MULTI_STEP = re.compile(
    r'\s+and\s+then\s+|\s+then\s+|\s+after\s+that\s+|\s+next\s+'
    r'|\bfirst\b|\bresearch.*analyze\b|\bfind.*summarize\b',
    re.IGNORECASE
)
ROUTER_ONE_TIME_INGEST = re.compile(
    r'\bstop\b|\bthen\s+stop\b|\bdon\'?t\s+continue\b|\bdon\'?t\s+go\s+on\b',
    re.IGNORECASE
)
ROUTER_EXPLICIT_RAG = re.compile(
    r'\busing\s+rag\b|\buse\s+rag\b|\brag\s+tool\b'
    r'|\bwith\s+rag\b|\bsearch\s+rag\b|\bquery\s+rag\b',
    re.IGNORECASE
)
ROUTER_KNOWLEDGE_QUERY = re.compile(
    r'\bwhat\s+is\b|\bwho\s+is\b|\bexplain\b|\btell\s+me\s+about\b',
    re.IGNORECASE
)
ROUTER_EXCLUDE_MEDIA = re.compile(
    r'\bmovie\b|\bplex\b|\bsearch\b|\bfind\b|\bshow\b|\bmedia\b',
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════════
# RESEARCH SOURCE EXTRACTION
# Moved from langgraph.py — used to parse "using X as source" queries
# ═══════════════════════════════════════════════════════════════════

RESEARCH_SOURCE_PATTERN = re.compile(
    r'\busing\s+(?P<source>(?:https?://)?[\w\s\.\-/:]+?)\s+as\s+(a\s+)?source\b'
    r'|\bbased\s+on\s+(?P<source2>(?:https?://)?[\w\s\.\-/:]+?)(?:\s|,|$)'
    r'|\bfrom\s+(?P<source3>(?:https?://)?[\w\s\.\-/:]+?)\s+(?:find|search|get|tell)\b',
    re.IGNORECASE
)


def extract_research_sources(content: str) -> list:
    """
    Extract all sources from a query (handles multiple sources).
    Returns list of sources e.g. ['url1', 'url2', 'domain.com']
    """
    sources = []

    pattern1 = re.compile(
        r'\busing\s+(.+?)\s+as\s+(a\s+)?(source|sources)\b',
        re.IGNORECASE
    )
    match1 = pattern1.search(content)
    if match1:
        source_text = match1.group(1)
        parts = re.split(r'\s+and\s+|,\s*', source_text)
        sources.extend([p.strip().rstrip(',.;:!?') for p in parts if p.strip()])

    pattern2 = re.compile(
        r'\bbased\s+on\s+(.+?)(?:\s+write|\s+create|\s+explain|,|$)',
        re.IGNORECASE
    )
    match2 = pattern2.search(content)
    if match2:
        source_text = match2.group(1)
        parts = re.split(r'\s+and\s+|,\s*', source_text)
        sources.extend([p.strip().rstrip(',.;:!?') for p in parts if p.strip()])

    _image_url_re = re.compile(
        r'/api/v1/(thumbnails|image)/'
        r'|\.(?:jpg|jpeg|png|gif|webp|bmp|heic|tiff?)(?:[?#]|$)',
        re.IGNORECASE
    )
    url_pattern = re.compile(r'https?://[^\s]+')
    for url in url_pattern.findall(content):
        cleaned = url.rstrip(',.;:!?')
        if cleaned and not _image_url_re.search(cleaned):
            sources.append(cleaned)

    return sources


# ═══════════════════════════════════════════════════════════════════
# EXPLICIT SEARCH OVERRIDE PATTERNS
# Moved from langgraph.py — triggers that bypass normal routing
# ═══════════════════════════════════════════════════════════════════

OLLAMA_SEARCH_PATTERN = re.compile(
    r'\bollama\s+search\b'
    r'|\bollama\s+search\s+(for|about|on)\b'
    r'|\bweb\s+search\s+using\s+ollama\b',
    re.IGNORECASE
)

WEB_SEARCH_EXPLICIT_PATTERN = re.compile(
    r'\buse\s+web\s+search\b'
    r'|\busing\s+web\s+search\b'
    r'|\bwith\s+web\s+search\b'
    r'|\bweb\s+search\s+for\b'
    r'|\bvia\s+web\s+search\b',
    re.IGNORECASE
)