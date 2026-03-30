"""
Google OAuth setup script for mcp-platform.
Run this once to generate servers/google/token.json.

Requirements:
  - servers/google/credentials.json must exist (downloaded from Google Cloud Console)
  - OAuth app must be published to "In Production" in Google Auth Platform → Audience
  - mcp-platform must be stopped before running this

Usage:
  .venv/bin/python auth_google.py
"""
from google_auth_oauthlib.flow import InstalledAppFlow
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

CREDENTIALS_FILE = "servers/google/credentials.json"
TOKEN_FILE = "servers/google/token.json"

if not Path(CREDENTIALS_FILE).exists():
    print(f"❌ {CREDENTIALS_FILE} not found — download it from Google Cloud Console")
    exit(1)

print("🌐 Opening browser for Google authentication...")
print("   If you see an 'unverified app' warning:")
print("   Click Advanced → Go to mcp-platform (unsafe) → grant permissions")
print()

flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
creds = flow.run_local_server(port=0)

with open(TOKEN_FILE, "w") as f:
    f.write(creds.to_json())

print(f"✅ {TOKEN_FILE} written — restart mcp-platform to apply")