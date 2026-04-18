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
import json
import sys
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

# Validate credentials.json exists
if not Path(CREDENTIALS_FILE).exists():
    print(f"❌ {CREDENTIALS_FILE} not found — download it from Google Cloud Console")
    sys.exit(1)

# Validate credentials.json is valid JSON with expected structure
try:
    with open(CREDENTIALS_FILE) as f:
        creds_data = json.load(f)
    if "installed" not in creds_data and "web" not in creds_data:
        print(f"❌ {CREDENTIALS_FILE} is not a valid OAuth client credentials file")
        print("   Download a fresh copy from Google Cloud Console → APIs & Services → Credentials")
        sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ {CREDENTIALS_FILE} is malformed JSON: {e}")
    print("   Download a fresh copy from Google Cloud Console → APIs & Services → Credentials")
    sys.exit(1)
except Exception as e:
    print(f"❌ Could not read {CREDENTIALS_FILE}: {e}")
    sys.exit(1)

print("🌐 Opening browser for Google authentication...")
print("   If you see an 'unverified app' warning:")
print("   Click Advanced → Go to mcp-platform (unsafe) → grant permissions")
print()

try:
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
except Exception as e:
    print(f"❌ OAuth flow failed: {e}")
    print("   Make sure your browser opened and you completed the sign-in.")
    print("   If the browser did not open, check that a browser is available.")
    sys.exit(1)

try:
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    print(f"✅ {TOKEN_FILE} written — restart mcp-platform to apply")
except Exception as e:
    print(f"❌ Could not write {TOKEN_FILE}: {e}")
    sys.exit(1)