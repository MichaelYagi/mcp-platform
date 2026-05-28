---
name: discord
description: >
  Send notifications to Discord channels via webhooks.
  No bot token or server membership required — just a webhook URL.
  Supports multiple named webhooks for different channels.
tags:
  - discord
  - notifications
  - external
tools:
  - discord_notify
  - discord_list_webhooks
---

# Discord Skill

## 🎯 Overview

Post notifications to Discord channels via webhooks. No bot setup, no OAuth, no server membership — just a webhook URL from Discord channel settings.

**Capabilities:**
1. **Notify** — post a message to any configured webhook channel
2. **List webhooks** — see which channels are configured

---

## 📋 Setup

### Step 1 — Create a webhook in Discord

1. Right-click any channel → **Edit Channel**
2. Go to **Integrations → Webhooks → New Webhook**
3. Give it a name (e.g. "MCP Platform")
4. Click **Copy Webhook URL**

### Step 2 — Add to .env

```bash
# Default notification channel
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Optional additional channels (accessible by name)
DISCORD_WEBHOOK_ALERTS=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_GENERAL=https://discord.com/api/webhooks/...
```

### Step 3 — Install dependency

```bash
pip install httpx
```

---

## 📋 Workflows

### Send a notification

```python
# Send to default webhook (DISCORD_WEBHOOK_URL)
discord_notify(message="Deploy is done ✅")

# With a title
discord_notify(title="Invoice Alert", message="New invoice email from john@company.com")

# To a named webhook
discord_notify(message="System alert!", webhook="alerts")

# Custom display name
discord_notify(message="All good.", username="MCP Platform")
```

**Returns:**
```json
{
  "status": "sent",
  "channel": "default",
  "content": "Deploy is done ✅"
}
```

### List configured webhooks

```python
discord_list_webhooks()
```

**Returns:**
```json
{
  "count": 2,
  "webhooks": [
    { "name": "url", "url_masked": "https://discord.com/api/webhooks/123/***/***" },
    { "name": "alerts", "url_masked": "https://discord.com/api/webhooks/456/***/***" }
  ]
}
```

---

## 🚀 Examples

### Example 1: Email condition → Discord notification

```
User: "Every 15 minutes, check if I have unread emails from john@company.com —
       if so, reply saying I'll get back to him and notify me on Discord"

Scheduler job:
  condition_tool:      gmail_search
  condition_tool_args: {query: "from:john@company.com is:unread"}
  condition_expr:      len_messages > 0
  llm_prompt:          "Reply to the first matching email saying I'll get back
                        to them shortly, then call discord_notify with a message
                        saying I got an email from John"
  interval:            15 min
```

### Example 2: Invoice alert

```
User: "When an invoice email arrives, notify me on Discord"

Scheduler job:
  condition_tool:      gmail_search
  condition_tool_args: {query: "subject:invoice is:unread"}
  condition_expr:      len_messages > 0
  llm_prompt:          "Call discord_notify with title='Invoice Received' and
                        message showing who the email is from"
  interval:            15 min
```

### Example 3: Multiple channels

```python
# Alert channel for urgent things
discord_notify(message="🚨 Server down!", webhook="alerts")

# General channel for info
discord_notify(message="Daily briefing ready", webhook="general")
```

---

## 🔧 Tool Reference

| Tool | Purpose | Key Args |
|------|---------|----------|
| `discord_notify` | Post a message to a channel | `message`, `webhook`, `title`, `username` |
| `discord_list_webhooks` | List configured webhooks | — |

---

## ⚠️ Notes

**Webhook URLs are secrets.** Keep them in `.env` and out of git. Anyone with the URL can post to that channel.

**Multiple channels** are supported via `DISCORD_WEBHOOK_<NAME>` env vars. The name after `DISCORD_WEBHOOK_` becomes the webhook's alias (lowercased), usable as the `webhook` argument.

**No bot needed.** Webhooks are outbound-only — perfect for notifications. If you need to read messages or react to incoming messages, a bot would be required.

**Discord markdown** works in messages: `**bold**`, `*italic*`, `` `code` ``, `> quote`.