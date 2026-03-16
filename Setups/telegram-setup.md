# Telegram Setup

## Overview

AFM can expose a remote chat channel through a Telegram bot. The secure model is:

- AFM authenticates to Telegram with a bot token
- AFM only responds to allowlisted Telegram numeric user IDs
- only private chats are accepted
- Telegram messages are accepted directly by default; if you want command gating, set a required prefix such as `--telegram-require-prefix "/afm"`

## Step 1: Create a Telegram Account

Install Telegram on your iPhone and create an account if you do not already have one.

## Step 2: Create a Bot with BotFather

In Telegram, open a chat with `@BotFather`.

Send:

```text
/newbot
```

Example:

```text
You -> @BotFather: /newbot
BotFather -> You: Alright, a new bot. How are we going to call it?
You -> @BotFather: AFM Remote Bot
BotFather -> You: Good. Now let's choose a username for your bot.
You -> @BotFather: afm_remote_helper_bot
BotFather -> You: Done. Use this token to access the HTTP API:
123456789:AAExampleTokenValueExample123456789
```

From that example:

- bot display name: `AFM Remote Bot`
- bot username: `afm_remote_helper_bot`
- bot token: `123456789:AAExampleTokenValueExample123456789`

The full `123456789:...` string is the token. Treat it like a password.

## Step 3: Get Your Telegram User ID

Open a chat with `@userinfobot` or `@RawDataBot`.

Example reply:

```text
Id: 555111222
First: Sylvain
Username: scouzi1966
```

From that example:

- Telegram user ID: `555111222`

This is what AFM uses for the allowlist.

## Step 4: Start a Chat with Your Bot

In Telegram, search for your bot username:

```text
@afm_remote_helper_bot
```

Open the private chat and press `Start`.

This matters because Telegram bots generally need the user to initiate the conversation first.

## Step 5: Start AFM with Telegram Enabled

Example:

```bash
/Volumes/edata/codex/dev/git/maclocal-api/.build/arm64-apple-macosx/release/afm mlx \
  -m mlx-community/Qwen3.5-35B-A3B-4bit \
  --telegram-bot-token "123456789:AAExampleTokenValueExample123456789" \
  --telegram-allow "555111222" \
  --telegram-format "markdown"
```

Meaning:

- `--telegram-bot-token`: the bot token from BotFather
- `--telegram-allow`: your numeric Telegram user ID
- `--telegram-format`: reply format: `markdown`, `plain`, or `html`
- `--telegram-require-prefix`: require a specific prefix for Telegram messages, for example `"/afm"`; off by default

`markdown` is the default.

## Step 6: Send a Test Message

In the private bot chat, send:

```text
hello
```

You should receive a reply from AFM.

## Step 7: Try an Image

Send a photo in the same private chat with:

```text
describe this image
```

## Expected AFM Logs

You should see logs like:

```text
[Telegram] detected event from user 555111222 chat=555111222 type=private photos=0
[Telegram] accepted Telegram AFM request from user 555111222 with 0 image attachment(s)
[Telegram] sent reply to Telegram user 555111222
```

AFM logs sender IDs and decisions, but not message contents.

## Rules

- only private chats are supported
- only allowlisted numeric user IDs are accepted
- plain chat messages are accepted by default
- if you prefer explicit command gating, start AFM with `--telegram-require-prefix "/afm"`
- photos are supported
- tools are not exposed through Telegram
- replies default to Telegram Markdown rendering

## Common Confusion

These are three different values:

- bot username: `afm_remote_helper_bot`
  - used to find the bot in Telegram
- bot token: `123456789:AAExampleTokenValueExample123456789`
  - used by AFM to connect to Telegram
- Telegram user ID: `555111222`
  - used by AFM to decide who is allowed to talk to the bot

## Security Notes

- keep the bot token secret
- if the token leaks, regenerate it through `@BotFather`
- do not rely on usernames for authorization
- use only numeric Telegram user IDs in `--telegram-allow`
- do not add the bot to public groups if you want a narrow control surface
