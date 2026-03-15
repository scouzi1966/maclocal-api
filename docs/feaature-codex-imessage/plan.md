# Telegram Follow-up Plan

## Current iMessage Status

The `--imessage` bridge is implemented as an experimental channel for AFM, with a security-first posture:

- direct iMessage chats only
- explicit sender allowlist
- `/afm` prefix required
- one-time pairing challenge per sender/chat
- image attachments only
- no tools exposed through the bridge

The bridge currently reads inbound events from the local Messages database and sends replies through Messages automation. This works best when AFM runs under a dedicated macOS user signed into a dedicated Apple Account.

## Why We Are Parking It

Operational testing showed a core limitation for local testing and long-term support:

- replies sent from the same Apple account appear as `is_from_me = 1`
- those rows are intentionally ignored for security reasons
- the safest deployment model requires a separate bot Apple Account and separate macOS user session

That makes iMessage viable for private, dedicated setups, but not the best first-class remote control channel.

## Next Feature: Telegram

Telegram is the preferred follow-up integration because it has a cleaner trust and deployment model:

- official Bot API
- stable numeric sender identity
- no local chat DB scraping
- no AppleScript or Messages automation
- simpler allowlisting and replay protection
- easier testing from one machine/account setup

## Telegram v1 Scope

- `--telegram-bot-token <token>`
- `--telegram-allow <comma-separated-user-ids>`
- private chats only
- `/afm` prefix required
- text plus image attachments only
- no tools, shell, filesystem, or gateway actions
- local reuse of AFM `/v1/chat/completions`

## Security Defaults

- allowlist on numeric Telegram user ID only
- reject groups, supergroups, and channels
- optional per-chat pairing or shared-secret auth
- enforce attachment size and type limits
- persist update offsets to prevent replay
- log sender IDs and decisions, never message contents by default

## Recommendation

Keep iMessage experimental and private-use only. Build Telegram as the primary supported remote AFM chat integration.
