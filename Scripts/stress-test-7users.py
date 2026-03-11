#!/usr/bin/env python3
"""
Stress test: 7 sustained concurrent users with AI-driven conversations.
Each user is a persona with a communication style, topic interests, and
intelligent follow-up logic that reads the model's actual response to
craft contextual multi-turn exchanges.

Usage: python3 Scripts/stress-test-7users.py [--port 9999] [--duration 1800] [--users 7]
"""

import asyncio
import aiohttp
import json
import time
import random
import re
import argparse
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:9999"
MAX_USERS = 7
TEST_DURATION = 1800  # 30 minutes
MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"

# ─── Tool definitions ────────────────────────────────────────────────────────

TOOLS_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or region name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}

TOOLS_CALC = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression"},
                "precision": {"type": "integer", "description": "Decimal places in result"}
            },
            "required": ["expression"]
        }
    }
}

TOOLS_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search internal knowledge base documents",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (1-20)"},
                "category": {"type": "string", "enum": ["all", "engineering", "product", "legal", "hr"]}
            },
            "required": ["query"]
        }
    }
}

TOOLS_TICKET = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": "Create a JIRA ticket",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "assignee": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["title", "description", "priority"]
        }
    }
}

TOOLS_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "encoding": {"type": "string", "enum": ["utf-8", "ascii", "latin-1"]}
            },
            "required": ["path"]
        }
    }
}

TOOLS_DB = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Execute a read-only SQL query against the analytics database",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT query"},
                "database": {"type": "string", "enum": ["analytics", "users", "products"]},
                "limit": {"type": "integer"}
            },
            "required": ["sql", "database"]
        }
    }
}

ALL_TOOLS = [TOOLS_WEATHER, TOOLS_CALC, TOOLS_SEARCH, TOOLS_TICKET, TOOLS_FILE, TOOLS_DB]

# ─── Fake tool responses ─────────────────────────────────────────────────────

# Precomputed results for calculator to avoid code execution
CALC_RESULTS = {
    "compound interest": 11576.25,
    "10000": 10000,
    "mortgage": 1342.05,
    "savings": 52750.0,
    "roi": 23.5,
    "default": 42.0,
}

def fake_tool_response(name: str, args: dict) -> str:
    """Generate a plausible fake response for a tool call."""
    if name == "get_weather":
        loc = args.get("location", "Unknown")
        temp = random.randint(-5, 38)
        conditions = random.choice(["sunny", "partly cloudy", "overcast", "light rain", "heavy rain", "thunderstorm", "snow", "foggy", "windy", "clear"])
        humidity = random.randint(30, 95)
        return json.dumps({"location": loc, "temperature": temp, "units": "celsius", "conditions": conditions, "humidity": humidity, "wind_speed_kmh": random.randint(5, 60)})
    elif name == "calculate":
        expr = args.get("expression", "")
        # Return a plausible result without code execution
        result = CALC_RESULTS.get("default")
        for key, val in CALC_RESULTS.items():
            if key in expr.lower():
                result = val
                break
        # Add some variation
        result = result * (1 + random.uniform(-0.05, 0.05))
        return json.dumps({"expression": expr, "result": round(result, args.get("precision", 4))})
    elif name == "search_documents":
        query = args.get("query", "")
        n = min(args.get("max_results", 5), 5)
        docs = []
        titles = [
            f"Architecture Decision Record: {query.title()}",
            f"RFC: {query.title()} Redesign Proposal",
            f"Runbook: {query.title()} Troubleshooting",
            f"Design Doc: {query.title()} v2",
            f"Post-mortem: {query.title()} Incident 2025-11",
            f"Guide: Getting Started with {query.title()}",
        ]
        for i in range(n):
            docs.append({"title": titles[i % len(titles)], "path": f"/docs/{query.replace(' ', '-')}/{i+1}.md", "relevance": round(random.uniform(0.6, 0.99), 2), "last_updated": "2025-12-15"})
        return json.dumps({"results": docs, "total": random.randint(n, n*3)})
    elif name == "create_ticket":
        return json.dumps({"ticket_id": f"PROJ-{random.randint(1000, 9999)}", "status": "created", "url": f"https://jira.example.com/browse/PROJ-{random.randint(1000, 9999)}"})
    elif name == "read_file":
        path = args.get("path", "unknown.txt")
        if path.endswith(".py"):
            return json.dumps({"content": f"# {path}\nimport os\nimport sys\n\ndef main():\n    print('Hello from {path}')\n\nif __name__ == '__main__':\n    main()\n", "size_bytes": random.randint(200, 5000)})
        elif path.endswith(".json"):
            return json.dumps({"content": json.dumps({"version": "2.1.0", "database": {"host": "localhost", "port": 5432}}, indent=2), "size_bytes": random.randint(100, 2000)})
        else:
            return json.dumps({"content": f"Contents of {path}\nLine 1\nLine 2\nLine 3\n", "size_bytes": random.randint(50, 10000)})
    elif name == "query_database":
        return json.dumps({"columns": ["id", "name", "value", "created_at"], "rows": [[random.randint(1, 1000), f"item_{i}", round(random.uniform(1, 1000), 2), "2025-12-01"] for i in range(min(args.get("limit", 5), 5))], "row_count": random.randint(1, 500), "execution_time_ms": random.randint(5, 250)})
    return json.dumps({"status": "ok", "result": "done"})


# ─── Personas ────────────────────────────────────────────────────────────────

PERSONAS = [
    {
        "name": "alex-backend",
        "style": "terse",
        "role": "Senior backend engineer",
        "interests": ["distributed systems", "databases", "APIs", "caching", "concurrency"],
        "tool_affinity": 0.6,
        "session_length": (2, 6),
        "think_time": (1.0, 4.0),
        "system": "You are a helpful engineering assistant. Be precise and technical.",
    },
    {
        "name": "priya-pm",
        "style": "verbose",
        "role": "Product manager",
        "interests": ["user metrics", "feature specs", "roadmap", "competitive analysis", "A/B testing"],
        "tool_affinity": 0.4,
        "session_length": (3, 8),
        "think_time": (3.0, 10.0),
        "system": "You are a helpful product assistant. Consider business impact and user experience.",
    },
    {
        "name": "sam-junior",
        "style": "questioning",
        "role": "Junior developer",
        "interests": ["Python basics", "git", "debugging", "code review", "testing"],
        "tool_affinity": 0.2,
        "session_length": (3, 7),
        "think_time": (2.0, 8.0),
        "system": "You are a patient mentor helping a junior developer learn.",
    },
    {
        "name": "maria-data",
        "style": "analytical",
        "role": "Data scientist",
        "interests": ["SQL queries", "statistics", "ML pipelines", "data quality", "visualization"],
        "tool_affinity": 0.7,
        "session_length": (2, 5),
        "think_time": (2.0, 6.0),
        "system": "You are a data engineering assistant. Be precise with numbers and queries.",
    },
    {
        "name": "kai-devops",
        "style": "urgent",
        "role": "DevOps/SRE engineer",
        "interests": ["kubernetes", "CI/CD", "monitoring", "incidents", "infrastructure"],
        "tool_affinity": 0.5,
        "session_length": (1, 4),
        "think_time": (0.5, 3.0),
        "system": "You are an infrastructure assistant. Help with ops, deployments, and incident response.",
    },
    {
        "name": "lisa-frontend",
        "style": "creative",
        "role": "Frontend engineer",
        "interests": ["React", "CSS", "accessibility", "animations", "design systems"],
        "tool_affinity": 0.3,
        "session_length": (2, 6),
        "think_time": (2.0, 7.0),
        "system": "You are a frontend development assistant. Help with UI, components, and web technologies.",
    },
    {
        "name": "chen-security",
        "style": "thorough",
        "role": "Security engineer",
        "interests": ["auth", "encryption", "CVEs", "penetration testing", "compliance"],
        "tool_affinity": 0.5,
        "session_length": (2, 5),
        "think_time": (3.0, 8.0),
        "system": "You are a security-focused assistant. Consider threats, vulnerabilities, and best practices.",
    },
    {
        "name": "diego-mobile",
        "style": "terse",
        "role": "iOS/Android developer",
        "interests": ["Swift", "SwiftUI", "Kotlin", "app performance", "push notifications"],
        "tool_affinity": 0.3,
        "session_length": (2, 5),
        "think_time": (2.0, 6.0),
        "system": "You are a mobile development assistant. Help with iOS, Android, and cross-platform topics.",
    },
    {
        "name": "nina-ml",
        "style": "analytical",
        "role": "ML engineer",
        "interests": ["model training", "inference optimization", "transformers", "quantization", "MLX"],
        "tool_affinity": 0.5,
        "session_length": (3, 7),
        "think_time": (3.0, 10.0),
        "system": "You are an ML engineering assistant. Help with model development, training, and deployment.",
    },
    {
        "name": "omar-manager",
        "style": "verbose",
        "role": "Engineering manager",
        "interests": ["technical debt", "roadmap", "architecture reviews", "sprint planning", "hiring"],
        "tool_affinity": 0.4,
        "session_length": (2, 5),
        "think_time": (4.0, 12.0),
        "system": "You are an engineering management assistant. Help with leadership, planning, and team concerns.",
    },
]

# ─── Opening prompts by domain ──────────────────────────────────────────────

OPENERS = {
    "distributed systems": [
        "We're seeing intermittent timeouts between our API gateway and the user service. p99 latency jumped from 50ms to 800ms. Where should I look?",
        "I need to design a distributed rate limiter that works across 12 pods. Token bucket or sliding window?",
        "Our Kafka consumer group is lagging by 2 million messages. The producers are fine. What's the debugging approach?",
        "Explain the CAP theorem trade-offs for our new inventory system. We need strong consistency for stock counts.",
    ],
    "databases": [
        "This Postgres query takes 4.2 seconds on 50M rows:\n```sql\nSELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.created_at > '2025-01-01' GROUP BY u.name ORDER BY COUNT(o.id) DESC LIMIT 100;\n```\nHow do I optimize it?",
        "Should we shard our users table? We're at 200M rows, writes are 5k/sec, reads 50k/sec.",
        "Compare DynamoDB vs ScyllaDB for a time-series IoT workload — 100k events/sec, 90-day retention.",
        "We need to migrate from MongoDB to Postgres without downtime. 500GB of data. What's the strategy?",
    ],
    "APIs": [
        "Design the REST API for a multi-tenant SaaS billing system. Need to handle subscriptions, invoices, and usage-based pricing.",
        "Our API returns nested JSON 6 levels deep. Clients hate it. How do we flatten without breaking backwards compatibility?",
        "What's the best approach for API versioning? We have 200+ endpoints and 50 external consumers.",
        "Implement idempotency keys for our payment API. What's the storage and retry strategy?",
    ],
    "caching": [
        "Our Redis cluster is using 120GB. We allocated 64GB. What's the eviction strategy?",
        "Design a multi-layer cache: L1 in-process, L2 Redis, L3 database. How do I handle invalidation?",
        "Cache stampede is killing our DB during cache expiry. How do we implement probabilistic early expiration?",
    ],
    "user metrics": [
        "Pull the DAU/MAU ratio for the last 6 months and identify the drop-off point.",
        "We need to measure feature adoption for the new search. What events should we track?",
        "Build a cohort analysis showing retention by signup month for Q4 2025.",
        "Our NPS dropped from 42 to 31. How do I investigate which user segment is driving the drop?",
    ],
    "feature specs": [
        "Write the spec for a notification preferences system. Users should control email, push, and in-app for each notification type.",
        "We want to add collaborative editing to our docs feature. What are the key technical decisions?",
        "Spec out an AI-powered search that combines keyword matching with semantic search. Budget: 2 engineers, 6 weeks.",
    ],
    "Python basics": [
        "What's the difference between a list and a tuple? When should I use each?",
        "I keep getting 'TypeError: unhashable type: list' when using a list as a dict key. Why?",
        "How do decorators work in Python? Can you explain with a simple example?",
        "What's the difference between `==` and `is` in Python?",
    ],
    "debugging": [
        "My Python script works fine locally but crashes in production with `MemoryError`. It processes a 2GB CSV file. Help.",
        "I have a race condition in my async code. Two coroutines are updating the same dict. How do I debug it?",
        "Getting `RecursionError: maximum recursion depth exceeded`. The function processes a tree structure. What's wrong?",
    ],
    "git": [
        "I accidentally committed a secret to git. It's already pushed. How do I remove it from history?",
        "Explain git rebase vs merge. My team argues about this constantly.",
        "I did `git reset --hard` and lost my changes. Is there any way to recover?",
    ],
    "SQL queries": [
        "Write a query to find users who made purchases in every month of 2025.",
        "I need a moving 7-day average of daily revenue. The data has gaps (no rows for zero-revenue days).",
        "Optimize this: `SELECT * FROM events WHERE json_extract(payload, '$.type') = 'purchase'` — it's scanning 100M rows.",
    ],
    "statistics": [
        "Is 3% conversion rate improvement from our A/B test statistically significant? Control: 12.1% (n=50k), Treatment: 15.1% (n=48k).",
        "Explain the difference between correlation and causation with a real example from our product data.",
        "Our model's AUC is 0.85 but precision at the decision threshold is only 0.3. What's going on?",
    ],
    "kubernetes": [
        "Pod keeps getting OOMKilled. Memory limit is 512Mi, actual usage peaks at 480Mi. What's happening?",
        "Design a blue-green deployment strategy for our stateful service that uses PersistentVolumes.",
        "Our cluster costs $40k/month. How do I identify which teams/services are overspending?",
        "HPA keeps scaling to max replicas (20) even though CPU is only at 30%. What's wrong?",
    ],
    "incidents": [
        "URGENT: Production database is at 99% disk. It's 3 AM. Walk me through emergency remediation.",
        "Our CDN is serving stale content after a deploy 2 hours ago. Cache invalidation didn't work. What now?",
        "SSL cert expired on our main domain. Automated renewal failed. 500 errors everywhere.",
    ],
    "React": [
        "My React component re-renders 47 times when I type one character. How do I debug and fix this?",
        "Implement a virtualized infinite scroll list that handles 100k items smoothly.",
        "Should we migrate from Redux to Zustand? We have 200 components using Redux. What's the incremental strategy?",
    ],
    "CSS": [
        "This flexbox layout breaks on Safari. Works fine in Chrome and Firefox. The container has `gap: 16px` and nested flex items.",
        "Build a responsive grid that's 4 columns on desktop, 2 on tablet, 1 on mobile, with smooth transitions between breakpoints.",
        "How do I create a glassmorphism card component that works across all browsers including Firefox?",
    ],
    "auth": [
        "Design an OAuth2 + PKCE flow for our mobile app that also supports biometric login.",
        "We store passwords with bcrypt (cost 10). Should we migrate to argon2id? What's the incremental approach?",
        "A researcher found an IDOR vulnerability in our API. The endpoint is `/api/users/{id}/documents`. How do we fix it properly?",
    ],
    "encryption": [
        "We need to encrypt PII at rest in Postgres. Column-level encryption vs transparent data encryption?",
        "Design a key rotation strategy for our AES-256-GCM encrypted files. We have 50TB encrypted data.",
        "Our JWT tokens are signed with HS256. Security audit says switch to RS256. What's the migration plan?",
    ],
    "Swift": [
        "What's the difference between `actor` and `class` in Swift? When should I use actors?",
        "My SwiftUI view is laggy when scrolling a List with 10k items. How do I profile and fix it?",
        "Implement a generic, thread-safe cache in Swift using actors. Should support TTL expiration.",
        "Explain Swift's ownership model — borrowing, consuming, sending. I'm confused by the compiler errors.",
    ],
    "model training": [
        "My transformer model's loss plateaus at 2.3 after 10k steps. Learning rate is 1e-4, batch size 32. What should I try?",
        "Compare LoRA vs QLoRA vs full fine-tuning for adapting a 7B model to medical Q&A. Budget: 1x A100 for 24 hours.",
        "Training a GPT-2 from scratch on 100GB of code. What's the optimal tokenizer configuration?",
    ],
    "inference optimization": [
        "Our BERT inference takes 45ms on CPU. Need to get under 10ms. What are the optimization options?",
        "Compare vLLM vs TGI vs SGLang for serving a 70B model. We need 100 concurrent users.",
        "Quantize our 13B model from fp16 to int4. What's the expected quality degradation and speedup?",
    ],
    "technical debt": [
        "We have 3 services that each implement their own auth. How do I make the case for consolidation?",
        "Our test suite takes 45 minutes. It was 10 minutes a year ago. How do we diagnose and fix?",
        "The team wants to rewrite the monolith in Go. Current stack is Python/Django. Should we?",
    ],
    "roadmap": [
        "We need to plan Q2. We have 8 engineers, 3 major features requested, and a pile of tech debt. How do I prioritize?",
        "Our competitor just launched real-time collaboration. Do we fast-follow or differentiate?",
    ],
    "concurrency": [
        "Explain the difference between async/await, threads, and multiprocessing in Python. When to use which?",
        "I have a producer-consumer pattern with 1 producer and 8 consumers. The queue backs up. How do I tune it?",
        "Implement a read-write lock in Swift that favors writers to prevent starvation.",
    ],
    "ML pipelines": [
        "Design a feature store that serves both batch training and real-time inference. What's the architecture?",
        "Our ML pipeline takes 6 hours. Feature engineering is 4 of those hours. How do we optimize?",
    ],
    "monitoring": [
        "Set up alerting for our payment service. What are the golden signals and thresholds?",
        "Our Prometheus is using 200GB of storage for 2 weeks of data. How do we reduce it without losing visibility?",
    ],
    "accessibility": [
        "Audit this component for WCAG 2.1 AA compliance:\n```jsx\n<div onClick={handleClick} style={{color: '#999', fontSize: '11px'}}>\n  <img src='icon.png'/> Click here\n</div>```",
        "How do I make a custom dropdown accessible? It needs to work with screen readers and keyboard navigation.",
    ],
    "quantization": [
        "What's the difference between GPTQ, AWQ, and GGUF quantization? Which one should I use for serving on Apple Silicon?",
        "Our 4-bit quantized model has noticeable quality degradation on math tasks. Is there a way to selectively quantize?",
    ],
    "MLX": [
        "How does MLX handle memory differently from PyTorch on Apple Silicon? I'm seeing different peak memory usage.",
        "What's the optimal batch size for MLX inference on M3 Ultra with a 35B MoE model?",
        "Compare MLX vs llama.cpp for local inference. Which is faster for interactive use?",
    ],
    "competitive analysis": [
        "Our main competitor just raised $50M and is expanding into our market. How should we respond strategically?",
        "Compare our product's feature set against the top 3 competitors. Where are we falling behind?",
    ],
    "A/B testing": [
        "Design an A/B test framework for our checkout flow. We need to handle multiple concurrent experiments without interference.",
        "Our A/B test shows a 2% lift in conversion but the p-value is 0.08. Should we ship it?",
    ],
    "code review": [
        "Review this pull request diff — it adds retry logic to our HTTP client:\n```python\ndef fetch(url, retries=3):\n    for i in range(retries):\n        try:\n            return requests.get(url, timeout=30)\n        except:\n            if i == retries - 1:\n                raise\n            time.sleep(2 ** i)\n```\nWhat issues do you see?",
        "Our code review process takes 3 days on average. How do we speed it up without sacrificing quality?",
    ],
    "testing": [
        "What's the right ratio of unit tests to integration tests to e2e tests? We're at 80/15/5 and it feels wrong.",
        "How do I test async Python code that uses aiohttp? I keep getting event loop errors in pytest.",
        "Our test suite is flaky — 5% of runs fail due to timing issues. How do we fix this systematically?",
    ],
    "CI/CD": [
        "Our CI pipeline takes 25 minutes. Breakdown: 5min build, 15min tests, 5min deploy. Where do we optimize?",
        "Design a CI/CD pipeline for a monorepo with 8 services. Only changed services should be built and deployed.",
    ],
    "infrastructure": [
        "We're migrating from EC2 to EKS. 40 services, 200 instances. What's the migration plan?",
        "Our AWS bill is $80k/month. The biggest items are RDS ($20k) and EC2 ($35k). How do we optimize?",
    ],
    "data quality": [
        "We found that 12% of our user records have invalid email addresses. How do we clean this up without breaking things?",
        "Design a data quality monitoring system that catches schema drift, null rate changes, and distribution shifts.",
    ],
    "visualization": [
        "What's the best chart type for showing conversion funnel data across 5 steps and 3 user segments?",
        "Build a dashboard layout for our executive team. They need: revenue, DAU, churn rate, and NPS at a glance.",
    ],
    "animations": [
        "Implement a smooth page transition animation between a list view and detail view in React.",
        "How do I create a physics-based spring animation for a draggable card component?",
    ],
    "design systems": [
        "We're building a design system from scratch. What are the essential components to start with?",
        "Our design system has 200 components but teams keep building custom ones. How do we improve adoption?",
    ],
    "CVEs": [
        "We just got flagged for CVE-2024-3094 (xz backdoor). Our systems use xz 5.6.1. What's the immediate action plan?",
        "How do we set up automated CVE scanning for our Docker images in the CI pipeline?",
    ],
    "penetration testing": [
        "Walk me through a basic penetration test plan for our REST API. What should I test first?",
        "We found an XSS vulnerability in our search page. The input isn't sanitized before rendering. What's the fix?",
    ],
    "compliance": [
        "We need SOC 2 Type II certification. What are the most painful requirements and how do we prepare?",
        "Our EU customers are asking about GDPR compliance. We store PII in 3 different databases. What's the audit plan?",
    ],
    "SwiftUI": [
        "How do I build a custom navigation stack in SwiftUI that supports deep linking?",
        "My SwiftUI app's memory usage grows every time I navigate between views. How do I debug this?",
    ],
    "Kotlin": [
        "Compare Kotlin coroutines with Swift's structured concurrency. Which model is safer?",
        "How do I implement offline-first data sync in a Kotlin Android app? We use Room and Retrofit.",
    ],
    "app performance": [
        "Our iOS app's launch time is 4.2 seconds. The Xcode Instruments trace shows dylib loading takes 2.8s. How do we fix this?",
        "The app drops to 30fps when scrolling our feed. It has images, text, and video thumbnails. What's the optimization strategy?",
    ],
    "push notifications": [
        "Design a push notification system that handles delivery across iOS, Android, and web. We need read receipts.",
        "Our push notification open rate dropped from 8% to 3%. How do we diagnose and improve?",
    ],
    "transformers": [
        "Explain the attention mechanism in transformers. Specifically, why is the scaling factor 1/sqrt(d_k) important?",
        "How does Flash Attention reduce memory usage? What's the trade-off?",
    ],
    "hiring": [
        "Design a technical interview process for senior backend engineers. We want to assess system design and coding.",
        "We're hiring 5 engineers in Q2. How do we maintain our technical bar while scaling the team?",
    ],
    "sprint planning": [
        "Our sprints keep running over. We commit to 40 story points but deliver 25. How do we improve estimation?",
        "How do we balance feature work, tech debt, and bug fixes in our sprint planning?",
    ],
    "architecture reviews": [
        "Review this architecture: Client -> API Gateway -> Auth Service -> [User Service, Order Service, Payment Service] -> Postgres. What are the weaknesses?",
        "We're considering event sourcing for our order management system. What are the gotchas?",
    ],
}

# ─── Follow-up generators ───────────────────────────────────────────────────

def generate_followup(persona: dict, response_text: str, turn: int, total_turns: int, had_tool_calls: bool) -> str:
    """Generate a contextual follow-up based on the model's response and persona style."""

    style = persona["style"]
    response_lower = response_text.lower() if response_text else ""
    response_len = len(response_text) if response_text else 0

    # If model gave code, ask about it
    if "```" in response_text:
        code_followups = [
            "Can you add error handling to that?",
            "What about edge cases? What if the input is empty or null?",
            "How would I write tests for this?",
            "Can you make this more performant?",
            "What's the time complexity of this approach?",
            "Can you refactor this to be more readable?",
            "Is there a more idiomatic way to write this?",
            "What imports/dependencies does this need?",
            "Can you add type hints/annotations?",
            "How would I integrate this into an existing codebase?",
        ]
        return random.choice(code_followups)

    # If model mentioned trade-offs or options
    if any(w in response_lower for w in ["option 1", "option 2", "alternatively", "trade-off", "on the other hand", "approach 1", "approach 2"]):
        tradeoff_followups = [
            "Let's go with the first approach. Can you elaborate on the implementation?",
            "What would you recommend for our scale (100k requests/sec)?",
            "Which option has the lowest operational overhead?",
            "Can you compare them in terms of cost?",
            "What are the failure modes of each approach?",
            "Which one is easier to roll back if something goes wrong?",
            "Let's go with option 2. Walk me through the steps.",
        ]
        return random.choice(tradeoff_followups)

    # If model asked a clarifying question
    if "?" in response_text[-200:] and any(w in response_lower for w in ["could you", "do you", "what is", "how many", "which", "are you"]):
        clarification_answers = [
            "We have about 50 million users, 5k requests per second at peak.",
            "It's a Python 3.11 backend with FastAPI, Postgres, and Redis.",
            "The team is 6 engineers, we need this done in 4 weeks.",
            "We're on AWS, using EKS for orchestration. Budget isn't a major constraint.",
            "It needs to be backwards compatible with our existing API consumers.",
            "We're running on Apple Silicon M3 Ultra machines with 192GB RAM.",
            "The data is mostly structured JSON, average document size is 2KB.",
            "Yes, we already have monitoring with Prometheus and Grafana.",
        ]
        return random.choice(clarification_answers)

    # If model gave a list/steps
    if any(f"{i}." in response_text for i in range(1, 6)) or "- " in response_text:
        list_followups = [
            "Can you expand on step 1? That's the part I'm least familiar with.",
            "What's the most common mistake people make with this approach?",
            "How long would each step take approximately?",
            "Which step is the riskiest? Where should I be extra careful?",
            "Can you give me a concrete example for the third point?",
            "Are these steps sequential or can some run in parallel?",
        ]
        return random.choice(list_followups)

    # If response was very long
    if response_len > 1500:
        return random.choice([
            "That's very thorough. Can you summarize the key takeaways in 3 bullet points?",
            "TL;DR? What's the one thing I should do first?",
            "Good overview. What's the most critical thing to get right?",
            "If I only have 2 hours, which part should I focus on?",
        ])

    # If tool calls happened
    if had_tool_calls:
        tool_followups = [
            "Interesting results. Can you dig deeper into the data?",
            "Can you cross-reference that with last month's numbers?",
            "Run the same query but filter for the US region only.",
            "OK now based on those results, what do you recommend?",
            "Can you search for more context on this?",
            "Calculate the percentage change from last quarter.",
            "Good. Now create a ticket to track the action items from this analysis.",
        ]
        return random.choice(tool_followups)

    # Style-based generic follow-ups
    generic = {
        "terse": [
            "Go deeper on that.",
            "Example?",
            "What about performance implications?",
            "OK but what about at our scale?",
            "Show me the code.",
            "How would you test this?",
            "What's the failure mode?",
        ],
        "verbose": [
            "That makes sense. I'm also wondering about how this would impact our roadmap — we committed to launching the new dashboard by end of Q2, and if this takes longer than expected, we might need to re-prioritize. What's your estimate for effort?",
            "Great point about the trade-offs. From a product perspective, I think we should optimize for user experience first and worry about performance later. Can you sketch out what the ideal user flow would look like?",
            "I had a conversation with the VP of engineering about this last week. They mentioned concerns about the maintenance burden. How do we mitigate that?",
            "Hmm, let me think about that from the customer's perspective. Our enterprise clients specifically asked for audit logging on this. Can we bake that in from the start?",
        ],
        "questioning": [
            "Wait, why does that work? I thought Python was pass-by-reference?",
            "Can you explain that more simply? I'm not sure I understand the async part.",
            "Is there a YouTube video or blog post that explains this well? I learn better visually.",
            "So if I change this, will it break the tests? I'm worried about breaking things.",
            "How would a senior developer approach this differently?",
            "I tried your suggestion but got a different error now: `KeyError: 'config'`. What does that mean?",
        ],
        "analytical": [
            "What's the statistical significance of this result?",
            "Can you show me the distribution, not just the average?",
            "What's the confidence interval on that estimate?",
            "How would this change if we used a different baseline period?",
            "Run the numbers with a 30-day window instead of 7.",
            "What's the correlation between these two metrics?",
        ],
        "urgent": [
            "OK what's the FASTEST fix? Production is down.",
            "Skip the theory. Give me the commands to run.",
            "Is there a rollback option? We might need to revert.",
            "Who else should I loop in on this?",
            "ETA on a permanent fix?",
        ],
        "creative": [
            "Ooh I like that approach! Can we add a subtle animation when the state changes?",
            "What if we used CSS grid instead? I feel like flexbox is getting complicated here.",
            "How do we make this look good on both light and dark mode?",
            "Can you make it more playful? The current design feels too corporate.",
            "What would this look like with Tailwind instead of vanilla CSS?",
        ],
        "thorough": [
            "What are the OWASP implications of this approach?",
            "Have you considered the case where an attacker sends malformed input?",
            "What happens if this component is compromised? What's the blast radius?",
            "We need to log this for our SOC 2 audit trail. How?",
            "Walk me through the threat model for this flow.",
        ],
    }

    followups = generic.get(style, generic["terse"])

    # 15% chance of topic change on later turns
    if turn >= 2 and random.random() < 0.15:
        topic = random.choice(persona["interests"])
        openers = OPENERS.get(topic, [])
        if openers:
            return random.choice(openers)

    return random.choice(followups)


# ─── Tracking ────────────────────────────────────────────────────────────────

@dataclass
class RequestLog:
    user_id: str
    session_id: str
    persona: str
    scenario_topic: str
    turn: int
    timestamp_start: str
    timestamp_end: str
    elapsed_ms: int
    status: int
    streaming: bool
    has_tools: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    tool_calls_count: int = 0
    tool_names: list = field(default_factory=list)
    has_thinking: bool = False
    error: Optional[str] = None
    user_message: str = ""
    assistant_preview: str = ""
    messages_count: int = 0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class StressTestTracker:
    def __init__(self, report_dir: str):
        self.report_dir = report_dir
        self.logs: list[RequestLog] = []
        self.lock = asyncio.Lock()
        self.active_users = 0
        self.total_requests = 0
        self.total_errors = 0
        self.total_sessions = 0
        self.start_time = time.time()

    async def log_request(self, entry: RequestLog):
        async with self.lock:
            self.logs.append(entry)
            self.total_requests += 1
            if entry.error:
                self.total_errors += 1

    def elapsed(self):
        return time.time() - self.start_time

    def print_status(self):
        elapsed = self.elapsed()
        rps = self.total_requests / elapsed if elapsed > 0 else 0
        errs = f"  \033[91m{self.total_errors} errors\033[0m" if self.total_errors else ""
        print(f"  [{elapsed:7.1f}s] active={self.active_users}/{MAX_USERS}  reqs={self.total_requests}  sessions={self.total_sessions}  rps={rps:.2f}{errs}")

    def save_jsonl(self):
        path = os.path.join(self.report_dir, "stress-test-log.jsonl")
        with open(path, "w") as f:
            for entry in self.logs:
                f.write(json.dumps(entry.to_dict()) + "\n")
        return path

    def save_report(self):
        path = os.path.join(self.report_dir, "stress-test-report.md")
        elapsed = self.elapsed()

        by_persona = {}
        for e in self.logs:
            by_persona.setdefault(e.persona, []).append(e)

        by_topic = {}
        for e in self.logs:
            by_topic.setdefault(e.scenario_topic, []).append(e)

        latencies = [e.elapsed_ms for e in self.logs if not e.error]
        stream_lat = [e.elapsed_ms for e in self.logs if e.streaming and not e.error]
        nonstream_lat = [e.elapsed_ms for e in self.logs if not e.streaming and not e.error]
        tool_lat = [e.elapsed_ms for e in self.logs if e.has_tools and not e.error]
        thinking_entries = [e for e in self.logs if e.has_thinking]

        def stats(vals):
            if not vals:
                return "n/a", "n/a", "n/a", "n/a", 0
            vals = sorted(vals)
            return (
                f"{sum(vals)/len(vals):.0f}",
                f"{vals[0]}",
                f"{vals[-1]}",
                f"{vals[int(len(vals)*0.95)]}",
                len(vals),
            )

        with open(path, "w") as f:
            f.write(f"# Stress Test Report\n\n")
            f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Duration**: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
            f.write(f"- **Max concurrent users**: {MAX_USERS}\n")
            f.write(f"- **Total sessions**: {self.total_sessions}\n")
            f.write(f"- **Total requests**: {self.total_requests}\n")
            f.write(f"- **Total errors**: {self.total_errors}\n")
            f.write(f"- **Error rate**: {self.total_errors/max(self.total_requests,1)*100:.1f}%\n")
            f.write(f"- **Requests/sec**: {self.total_requests/elapsed:.2f}\n")
            f.write(f"- **Thinking responses**: {len(thinking_entries)} ({len(thinking_entries)/max(len(self.logs),1)*100:.0f}%)\n")
            f.write(f"- **Model**: `{MODEL}`\n\n")

            f.write(f"## Latency (ms)\n\n")
            f.write(f"| Type | Avg | Min | Max | P95 | Count |\n")
            f.write(f"|------|-----|-----|-----|-----|-------|\n")
            for label, vals in [("All", latencies), ("Non-streaming", nonstream_lat), ("Streaming", stream_lat), ("Tool calls", tool_lat)]:
                avg, mn, mx, p95, n = stats(vals)
                f.write(f"| {label} | {avg} | {mn} | {mx} | {p95} | {n} |\n")

            f.write(f"\n## By Persona\n\n")
            f.write(f"| Persona | Role | Sessions | Requests | Errors | Avg ms | Tool Calls | Thinking |\n")
            f.write(f"|---------|------|----------|----------|--------|--------|------------|----------|\n")
            for name, entries in sorted(by_persona.items()):
                sessions = len(set(e.session_id for e in entries))
                errs = sum(1 for e in entries if e.error)
                avg = sum(e.elapsed_ms for e in entries) / len(entries)
                tcs = sum(e.tool_calls_count for e in entries)
                thinks = sum(1 for e in entries if e.has_thinking)
                role = next((p["role"] for p in PERSONAS if p["name"] == name), "?")
                f.write(f"| {name} | {role} | {sessions} | {len(entries)} | {errs} | {avg:.0f} | {tcs} | {thinks} |\n")

            f.write(f"\n## By Topic\n\n")
            f.write(f"| Topic | Requests | Errors | Avg ms |\n")
            f.write(f"|-------|----------|--------|--------|\n")
            for topic, entries in sorted(by_topic.items(), key=lambda x: -len(x[1])):
                errs = sum(1 for e in entries if e.error)
                avg = sum(e.elapsed_ms for e in entries) / len(entries)
                f.write(f"| {topic} | {len(entries)} | {errs} | {avg:.0f} |\n")

            f.write(f"\n## Errors\n\n")
            errors = [e for e in self.logs if e.error]
            if errors:
                f.write(f"| Time | User | Persona | Topic | Turn | Error |\n")
                f.write(f"|------|------|---------|-------|------|-------|\n")
                for e in errors:
                    f.write(f"| {e.timestamp_start[11:]} | {e.user_id} | {e.persona} | {e.scenario_topic} | {e.turn} | `{e.error[:120]}` |\n")
            else:
                f.write("No errors.\n")

            f.write(f"\n## Tool Usage\n\n")
            tool_name_counts = {}
            for e in self.logs:
                for tn in e.tool_names:
                    tool_name_counts[tn] = tool_name_counts.get(tn, 0) + 1
            if tool_name_counts:
                f.write(f"| Tool | Times Called |\n")
                f.write(f"|------|-------------|\n")
                for tn, cnt in sorted(tool_name_counts.items(), key=lambda x: -x[1]):
                    f.write(f"| {tn} | {cnt} |\n")
            tool_entries = [e for e in self.logs if e.tool_calls_count > 0]
            f.write(f"\nTotal responses with tool calls: {len(tool_entries)}\n")

            f.write(f"\n## Request Timeline\n\n")
            f.write(f"| # | Time | User | Persona | Topic | Turn | ms | Tokens | Finish | Tools | Think | Err |\n")
            f.write(f"|---|------|------|---------|-------|------|----|--------|--------|-------|-------|-----|\n")
            for i, e in enumerate(self.logs):
                tk = f"{e.prompt_tokens or '?'}/{e.completion_tokens or '?'}"
                tc = ",".join(e.tool_names) if e.tool_names else ""
                th = "Y" if e.has_thinking else ""
                err = "ERR" if e.error else ""
                f.write(f"| {i+1} | {e.timestamp_start[11:19]} | {e.user_id} | {e.persona} | {e.scenario_topic} | {e.turn} | {e.elapsed_ms} | {tk} | {e.finish_reason or ''} | {tc} | {th} | {err} |\n")

            f.write(f"\n## Prompt Correlation Log\n\n")
            f.write(f"Full prompt text for correlating with server logs.\n\n")
            for i, e in enumerate(self.logs):
                f.write(f"### Request {i+1} — {e.timestamp_start} — {e.user_id} ({e.persona})\n\n")
                f.write(f"**Topic**: {e.scenario_topic} | **Turn**: {e.turn} | **Messages**: {e.messages_count}\n\n")
                f.write(f"**User**: {e.user_message[:500]}\n\n")
                if e.assistant_preview:
                    f.write(f"**Response**: {e.assistant_preview[:300]}...\n\n")
                if e.tool_names:
                    f.write(f"**Tools called**: {', '.join(e.tool_names)}\n\n")
                if e.error:
                    f.write(f"**Error**: `{e.error}`\n\n")
                f.write(f"---\n\n")

        return path


# ─── Request execution ───────────────────────────────────────────────────────

async def make_request(http: aiohttp.ClientSession, messages: list, params: dict,
                       tools: Optional[list], streaming: bool) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        **{k: v for k, v in params.items() if k != "stream"},
    }
    if tools:
        body["tools"] = tools
    if streaming:
        body["stream"] = True

    async with http.post(f"{BASE_URL}/v1/chat/completions", json=body) as resp:
        result = {
            "status": resp.status, "content": "", "finish_reason": None,
            "prompt_tokens": None, "completion_tokens": None,
            "tool_calls": [], "has_thinking": False, "error": None,
            "reasoning_content": "",
        }
        if resp.status != 200:
            result["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
            return result

        if streaming:
            parts = []
            tc_raw = {}
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        result["has_thinking"] = True
                        result["reasoning_content"] += delta["reasoning_content"]
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tc_raw:
                                tc_raw[idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                            if tc.get("function", {}).get("name"):
                                tc_raw[idx]["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                tc_raw[idx]["arguments"] += tc["function"]["arguments"]
                            if tc.get("id"):
                                tc_raw[idx]["id"] = tc["id"]
                    fr = chunk.get("choices", [{}])[0].get("finish_reason")
                    if fr:
                        result["finish_reason"] = fr
                    usage = chunk.get("usage")
                    if usage:
                        result["prompt_tokens"] = usage.get("prompt_tokens")
                        result["completion_tokens"] = usage.get("completion_tokens")
                except json.JSONDecodeError:
                    pass
            result["content"] = "".join(parts)
            result["tool_calls"] = list(tc_raw.values())
        else:
            data = await resp.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            result["content"] = msg.get("content", "") or ""
            result["finish_reason"] = choice.get("finish_reason")
            result["has_thinking"] = bool(msg.get("reasoning_content"))
            result["reasoning_content"] = msg.get("reasoning_content", "") or ""
            if msg.get("tool_calls"):
                result["tool_calls"] = [
                    {"id": tc.get("id", ""), "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                    for tc in msg["tool_calls"]
                ]
            usage = data.get("usage", {})
            result["prompt_tokens"] = usage.get("prompt_tokens")
            result["completion_tokens"] = usage.get("completion_tokens")
        return result


# ─── Session runner ──────────────────────────────────────────────────────────

_session_counter = 0

def next_session_id():
    global _session_counter
    _session_counter += 1
    return f"sess-{_session_counter:04d}-{int(time.time()*1000) % 100000}"


async def run_session(user_id: str, persona: dict, tracker: StressTestTracker, end_time: float):
    session_id = next_session_id()
    tracker.total_sessions += 1

    # Pick topic and opener
    topic = random.choice(persona["interests"])
    openers = OPENERS.get(topic, [])
    if not openers:
        topic = random.choice(list(OPENERS.keys()))
        openers = OPENERS[topic]
    opener = random.choice(openers)

    # Session shape
    num_turns = random.randint(*persona["session_length"])
    use_tools = random.random() < persona["tool_affinity"]
    streaming = random.random() < 0.4

    tools = None
    if use_tools:
        n_tools = random.randint(1, min(4, len(ALL_TOOLS)))
        tools = random.sample(ALL_TOOLS, n_tools)

    params = {
        "max_tokens": random.choice([256, 512, 800, 1024, 1500]),
        "temperature": random.choice([0.0, 0.3, 0.5, 0.7, 0.9]),
    }

    mode = "stream" if streaming else "sync"
    tool_info = f", {len(tools)} tools" if tools else ""
    print(f"  >> {user_id} ({persona['name']}) starts {session_id[-10:]}: {topic} ({num_turns}t, {mode}{tool_info})")

    conversation = [{"role": "system", "content": persona["system"]}]

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as http:
        current_message = opener

        for turn_idx in range(num_turns):
            if time.time() > end_time:
                break

            conversation.append({"role": "user", "content": current_message})

            ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            t0 = time.time()

            try:
                result = await make_request(http, conversation, params, tools, streaming)
                elapsed_ms = int((time.time() - t0) * 1000)
                ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                tool_names = [tc["name"] for tc in result["tool_calls"]]

                entry = RequestLog(
                    user_id=user_id, session_id=session_id, persona=persona["name"],
                    scenario_topic=topic, turn=turn_idx + 1,
                    timestamp_start=ts_start, timestamp_end=ts_end,
                    elapsed_ms=elapsed_ms, status=result["status"],
                    streaming=streaming, has_tools=bool(tools),
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    finish_reason=result["finish_reason"],
                    tool_calls_count=len(result["tool_calls"]),
                    tool_names=tool_names,
                    has_thinking=result["has_thinking"],
                    error=result["error"],
                    user_message=current_message[:500],
                    assistant_preview=(result["content"] or "")[:300],
                    messages_count=len(conversation),
                )
                await tracker.log_request(entry)

                ok = "+" if not result["error"] else "X"
                tok = f"{result['completion_tokens'] or '?'}tok"
                tc_str = f" tools=[{','.join(tool_names)}]" if tool_names else ""
                think_str = " +think" if result["has_thinking"] else ""
                print(f"     {ok} {user_id} t{turn_idx+1}/{num_turns}: {elapsed_ms}ms {tok}{tc_str}{think_str}")

                if result["error"]:
                    break

                # Add response to conversation
                had_tool_calls = len(result["tool_calls"]) > 0
                if had_tool_calls:
                    tc_msg = {
                        "role": "assistant",
                        "content": result["content"] if result["content"] else None,
                        "tool_calls": [
                            {
                                "id": tc.get("id", f"call_{turn_idx}_{i}"),
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]}
                            }
                            for i, tc in enumerate(result["tool_calls"])
                        ]
                    }
                    conversation.append(tc_msg)

                    for i, tc in enumerate(result["tool_calls"]):
                        try:
                            args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                        except json.JSONDecodeError:
                            args = {}
                        fake_resp = fake_tool_response(tc["name"], args)
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", f"call_{turn_idx}_{i}"),
                            "content": fake_resp,
                        })
                else:
                    conversation.append({"role": "assistant", "content": result["content"] or "(empty)"})

                # Generate next user message
                if turn_idx < num_turns - 1:
                    response_text = result["content"] or result.get("reasoning_content", "") or ""
                    current_message = generate_followup(persona, response_text, turn_idx + 1, num_turns, had_tool_calls)
                    lo, hi = persona["think_time"]
                    await asyncio.sleep(random.uniform(lo, hi))

            except Exception as e:
                elapsed_ms = int((time.time() - t0) * 1000)
                ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                entry = RequestLog(
                    user_id=user_id, session_id=session_id, persona=persona["name"],
                    scenario_topic=topic, turn=turn_idx + 1,
                    timestamp_start=ts_start, timestamp_end=ts_end,
                    elapsed_ms=elapsed_ms, status=0, streaming=streaming,
                    has_tools=bool(tools), error=str(e)[:200],
                    user_message=current_message[:500], messages_count=len(conversation),
                )
                await tracker.log_request(entry)
                print(f"     X {user_id} t{turn_idx+1}: EXCEPTION {str(e)[:80]}")
                break


async def user_slot(slot: int, tracker: StressTestTracker, end_time: float):
    while time.time() < end_time:
        persona = random.choice(PERSONAS)
        user_id = f"U{slot+1}-{persona['name']}-{tracker.total_sessions+1}"
        tracker.active_users += 1
        try:
            await run_session(user_id, persona, tracker, end_time)
        finally:
            tracker.active_users -= 1
        if time.time() < end_time:
            await asyncio.sleep(random.uniform(1.0, 6.0))


async def status_printer(tracker: StressTestTracker, end_time: float):
    while time.time() < end_time:
        await asyncio.sleep(15)
        tracker.print_status()


async def main():
    global MAX_USERS, TEST_DURATION, BASE_URL, MODEL

    parser = argparse.ArgumentParser(description="AFM Stress Test — sustained AI-driven concurrent users")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--duration", type=int, default=1800, help="Duration in seconds (default: 1800 = 30 min)")
    parser.add_argument("--users", type=int, default=7, help="Concurrent user slots")
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    MAX_USERS = args.users
    TEST_DURATION = args.duration
    BASE_URL = f"http://127.0.0.1:{args.port}"
    MODEL = args.model

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"test-reports/stress-test-{timestamp}"
    os.makedirs(report_dir, exist_ok=True)

    dur_min = TEST_DURATION / 60
    n_openers = sum(len(v) for v in OPENERS.values())
    print(f"")
    print(f"  AFM Stress Test - {MAX_USERS} sustained AI-driven users")
    print(f"  {'='*50}")
    print(f"  Server:    {BASE_URL}")
    print(f"  Model:     {MODEL}")
    print(f"  Duration:  {dur_min:.0f} min ({TEST_DURATION}s)")
    print(f"  Personas:  {len(PERSONAS)} unique roles")
    print(f"  Topics:    {len(OPENERS)} domains, {n_openers} opening prompts")
    print(f"  Reports:   {report_dir}")
    print(f"  {'='*50}")
    print()

    # Verify server
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{BASE_URL}/v1/models") as r:
                if r.status != 200:
                    print(f"ERROR: Server returned {r.status}")
                    return
                data = await r.json()
                print(f"  Server OK - model: {data.get('data', [{}])[0].get('id', '?')}")
    except Exception as e:
        print(f"ERROR: Cannot reach server: {e}")
        return

    tracker = StressTestTracker(report_dir)
    end_time = time.time() + TEST_DURATION

    print(f"\n  Launching {MAX_USERS} user slots...\n")

    tasks = [asyncio.create_task(user_slot(i, tracker, end_time)) for i in range(MAX_USERS)]
    tasks.append(asyncio.create_task(status_printer(tracker, end_time)))

    await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\n{'='*60}")
    print(f"Test complete. Generating reports...\n")

    jsonl_path = tracker.save_jsonl()
    report_path = tracker.save_report()

    print(f"  JSONL log:    {jsonl_path}")
    print(f"  Report:       {report_path}")
    print(f"")
    print(f"  Total requests:  {tracker.total_requests}")
    print(f"  Total errors:    {tracker.total_errors}")
    print(f"  Total sessions:  {tracker.total_sessions}")
    print(f"  Duration:        {tracker.elapsed():.1f}s")
    print(f"  Avg RPS:         {tracker.total_requests/tracker.elapsed():.2f}")
    print(f"  Thinking:        {sum(1 for e in tracker.logs if e.has_thinking)} responses")
    print(f"  Tool calls:      {sum(e.tool_calls_count for e in tracker.logs)} total")


if __name__ == "__main__":
    asyncio.run(main())
