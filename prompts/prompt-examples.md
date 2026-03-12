# Coding Prompt Examples

Test prompts for evaluating local LLM coding performance with `afm mlx`.

## Python — Async, Systems

```
Implement a Python async web crawler that respects robots.txt, handles rate limiting with exponential backoff, extracts all links from pages, and stores results in an SQLite database. Include proper error handling and graceful shutdown.
```

## Python — Data Structures, Threading

```
Write a Python class that implements a thread-safe LRU cache with TTL expiration, supporting get, put, and delete operations. Include type hints and a usage example.
```

## Rust — Concurrency, Lock-Free

```
Implement a lock-free concurrent hash map in Rust using atomic operations. Support insert, get, and delete with safe memory reclamation.
```

## Go — Networking, Load Balancing

```
Write a Go TCP reverse proxy that supports multiple backends with health checking, weighted round-robin load balancing, and graceful connection draining.
```

## TypeScript — Type System, Generics

```
Implement a type-safe event emitter in TypeScript with full generic inference, wildcard listeners, once-only handlers, and async event support.
```

## C — Low-Level, Memory Management

```
Implement a memory allocator in C using a segregated free list with coalescing. Support malloc, free, and realloc with alignment guarantees.
```

## Java — OOP, Concurrency

```
Write a Java work-stealing thread pool executor with task dependencies, priority queues, and deadlock detection. Include proper shutdown semantics.
```

## Swift — Actors, Structured Concurrency

```
Implement a Swift actor-based rate limiter using token bucket algorithm with sliding window support, distributed across multiple actor instances using async sequences.
```

## COBOL — Financial Batch Processing

```
Implement a COBOL batch program for end-of-day loan payment processing. It should read a sequential transaction file of payments, validate each against a master loan file (VSAM KSDS), apply payments to interest first then principal per standard amortization rules, handle overpayments and underpayments, update the master file balances, write an exception report for rejected transactions, and produce a GL summary file with debit/credit entries for the general ledger posting. Include proper file status checking, paragraph structure, and working storage for all accumulators.
```

## Math — Rigorous Notation (Non-Coding)

```
Explain calculus concepts from limits through multivariable calculus with rigorous mathematical notation.
```
