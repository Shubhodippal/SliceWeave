"""Calibration/eval prompt bank across task categories (§48, §52).

§48: diverse categories — do NOT calibrate on coding only.
Sample-size discipline (§97): ~400 hand-written prompts with per-language
depth (C, C++, Rust, Go, Java, Python, JS/TS, SQL, shell) and real-life
practical categories (debugging, concurrency, security, testing, performance,
code review, API/system design, DevOps). Pilots may subset via --prompts N or
sample_prompts(); decision-gate claims (§14) use the full bank.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# CODING — per-language depth (real-life practical phrasing)
# ---------------------------------------------------------------------------
CODING_PYTHON = [
    "Write a Python function to reverse a linked list in place.",
    "Implement an LRU cache in Python with O(1) get and put.",
    "Implement a thread-safe bounded queue in Python using condition variables.",
    "Why does mutating a list while iterating over it skip elements? Show the fix.",
    "How do I make an HTTP call retry with exponential backoff in Python?",
    "What's the difference between __new__ and __init__ in Python, with a singleton example?",
    "Explain Python's GIL: when do threads help and when do processes win?",
    "Write a generator that yields lines from a 10 GB log file without loading it.",
    "How do I profile a slow Python function with cProfile and read the output?",
    "Convert this class-based code to use dataclasses with validation.",
    "Write a context manager that times a block and logs slow executions above 100ms.",
    "How do I safely share a numpy array between multiprocessing workers?",
    "Implement a retry decorator that swallows only transient HTTP errors.",
    "What is the fastest way to dedupe a list of 10M dicts by key in Python?",
    "Why does my asyncio task never run? Explain event loop starvation.",
    "Write a script that watches a directory and processes new CSV files idempotently.",
    "Explain pickling pitfalls with lambdas and nested functions.",
    "Write a rate limiter using a token bucket, thread-safe, with tests.",
    "My pandas merge exploded memory. Walk through diagnosing dtype blowup.",
    "Why does sys.setrecursionlimit not save my deep recursion? Alternatives?",
]
CODING_C = [
    "Write a C function that reverses a string in place without extra memory.",
    "Why does my C program segfault when freeing a pointer twice?",
    "Explain the difference between malloc, calloc, and realloc failure modes.",
    "Write a C function to detect a cycle in a singly linked list.",
    "How do dangling pointers arise after realloc? Show a safe pattern.",
    "Implement a bounded strcpy-style copy and explain the overflow risk.",
    "What is strict aliasing and when does it break type-punning code?",
    "Explain why returning a pointer to a stack local is undefined behavior.",
    "Write a bit-twiddling function to count set bits and explain the trick.",
    "How does alignment affect casting a char buffer to a struct pointer?",
    "Implement a fixed-size ring buffer in C with head/tail indices.",
    "Explain volatile: when is it needed and when is it not a synchronization tool?",
    "Write signal-safe code: what can you do inside a signal handler?",
    "How do I debug a heap corruption that crashes far from the corrupting write?",
    "What is the difference between undefined, unspecified, and implementation-defined?",
    "Explain endianness pitfalls when serializing structs over a network.",
    "How do I use valgrind to find an off-by-one heap read?",
    "Why can sizeof('a') equal 4 in C but 1 in C++?",
    "Write a macro-free min/max for mixed integer types and explain pitfalls.",
    "Explain what happens when a buffer overflow overwrites a return address.",
]
CODING_CPP = [
    "Why does my C++ program segfault when freeing a pointer twice?",
    "Explain move semantics: what does std::move actually do, and not do?",
    "When should I use std::unique_ptr versus std::shared_ptr, with examples?",
    "Why is returning const std::string& from a function dangerous after C++11?",
    "Implement a scoped lock wrapper using RAII for a legacy mutex API.",
    "What is the rule of five and when can I default everything?",
    "Explain vector reallocation and iterator invalidation with a crash example.",
    "How does std::optional improve APIs over sentinel values?",
    "Why can template instantiation explode compile times? Mitigations?",
    "Explain SFINAE versus concepts with a before/after example.",
    "What is a vtable and when does virtual dispatch not happen?",
    "Implement a small thread pool in C++17 with futures for results.",
    "Why is std::vector<bool> weird, and what should I use instead?",
    "Explain dangling references with std::string_view and lifetime rules.",
    "Explain undefined behavior with signed overflow and optimizer assumptions.",
    "When is emplace_back faster than push_back, and when is it identical?",
    "Implement a type-safe variant visitor using std::visit and overload sets.",
    "Why does my lambda capture dangle when stored in a callback registry?",
    "Explain memory ordering for std::atomic with acquire/release in a spinlock.",
    "How do I write a custom allocator-aware container, minimal version?",
]
CODING_JAVA = [
    "Convert this recursive Fibonacci to a memoized iterative version in Java.",
    "Explain ConcurrentHashMap vs synchronized HashMap for a read-heavy cache.",
    "Implement a Producer-Consumer with BlockingQueue and graceful shutdown.",
    "What causes a Java OutOfMemoryError: metaspace, and how do I diagnose it?",
    "Explain equals/hashCode contracts; what breaks a HashSet lookup?",
    "Write a stream pipeline that groups orders by month and sums revenue.",
    "Why is double-checked locking broken without volatile?",
    "Explain GC pause types in the JVM and how G1 mitigates them.",
    "How do virtual threads change blocking-IO server design in Java 21?",
    "Implement a custom AutoCloseable resource with correct exception handling.",
    "Why are records better than Lombok data classes for DTOs?",
    "Explain classloader leaks in application servers with static caches.",
    "What is the difference between checked and unchecked exceptions by design intent?",
    "Implement a bounded rate limiter with Semaphore and a scheduled refill.",
    "Why can floating point money calculations fail? Show BigDecimal usage.",
    "Explain immutability benefits with a thread-safe value object example.",
    "How do I profile a Java service to find lock contention hotspots?",
    "Explain the module system's strong encapsulation vs classpath JARs.",
    "What does String.intern do and when does it help parsing hot paths?",
    "Why can an unbounded thread pool exhaust memory under burst load?",
]
CODING_RUST = [
    "Implement Dijkstra in Rust using a binary heap.",
    "What does the Rust borrow checker complain about when I mutate a Vec while iterating?",
    "Explain ownership move semantics when passing a String into a function.",
    "When do I choose Rc<RefCell<T>> versus Arc<Mutex<T>>?",
    "Write a Rust iterator adapter chain that parses CSV rows into structs.",
    "What is a lifetime 'a and how do I annotate a struct holding a reference?",
    "Why does the compiler reject returning a reference to a local? Explain elision.",
    "Implement a trait with default methods and explain dynamic vs static dispatch.",
    "Explain Result propagation with the ? operator and custom error types.",
    "What does Send + Sync mean for spawning threads with std::thread?",
    "How do I safely share mutable state across await points in async Rust?",
    "Explain Box, Rc, Arc, and when each is the right tool.",
    "Write a Rust function using pattern matching on an enum state machine.",
    "Why does closure capture move or borrow depending on usage?",
    "Implement a linked list in Rust and discuss why it is famously hard.",
    "Explain unsafe blocks: what responsibilities does the author take on?",
    "What is zero-cost abstraction with an iterator optimization example?",
    "Explain panics versus Result for recoverable versus unrecoverable errors.",
    "How does Cargo handle feature flags and conditional compilation?",
    "What does the borrow checker guarantee about data races in safe Rust?",
]
CODING_GO = [
    "Explain Go goroutine leaks from unbuffered channels and how to fix them.",
    "Why does Go prefer composition over inheritance? Show embedding.",
    "Implement a worker pool in Go with graceful shutdown via context.",
    "What is the difference between make and new in Go?",
    "Explain Go's panic/recover philosophy versus exceptions.",
    "How does the sync.WaitGroup work and what does misuse look like?",
    "Write a concurrent web scraper in Go with a bounded semaphore channel.",
    "Why are Go interfaces satisfied implicitly, and what does that enable?",
    "Explain the nil interface gotcha with a typed nil pointer.",
    "How does Go's scheduler multiplex goroutines onto OS threads?",
    "Implement a rate-limited HTTP client using time.Ticker and context.",
    "Explain slices versus arrays: what does append do under the hood?",
    "How do I write table-driven tests in Go with subtests?",
    "Explain defer ordering and evaluation of arguments at defer time.",
    "Implement a JSON API handler with proper error wrapping in Go.",
    "Explain context cancellation propagation through a call tree.",
    "How do modules, versioning, and the go.sum file work?",
    "What does go vet catch that compilers do not? Examples.",
    "Why is sharing memory by communicating preferred over mutexes here?",
    "What is the zero value and why does Go avoid uninitialized variables?",
]
CODING_JS_TS = [
    "Refactor this callback pyramid into async/await JavaScript.",
    "Why is my React useEffect firing twice in development?",
    "Explain the event loop: microtasks versus macrotasks with Promise ordering.",
    "What does 'this' refer to in arrow versus regular functions?",
    "Write a TypeScript type that extracts the return type of a generic function.",
    "Explain closure-based memoization and its memory tradeoffs in JS.",
    "Implement a debounce util in TypeScript with cancel and flush.",
    "Explain Promise.all vs allSettled vs any with failure scenarios.",
    "How do I avoid prop drilling with context without re-render storms?",
    "Write a typed event emitter in TypeScript with on/off/emit.",
    "Why can mutation of state break memoized components? Immutability pattern.",
    "Implement retries with exponential backoff for fetch in the browser.",
    "Explain CORS preflight: what triggers OPTIONS and how to fix failures.",
    "Write a Node.js stream pipeline that transforms a large CSV safely.",
    "Explain the module systems: ESM vs CommonJS interop pitfalls.",
    "How does garbage collection handle detached DOM nodes causing leaks?",
    "Implement a typed API client wrapper with discriminated union results.",
    "Explain structural typing in TypeScript with a surprising acceptance example.",
    "Why does typeof null return 'object'? Practical guard patterns.",
    "Explain shallow vs deep reactivity in frontend state libraries.",
]
CODING_SQL = [
    "Explain the difference between a SQL inner join and left join.",
    "Write a PostgreSQL query to find duplicate emails using a window function.",
    "Write a SQL migration to add a nullable column with a backfill strategy.",
    "Explain when a composite index helps and when it does nothing.",
    "Write a recursive CTE to flatten an org chart with cycle protection.",
    "Why does OR sometimes prevent index usage? Rewrite with UNION.",
    "Explain isolation levels: what anomaly does READ COMMITTED still allow?",
    "Write a query to find the top 3 products per category using row_number.",
    "How do I safely change a column type on a large table with zero downtime?",
    "Explain N+1 queries in an ORM and fix with eager loading.",
    "Write an upsert that handles concurrent inserts without deadlocks.",
    "Explain MVCC and how vacuum reclaims space in PostgreSQL.",
    "Write a query detecting gaps in a sequence of daily events.",
    "Explain EXPLAIN plans: what makes a seq scan acceptable?",
    "Write a query to pivot monthly revenue into columns.",
    "How do partial indexes speed up soft-delete filtered queries?",
    "Explain locking from FK checks and how to avoid deadlocks on delete.",
    "Why is SELECT * bad for wide tables in production code?",
    "Explain the tradeoffs of denormalization for a read-heavy dashboard.",
    "What does ON DELETE CASCADE do and when is it dangerous?",
]
CODING_SHELL = [
    "Write a bash one-liner to find the 10 largest files under /var recursively.",
    "Explain set -euo pipefail and what breaks without it.",
    "Write a script that retries a flaky curl with backoff and logs failures.",
    "How do I safely handle filenames with spaces in a for loop?",
    "Explain process substitution: diff two command outputs directly.",
    "Write an awk one-liner to sum the second column grouped by the first.",
    "What is the difference between $@ and $* inside quotes?",
    "Write a trap handler that cleans temp files on SIGINT.",
    "How do I atomically replace a config file from a deploy script?",
    "Explain exit codes and how to make a script pipeline-fail on inner failure.",
    "Write a jq pipeline to extract and count error types from JSON logs.",
    "Explain xargs quoting pitfalls with -I and spaces.",
    "Write a portable shell function to check if a TCP port is open.",
    "Explain cron vs systemd timers for a job needing env secrets.",
    "Write a script to diff two directory trees by content hash.",
    "How do I tail a log and alert on a regex within the last N lines?",
    "Write a systemd unit with restart policy and a health check timer.",
    "How do I limit a command's memory with ulimit and verify it?",
    "Explain what happens when a shell variable is referenced while unset.",
    "Write a here-document that safely embeds a SQL script with variables.",
    "How do I speed up a slow find command over millions of files?",
]
CODING_ALGORITHMS = [
    "Implement binary search on a rotated sorted array; state the invariants.",
    "Implement an LRU cache in O(1) using a hashmap plus doubly linked list.",
    "Explain union-find with path compression and rank, then solve accounts merge.",
    "Write topological sort with cycle detection for a build system.",
    "Implement a segment tree for range sum with point update.",
    "Explain Dijkstra versus Bellman-Ford and when negative edges matter.",
    "Write a sliding window solution for longest substring without repeats.",
    "Implement a trie for autocomplete with prefix counts.",
    "Explain dynamic programming for edit distance with the recurrence.",
    "Write KMP failure function and explain why naive search is O(nm).",
    "Implement quickselect for median and analyze expected time.",
    "Explain monotonic stack with the next-greater-element problem.",
    "Implement Floyd's cycle detection and prove the meeting point math.",
    "Write a heap-based solution for merging k sorted lists.",
    "Implement Kruskal's MST with a disjoint set and compare with Prim's.",
    "Explain the two-pointer technique for pair-sum in a sorted array.",
    "Write a matrix rotation in place for square matrices.",
    "Implement interval merging after sorting, with edge cases.",
    "Explain hashing collisions and open addressing load factor tradeoffs.",
    "Write a bitset solution for subset-sum with a memory analysis.",
]
CODING = (CODING_PYTHON + CODING_C + CODING_CPP + CODING_JAVA + CODING_RUST +
          CODING_GO + CODING_JS_TS + CODING_SQL + CODING_SHELL + CODING_ALGORITHMS)

# ---------------------------------------------------------------------------
# PRACTICAL ENGINEERING — real-life day-to-day scenarios
# ---------------------------------------------------------------------------
DEBUGGING = [
    "Debug this FastAPI JWT authentication bug: the token expires before refresh.",
    "Debug why my Docker container can't resolve DNS inside the build stage.",
    "Why does float arithmetic make 0.1 + 0.2 != 0.3 and how do languages mitigate it?",
    "My service works locally but 503s in staging with 4x traffic. Systematic debug plan?",
    "Why does my test pass alone but fail in the full suite? Ordering or state?",
    "Intermittent timeout at p99 only. What do you check first and why?",
    "My regex matches greedily across lines. Explain and fix lazily.",
    "A memory leak appears only under HTTPS. What instrumentation do you add?",
    "Why does my retry make the outage worse? Explain retry storms and jitter.",
    "The API returns stale data after deploy. Which caches invalidate and when?",
    "Bug reproduces only on ARM. What classes of bugs behave differently there?",
    "My binary works with -O0 but crashes with -O2. Likely culprits and how to bisect?",
    "Explain reading a flame graph: where do I start with a 3x regression?",
    "My docker build broke after a base image update. How do you pin and diff?",
    "Why do two identical requests return different orderings? Non-determinism sources.",
    "Kafka consumer lags then recovers nightly. What does that pattern indicate?",
    "A query jumped from 5ms to 4s after an insert-heavy day. Diagnose step by step.",
    "Why does my UI flash unstyled content only on cold start?",
    "Production-only NullPointerException: what must logs carry to debug async flows?",
    "My cron ran twice at a DST boundary. How do schedules handle time ambiguity?",
]
CONCURRENCY = [
    "Explain memory ordering: why can a lock-free counter miss increments?",
    "Explain a deadlock with lock ordering and fix a transfer-money example.",
    "What is a data race versus a race condition, with a Java and a C++ example?",
    "Explain compare-and-swap and implement a lock-free stack sketch.",
    "Why does double-checked locking need volatile in Java or atomics in C++?",
    "Explain goroutine leaks with a sender that never closes its channel.",
    "What is the thundering herd on a broadcast wake and how do you tame it?",
    "Implement a reader-writer lock policy for a hot config cache. Pitfalls?",
    "Explain ABA in lock-free algorithms and a tagged-pointer mitigation.",
    "Why can two threads see different sizes of the same concurrent map?",
    "Explain false sharing on cache lines with a counter-array benchmark.",
    "What does 'happens-before' mean and how do synchronization edges establish it?",
    "Design a bounded pipeline with backpressure across 3 stages.",
    "Why are idempotency keys needed for at-least-once queues? Show a duplicate.",
    "Explain cancellation: cooperative checks vs thread interruption semantics.",
    "How would you test for races systematically? Sanitizers, stress tests, model checkers.",
    "Explain optimistic vs pessimistic concurrency for a booking seat flow.",
    "Why does one slow consumer stall a shared queue? Fairness designs.",
    "Implement a single-flight guard that prevents cache stampede on miss.",
    "What goes wrong if a signal handler allocates memory? Async-signal-safety.",
]
SECURITY = [
    "Explain SQL injection with a vulnerable query and the parameterized fix.",
    "Why is storing passwords with SHA-256 alone wrong? Describe bcrypt/argon2 use.",
    "Explain CSRF and how SameSite plus tokens mitigate it.",
    "What is a JWT's weakness when it cannot be revoked? Refresh token design.",
    "Explain XSS types and a Content-Security-Policy that blocks inline scripts.",
    "Why is client-side validation never sufficient? Server-side trust boundaries.",
    "Explain a timing attack on string comparison and constant-time equality.",
    "What does TLS 1.3 change versus TLS 1.2 in handshake security?",
    "Explain IDOR with an API example and an object-level authz fix.",
    "Why should secrets never live in env files committed to git? Rotation plan.",
    "Explain SSRF with a webhook URL feature and egress filtering mitigation.",
    "What is a deserialization gadget chain and why is it dangerous?",
    "Explain least privilege with a CI service account example.",
    "How does a nonce prevent replay in authentication protocols?",
    "Explain secure random versus math random with a token generation bug.",
    "What is clickjacking and which headers prevent framing?",
    "Explain rate limiting and lockout tradeoffs in login endpoints.",
    "Why is logging sensitive data a compliance problem? Redaction strategies.",
    "Explain dependency confusion with a private package name example.",
    "What is privilege escalation via mass assignment and the fix in a serializer?",
]
TESTING = [
    "Design a test pyramid for a REST API: what lives at each layer?",
    "Write unit tests for a rate limiter including boundary and concurrency cases.",
    "What makes a test flaky and how do you quarantine and fix flakiness?",
    "Explain AAA structure and why one assertion per concept improves failures.",
    "How do you test async code deterministically? Fake clocks and schedulers.",
    "Write property-based tests for a JSON parser with shrinking examples.",
    "Explain test doubles: dummy, fake, stub, spy, mock with a checkout example.",
    "What is contract testing between microservices and what does it catch?",
    "How do you test migration scripts against production-like data safely?",
    "Write a golden-master test for a report generator refactoring.",
    "Explain coverage's limits: what does 100% coverage fail to catch?",
    "How do you fuzz an HTTP API for 500s and unhandled input classes?",
    "Design integration tests with Testcontainers for a Postgres-backed service.",
    "Explain snapshot testing pitfalls for UI components and when it helps.",
    "What is mutation testing and what does it reveal about assertion strength?",
    "How do you load test to find the knee, not just the max, of throughput?",
    "Write E2E tests that survive UI refactors: selector strategy.",
    "Explain chaos testing for a multi-AZ service: first three experiments.",
    "How do you test timezones in a scheduling feature across DST boundaries?",
    "What is a test fixture lifecycle per suite vs per test and why it matters.",
]
PERFORMANCE = [
    "A p99 jumped 10x after adding auth middleware. Profile plan for a Node service.",
    "Explain B-tree page cache behavior when an index outgrows RAM.",
    "Why is an O(n log n) sort slower than an O(n^2) one on 50 items? Constant factors.",
    "Explain branch prediction with a sorted-vs-unsorted loop benchmark.",
    "How does batch size affect throughput vs latency in a writer? Model the tradeoff.",
    "Explain why sequential IO beats random IO on NVMe and how to exploit it.",
    "A cache hit ratio is 99% but latency is high. What else dominates?",
    "Explain gzip level tradeoffs for a 10x traffic surge on static assets.",
    "How do you decide between adding indexes versus denormalizing a summary table?",
    "Explain connection pool sizing math for 50 rps and 40ms queries.",
    "Why does an ORM lazy-load cause 1000 queries on a list page? Fix patterns.",
    "What is tail latency amplification across 5 dependent services?",
    "Explain JIT warmup for a Java batch job and how to keep profiles warm.",
    "How would you reduce cold start time of a serverless function 5x?",
    "Explain the memory-vs-speed tradeoff of bloom filters in a key-value store.",
    "Why is per-request JSON parsing CPU-heavy at 10k rps? Alternatives.",
    "Explain NUMA effects for a memory-bound multithreaded benchmark.",
    "What makes a query parallelize poorly? Skew and barrier examples.",
    "Explain allocation churn in Go and GC tuning with pprof profiles.",
    "How do you benchmark honestly: warmup, variance, pinning, and reporting.",
]
CODE_REVIEW = [
    "Review this diff adding a global mutable config: what are the risks and fixes?",
    "A PR swallows exceptions with bare except. What feedback do you leave and why?",
    "Review a PR that introduces an N+1 ORM query behind a loop. Suggest fixes.",
    "How do you review for security in an auth PR? Checklist with concrete checks.",
    "A teammate's PR works but has no tests. Constructive review comment framework.",
    "Review a retry loop without backoff or jitter in a client library.",
    "What do you look for in a migration PR that alters a hot table?",
    "Review exposing an internal endpoint without authz. Severity and remediation.",
    "A PR mixes refactoring with feature changes. How do you ask to split and why?",
    "Review inconsistent error handling across a new module. Suggest a policy.",
    "How do you flag premature abstraction kindly, with a concrete simpler design?",
    "Review a PR adding a dependency: what supply-chain questions do you ask?",
    "A PR renames 400 files. How do you review semantics vs noise effectively?",
    "Review a caching PR for invalidation correctness and stampede risk.",
    "What comments belong in code vs docs vs issue tracker? With examples.",
    "Review a public API change for backward compatibility and versioning strategy.",
    "How do you review logging changes for PII and volume cost?",
    "A PR adds a benchmark. What makes benchmark code trustworthy in review?",
    "Review concurrency additions: locks, atomics, and task lifetime checks.",
    "Give feedback style rules: small nits vs blockers, tone, and blocking criteria.",
]
API_SYSTEM_DESIGN = [
    "Design a URL shortener: id strategy, redirects, analytics, and cache layout.",
    "Design a rate limiter for a multi-tenant API: algorithms and storage.",
    "Design a notification service: fanout, dedupe, and delivery guarantees.",
    "Design pagination for an infinite feed: cursor scheme and stability.",
    "Design a webhook delivery system with retries, signatures, and ordering.",
    "Design an image resizing pipeline: async jobs, CDN, and cost control.",
    "Design a feature flag system: evaluation, targeting, and consistency.",
    "Design idempotent payment APIs: keys, retries, and reconciliation.",
    "Design a search autosuggest service: indexing and freshness budget.",
    "Design a multi-region read model: replication lag and conflict policy.",
    "Design a file upload service: direct-to-storage, presigning, virus scan.",
    "Design an audit log: immutability, retention, and query patterns.",
    "Design a job scheduler: cron semantics, sharding, and missed-run policy.",
    "Design a real-time collaboration service: OT vs CRDT tradeoffs.",
    "Design a rate-limited public API SDK: retries, backoff, and circuit breaking.",
    "Design a metrics pipeline: cardinality control and downsampling.",
    "Design a session store: cookie vs server-side and revocation needs.",
    "Design a config service: versioning, rollback, and client caching.",
    "Design an offline-first mobile sync: conflict resolution strategies.",
    "Design an email ingestion pipeline: parsing, spam scoring, attachments.",
]
DEVOPS_CLOUD = [
    "Debug why my Kubernetes pod is CrashLoopBackOff: ordered triage steps.",
    "Explain blue-green versus canary deployments and rollback triggers.",
    "Write a GitHub Actions workflow that caches dependencies and runs matrix tests.",
    "What belongs in a Dockerfile for a Python service to stay small and cacheable?",
    "Explain liveness vs readiness probes with a slow-init Java app.",
    "How do you set SLOs and error budgets for a checkout API?",
    "Explain structured logging with correlation IDs across async boundaries.",
    "Design a database migration rollback plan for a 3-step schema change.",
    "What is a postmortem template that avoids blame and yields actions?",
    "Explain autoscaling signals: CPU vs queue depth vs latency-based scaling.",
    "How do you safely rotate database credentials with zero downtime?",
    "Explain IaC drift: detection and reconciliation in Terraform.",
    "What is a runbook and what makes one usable during an incident?",
    "Explain TLS termination choices at LB vs app with cert rotation.",
    "How do you capacity plan a cache: working set estimation and eviction checks?",
    "Explain graceful shutdown: signal handling and connection draining.",
    "What is cost observability: tagging, unit economics, and alerts on spend?",
    "Explain feature flags in a deploy pipeline: kill switches and cleanup.",
    "How do you version APIs deployed independently across teams?",
    "Explain incident severity levels and on-call escalation design.",
]

CATEGORIES = {
    "coding": CODING,
    "debugging": DEBUGGING,
    "concurrency": CONCURRENCY,
    "security": SECURITY,
    "testing": TESTING,
    "performance": PERFORMANCE,
    "code_review": CODE_REVIEW,
    "api_system_design": API_SYSTEM_DESIGN,
    "devops_cloud": DEVOPS_CLOUD,
    "math": [
        "Solve: x^2 - 5x + 6 = 0.",
        "Prove that the square root of 2 is irrational.",
        "Compute the derivative of sin(x)*e^x.",
        "What is the sum of the first 100 positive integers?",
        "Explain dynamic programming using the coin change problem.",
        "Find the eigenvalues of the matrix [[2,1],[1,2]].",
        "Integrate x^2 * ln(x) dx by parts.",
        "What is the probability of getting exactly 3 heads in 5 fair coin flips?",
        "Prove by induction that 1+2+...+n = n(n+1)/2.",
        "Solve the recurrence T(n) = 2T(n/2) + n using the master theorem.",
        "What is the determinant of a 3x3 matrix with rows (1,2,3),(4,5,6),(7,8,10)?",
        "Find the equation of the tangent line to y = x^3 at x = 2.",
        "How many ways can you arrange the letters of MISSISSIPPI?",
        "Explain why e^(i*pi) + 1 = 0.",
        "Compute the limit of (1 + 1/n)^n as n goes to infinity.",
        "A bag has 4 red and 6 blue balls; draw 2 without replacement. P(both red)?",
        "Find the area under y = sin(x) from 0 to pi.",
        "What is 17 * 23 using the difference of squares trick?",
        "Prove that there are infinitely many prime numbers.",
        "Solve the system: 2x + 3y = 12, x - y = 1.",
        "What is the expected value of rolling two six-sided dice?",
        "Explain the difference between a martingale and a random walk.",
        "Explain Bayes' theorem with a medical screening example.",
        "What is the pigeonhole principle? Give two nontrivial applications.",
        "Solve |x - 3| + |x + 1| = 6 and explain the case split.",
    ],
    "reasoning": [
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
        "A farmer has 17 sheep and all but 9 run away. How many remain?",
        "Explain why a red herring fallacy weakens an argument.",
        "Order these: losing keys, finding keys, locking door, leaving house.",
        "Which is heavier: a kilogram of steel or two pounds of feathers?",
        "Alice is faster than Bob, Bob is faster than Carol. Is Alice slower than Carol?",
        "If it rains, the street gets wet. The street is wet. Did it necessarily rain?",
        "A bat and a ball cost 1.10 together; the bat costs 1.00 more than the ball. Ball price?",
        "Explain the difference between correlation and causation with an example.",
        "Five people shake hands once with each other. How many handshakes?",
        "You have a 3L and a 5L jug; measure exactly 4 liters.",
        "Is the statement 'this sentence is false' resolvable? Explain the paradox.",
        "Tom is taller than 90% of his class. Can you conclude his exact percentile rank?",
        "If I flip a fair coin 10 times and get 10 heads, what is the probability the next flip is heads?",
        "Three switches, one lamp upstairs; you may go up once. How do you identify the switch?",
        "Explain Occam's razor and when it misleads.",
        "A train leaves at 60 km/h; a car leaves 1 hour later at 90 km/h. When does it catch up?",
        "Why is arguing from a small sample unreliable? Give a concrete example.",
        "Sort these by logical dependency: build foundation, frame walls, inspect permits, paint walls.",
        "If everyone reduces consumption by 10%, does total consumption drop by 10%? Explain rebound effects.",
        "Two doors, one truth-teller and one liar; ask one question to find the exit.",
        "A cube is painted and cut into 27 unit cubes. How many small cubes have exactly two painted faces?",
        "Why can't you conclude causation from an A/B test with a confounder? Example.",
        "Explain base rates with a rare disease testing puzzle.",
    ],
    "general_knowledge": [
        "What is the capital of Australia?",
        "Describe the causes of World War I.",
        "Explain photosynthesis in simple terms.",
        "Who wrote Pride and Prejudice?",
        "What is the boiling point of water at sea level?",
        "Name the planets of the solar system in order from the sun.",
        "What is the difference between weather and climate?",
        "Explain what the Internet's DNS system does.",
        "When did the Berlin Wall fall, and what followed?",
        "What are the three states of matter and their phase transitions?",
        "Who was Ada Lovelace and why is she significant in computing?",
        "Explain why the sky is blue during the day but red at sunset.",
        "What does the immune system do when it encounters a virus for the first time?",
        "List the branches of the US government and their primary functions.",
        "What causes tides on Earth?",
        "Explain the difference between AC and DC current.",
        "What is the significance of the Rosetta Stone?",
        "How do vaccines produce immunity?",
        "What is the largest ocean and roughly how much of Earth's surface does it cover?",
        "Explain how a refrigerator moves heat.",
        "What is compound interest and why does it grow exponentially?",
        "Describe the water cycle in four steps.",
        "Why do seasons occur, and why are they opposite in the two hemispheres?",
        "What is GDP and what does it fail to measure?",
        "Explain how noise-cancelling headphones work.",
    ],
    "writing": [
        "Write a haiku about autumn rain.",
        "Draft a polite email declining a meeting invitation.",
        "Compose a short poem about the ocean.",
        "Write a product description for a wooden coffee mug.",
        "Summarize the plot of Hamlet in three sentences.",
        "Rewrite this sentence to be clearer: 'The utilization of the aforementioned methodology was undertaken by the team.'",
        "Write a short story opening set in a lighthouse during a storm.",
        "Draft a release note announcing a breaking API change.",
        "Explain recursion to a 10-year-old in one paragraph.",
        "Write a witty one-line slogan for a keyboard-cleaning startup.",
        "Compose an apology letter for shipping a broken update, keeping customer trust.",
        "Describe a medieval market square using all five senses.",
        "Write a LinkedIn post about lessons from your first engineering failure.",
        "Turn these notes into a professional agenda: budget, hiring, roadmap, AOB.",
        "Write dialogue between a skeptical detective and an over-eager witness.",
        "Draft a README intro for a small open-source CLI tool.",
        "Write a toast celebrating a retiring codebase.",
        "Condense a 3-paragraph policy into 2 bullet points and 1 action item.",
        "Write a nursery rhyme that secretly teaches binary counting.",
        "Craft a rejection email that is kind but leaves no ambiguity.",
        "Write a user-facing error message for a failed payment, with next steps.",
        "Draft a changelog entry users will actually read, for a speed improvement.",
        "Write a 3-tweet thread explaining database indexes for founders.",
        "Compose a conference talk abstract on caching, under 120 words.",
        "Rewrite legalistic terms into plain English: a data retention clause.",
    ],
    "long_context": [
        "Summarize the following multi-paragraph document in three bullet points: "
        "Chapter 1 covers data pipelines and their failure modes, including schema "
        "drift, late-arriving events, and partition skew. Chapter 2 covers "
        "monitoring, defining SLOs for freshness and completeness, and alerting "
        "strategy. Chapter 3 covers backfills and idempotent job design.",
        "Given the meeting transcript about migrating services to a message queue, "
        "list every decision made and its stated rationale.",
        "From the API reference below, write the exact call to fetch a paginated "
        "list of datasets with an async client and retries enabled.",
        "Here is a rental agreement; state the notice period and the deposit return "
        "conditions in plain English.",
        "Read this changelog and tell me which version introduced the memory "
        "regression fix and what workaround existed before it.",
        "Given a protein FASTA header block, extract the organism, accession, and "
        "translation table in JSON.",
        "From this multi-email thread, identify the single decision owner and the "
        "current blocker.",
        "This contract clause is ambiguous; explain both readings and which favors "
        "the subcontractor.",
        "Summarize the following stack trace and suggested fix in two sentences.",
        "Given a 40-row CSV snippet of server logs, name the hour bucket with the "
        "most 5xx responses.",
        "Here are three product requirement drafts; list contradictions between "
        "them and propose a resolution.",
        "From this lecture transcript excerpt, write 4 flashcards with question "
        "and answer.",
        "This design doc proposes event sourcing; list operational risks it creates.",
        "From the error budget policy below, decide if a risky deploy is allowed today.",
        "Given a cron spec and holidays list, find dates the job should skip.",
        "From two candidate architectures below, build a comparison table with costs.",
        "Extract all API breaking changes from these 6 release notes into a table.",
        "Given a log excerpt spanning a failover, reconstruct the incident timeline.",
    ],
    "instruction_following": [
        "Rewrite this sentence in exactly five words: the system failed unexpectedly today.",
        "Answer with only the JSON key names, comma separated, no prose: "
        '{"user": {"name": "Ada", "roles": ["admin"]}}',
        "List three steps for rotating an API key. Use numbered markdown list format.",
        "Respond with a single word: what HTTP status means 'not found'?",
        "Translate 'deploy on Friday afternoon' into three different tones: formal, casual, humorous.",
        "Give the answer as a Python dict with keys a and b: 7+3 and 7*3.",
        "Write a commit message in conventional-commit style for adding a retry to the auth client.",
        "Produce exactly three bullet points about cache invalidation, each under 12 words.",
        "Output only the second paragraph of your answer, in all caps.",
        "End every sentence of your reply with the word 'confirmed'.",
        "Convert this list to a YAML array: apples, bananas, cherries.",
        "Answer with a table: column names lang, year; rows Python 1991, Rust 2010.",
        "Begin your response with 'Step 1:' and use exactly two steps total.",
        "Rewrite the sentence without any form of the verb 'to be': 'The cache is refreshed by the worker.'",
        "Give a one-sentence answer containing the number 42 and the word 'namespace'.",
        "Respond in JSON: {\"ok\": true} with an extra field for the current topic.",
        "Write the word 'banana' backwards, then give its length as a number.",
        "List exactly four colors, sorted alphabetically, comma-separated, nothing else.",
        "Answer in two sentences max: what makes a good code review comment?",
        "Respond with a haiku (5-7-5) about late-night debugging, no preamble.",
    ],
}


def generated_arithmetic(n: int = 40, seed: int = 7):
    """Deterministic arithmetic/word-problem variations for cheap scale (§97).

    Same underlying capability as 'math' but with parametric surface forms, so
    prompts are never exact duplicates across runs.
    """
    import random
    rng = random.Random(seed)
    out = []
    ops = ["+", "-", "*"]
    for _ in range(n):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        op = rng.choice(ops)
        if op == "*" and (a > 30 or b > 30):
            a, b = min(a, 30), min(b, 30)
        q = f"{a} {op} {b}"
        forms = [
            f"Compute exactly: {a} {op} {b}.",
            f"What is {q}? Show the result only.",
            f"Calculate {q} and explain the carries/borrows if any.",
            f"A crate holds {a} items; each shipment moves {b} crates. "
            if op == "*" else f"Start with {a} units; {('remove' if op == '-' else 'add')} {b}. What is left/total?",
            f"Answer as JSON: {{\"result\": <value>}} for {q}.",
        ]
        out.append(forms[rng.randrange(len(forms))])
    return out


def all_prompts():
    return [(cat, p) for cat, ps in CATEGORIES.items() for p in ps]


def prompts_by_category(n_per_category: int = 5):
    return {cat: ps[:n_per_category] for cat, ps in CATEGORIES.items()}


def sample_prompts(n_per_category: int, seed: int = 0, categories=None):
    """Reproducible balanced sample (§97): n per category, all categories.

    categories: optional subset of category names. Returns [(cat, prompt)].
    """
    import random
    rng = random.Random(seed)
    out = []
    for cat, ps in CATEGORIES.items():
        if categories and cat not in categories:
            continue
        k = min(n_per_category, len(ps))
        out += [(cat, p) for p in rng.sample(ps, k)]
    return out


def bank_stats() -> dict:
    """Per-category counts + total, for logging prompt-set provenance."""
    d = {cat: len(ps) for cat, ps in CATEGORIES.items()}
    d["total"] = sum(d.values())
    return d
