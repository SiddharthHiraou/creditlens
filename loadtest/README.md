# Load test

```bash
make serve       # terminal 1
make loadtest    # terminal 2 — headless, 50 users, 60s
```

Reported p99 in the project README comes from the headless run against a
single uvicorn worker on the developer machine, not from `TestClient`.
`TestClient` short-circuits the ASGI server and the network stack, so its
numbers understate the tail by the parts that actually degrade under load.
