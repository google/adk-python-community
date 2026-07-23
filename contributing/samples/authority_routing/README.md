# Authority Routing sample

Demonstrates `AuthorityRoutingPlugin`: it classifies the **authority** of each
user turn as `ADVISE` / `EXECUTE` / `DEFER` / `STOP` and enforces that posture
on every tool call. Only `EXECUTE` authorizes side effects; a deterministic
keyword guard can only ever *tighten* the model router's verdict (strictest
posture wins), and the plugin fails **closed**.

## Run

```bash
python contributing/samples/authority_routing/main.py
```

Expected output:

```
[ADVISE ] DENIED  edit_file  <- Review the retry settings in config.yaml.
[EXECUTE] ALLOWED edit_file  <- Rename foo to bar in utils.py.
[DEFER  ] DENIED  shell      <- Deploy the release to production.
```

The third case shows the guard at work: even if the router had returned
`EXECUTE`, the word "deploy" trips the irreversibility tripwire and the floor
is raised to `DEFER`.

## Wiring a real model router

`main.py` uses a trivial keyword `CallableRouter` so the sample runs with no API
key. For real use, implement `PostureRouter.route()` with a model call (a Gemini
sketch is at the bottom of `main.py`) and pass it as `router=`. Parse the model
output against a schema and fail closed (return `DEFER`) if it is unparseable.
