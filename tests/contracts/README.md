# Event contracts (step 7)

Pinned JSON Schemas of the cross-service events in `iqs_common.events`, plus one example
payload per event. Both services' test suites compare the live Pydantic schema with these
files (`packages/{vision,diagnosis}/tests/test_event_contracts.py`), so a change to an event
is visible in review and cannot slip through one service only.

Regenerate after a deliberate change (and say so in the commit message):

```bash
uv run python scripts/export_event_schemas.py
```
