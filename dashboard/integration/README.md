# Integration notes

The dashboard imports the engine only in `dashboard_service.py`: `iotguard.scenarios.make_source`, `iotguard.presets.room_setup`,
`iotguard.pipeline.Pipeline`, `iotguard.security.Ledger`, `iotguard.sources.strip_labels`.

The engine is vendored in `../iotguard/` (unmodified). To update it from the original repository, replace those files with the new
versions of the same names and run the tests. If a module the dashboard needs is renamed, adjust only `dashboard_service.py`.

Not integrated (not implemented): live ESP32 serial input, recorded-log replay, admin authentication, saved pipeline state.
`run.py` / `record.py` support these in the original project; they were deliberately not copied.
`experiments.py` is not copied because it loads pickle files.
