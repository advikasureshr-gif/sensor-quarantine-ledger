# Vendored engine - NOT modified

The files in this folder are an unmodified copy of the `iotguard/` package from the supplied project
`online-behavioural-anomaly-main.zip` (your teammate's work). Only the modules the dashboard needs were copied:
calibration, config, consensus, contract, features, gaussian, pipeline, preprocess, presets, scenarios, security,
sim_sensors, sim_world, sources, watchdog.

NOT copied: run.py, record.py, experiments.py (it uses pickle), tests and conftest.

The uploaded ZIP contained no LICENSE file, so no licence text could be carried over. Ask the author which licence
applies before redistributing this package. Copyright remains with the original author(s).
If the original repository changes, replace these files with the new versions (see ../integration/README.md).

Added afterwards: `chain.py` (the project's blockchain layer, an unmodified copy of `iotguard/chain.py`), used only by the
"Real Recording Run" page. It needs no other file beyond `security.py`.
