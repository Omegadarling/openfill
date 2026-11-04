# OpenFill for PhotoCraft development notes

Read this before editing the companion. This branch starts at OpenFill `photocraft-v0.2.4`. PhotoCraft-specific UI and control code live in `photocraft/openfill_photocraft`. The shared engine is in `gimp-openfill/openfill`. Build the distribution with `tools/make_photocraft_zip.py`.

- Existing-server mode defaults to a user-started ComfyUI at `127.0.0.1:8188`. Stock mode retains its managed engine. Model discovery reads actual loader choices and preserves their exact spelling, case, and path separators.
- For Qwen Image 2.1 editing, use `TextEncodeQwenImage21` and its source-aligned latent. Restore the original crop size before compositing. The Lightning checkpoint needs the exact published four sigmas plus the terminal zero.
- The document remains in reference slot 1. Use consecutive slots for attachments and derive the prompt labels from the selected model. The native and browser panels share reference state. Capture references at submission so later panel changes cannot affect an active edit.
- Preserve the document exterior with a hard selection in the graph. Apply the saved soft mask once when PhotoCraft inserts the result. A source image passed through an encoder alone does not constrain geometry or guarantee reference adherence.
- Persist model visibility separately from model installation and selection. The panel's layout preferences store stable section IDs, which can be restored when an older profile is loaded.
- Keep unsent Generation control edits through polling. Send the changed values with the next job so the service can validate and save them before starting generation. Keep Engine settings on their own Save action.
- Keep personal backend definitions, model inventories, test documents, configuration, logs, and weights out of public source and Git history.
- Run the shared and PhotoCraft companion unit tests. A live editor test should use a disposable PhotoCraft profile, verify exterior pixels and Discard, and leave existing ComfyUI jobs alone. Inspect renders separately for visual quality.
