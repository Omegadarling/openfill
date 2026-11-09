# Changelog

All notable changes to OpenFill for PhotoCraft are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.16.0] - 2026-11-08

- Remove now suggests an available model from the active selection and its surrounding composite pixels. Both panels show the reason and support manual override.
- The local sampler reads PhotoCraft directly. Switching documents, revising a selection or changing sources invalidates stale suggestions before a removal can run.

## [0.15.0] - 2026-11-08

- Added selectable Big-LaMa, RORem and Qwen Image 2.1 Lightning removal on compatible existing servers. Both panels offer scene-based suggestions and manual choice.
- Repair layers carry the document's RGB profile, fixing a visible seam on Adobe RGB images. Neural removals apply the prepared selection edge once.
- FIBO's input guide preserves the original source outside the prepared mask, including the texture sampler's extra donor margin.

## [0.14.0] - 2026-11-08

- Local removal automatically reconstructs smooth backgrounds with continuous lighting instead of copying texture into large holes. Textured backgrounds continue to use PhotoCraft's content-aware fill.
- Added a small, softly blended edge margin to clear object-selection fringes. Include cast shadows in the selection when they should disappear too.
- Smooth backgrounds produce one deterministic repair. The original selection remains available, and Keep, Discard and appended choices use the prepared removal mask.

## [0.13.0] - 2026-11-08

- Added a prompt-free Remove mode to the existing-ComfyUI panels. Nearby texture samples the surrounding image in PhotoCraft and works offline. Rebuild background uses FIBO Dev and its original encoder to retouch a clean plate inside the selection.
- Both methods use the captured selection for the result layer and support variation review, Keep, and Discard. The managed engine also offers Nearby texture alongside its removal models.

## [0.12.1] - 2026-11-04

- Publish the first installable release of this PhotoCraft fork.
- Restore Qwen Image Edit 2511 Q4 downloads by pinning the file revision that matches the existing checksum.
- Skip reverse hostname lookups when starting the loopback browser panel, using the approach from the Patchy companion.

## [0.12.0-portable.1]

- Drop image files into Reference Images from the Windows native panel or the browser view.

## [0.11.2-portable.1]

- Halve the section disclosure arrows in the native panel and browser view.

## [0.11.1-portable.1]

- Make the section disclosure arrows three times larger in the native panel and browser view.

## [0.11.0-portable.1]

- Add new Image Edit and Fill results to the pending choices when you generate again on the same document.
- Put Keep, Keep all, and Discard below the thumbnails, including when there is only one result.
- Keep earlier thumbnails and their selection visible while another batch runs.

## [0.10.1-portable.1]

- Apply edited Generation controls to the next job and save them when the job is accepted.

## [0.10.0-portable.1]

- Add ordered image attachments for Qwen Image 2.1 and FIBO 2.0 edits.
- Add model visibility switches in Manage Models.
- Move settings into the main panel and remember section order and collapsed state.

## [0.9.0-portable.1]

- Add native Qwen Image 2.1 and four-step Lightning v2 RL image editing.
- Add FIBO 2.0 image editing with Preserve and Replace controls.
- Soften selection edges with configurable Auto-feather and match FIBO boundary colors.
- Discover compatible edit models from an existing local ComfyUI server.

## [0.2.4] - 2026-10-26

- Fix Mac prompt typing by keeping the expanded panel keyboard-focusable.
- Use the native Mac model menu and restore focus after selection or cancellation.
- Open the no-Tk panel in the default browser on Windows, macOS and Linux.
- Let an explicitly selected ComfyUI Python run OpenFill without installing another Python.
- Fix Linux/macOS interpreter fallback and paths containing spaces; document setup and PhotoCraft update discovery.

## [0.2.3] - 2026-10-26

- Detect long Windows paths before ComfyUI startup or download and explain the shorter-folder fix.
- Add PhotoCraft installation on Apple Silicon, tested with all four tools on an M3 MacBook Air.
- Fix Finder Python architecture, Aqua panel startup and PhotoCraft discovery on macOS.
- Add uninstall launchers on Windows, Linux and macOS, preserving settings and models.
- Document downloads, model sizes, existing-engine limits and tested Mac setup.

## [0.2.2] - 2026-10-25

### Added

- AMD graphics cards. I don't have an AMD card, so this is untested on real
  AMD hardware so far; please tell me how it goes.
  - On Windows, **Download ComfyUI** gets ComfyUI's own version for AMD cards
    (ROCm 7.2.1, 1.6 GB) for a Radeon RX 7000 or RX 9000, Radeon PRO W7000,
    or Ryzen AI 300 or Max chip, from the same ComfyUI release and checked the
    same way as the NVIDIA one. The setup card says which one it gets. For an
    older Radeon it says to follow the AMD steps in the install guide and
    choose the folder.
  - On Linux, a ComfyUI with PyTorch for ROCm works through **Choose folder**,
    and **How to install** opens the AMD steps when the card is AMD.
  - Models shows an AMD card's name and graphics memory
    ("RX 7900 XT · 20 GB VRAM · ..."), and on an AMD card the GGUF models say
    that they have been reported to give only noise on some Radeon cards.
- The README says where the model manager is: the gear icon, Models.

## [0.2.1] - 2026-10-25

### Fixed

- Keep could lose the variations. Clicking in PhotoCraft (to deselect, say)
  while Keep was still working made it fail with "select.modify.feather is not
  available right now: no selection", and the variations were gone. Keep now
  makes the kept layer from the job's own selection first and only then takes
  the variations away, a click on the way is tried again, and if Keep still
  fails the variations stay as they were and Keep can be pressed again. The
  same goes for the result a Fill, Remove or Expand adds at the end.
- Remove with LaMa failed the first time on a ComfyUI you already had, with
  "needs the custom node pack(s) openfill-nodes; press Install nodes in the
  Models tab", until another job had started ComfyUI. OpenFill now puts its
  own node pack in place before ComfyUI starts, as it does for the ComfyUI it
  downloads for you.

## [0.2.0] - 2026-10-25

First public release.

### Added

- Generative Fill, Remove object, Generative Expand and Text to Image for
  PhotoCraft 0.12 to 0.16, running on a local ComfyUI with OpenFill's engine (the
  same models, model manager, seam correction and variations as OpenFill for
  GIMP).
- The OpenFill panel next to PhotoCraft: Fill, Remove, Expand and Create, a
  live preview, variations with thumbnails, and Models and settings behind the
  gear. It folds into a small strip, and on Windows it stays with PhotoCraft's
  window (it hides when PhotoCraft is minimized and comes back with it).
- Filter > Plug-ins > OpenFill in PhotoCraft brings the panel up.
- Remove object at the image's own resolution with LaMa, through OpenFill's
  own ComfyUI node. MAT, RORem, the FLUX.1 Fill object removal LoRA and every
  fill model can be picked for Remove too.
- Windows installer (`Install OpenFill for PhotoCraft.cmd`): "PhotoCraft with
  OpenFill" in the Start menu and on the desktop. It adds Pillow to a
  python.org Python that has Tk but not Pillow.
- Windows first run: **Download ComfyUI** in the panel downloads the same
  checked portable ComfyUI and inpaint nodes OpenFill for GIMP installs, in the
  background, and uses it right away.
- Linux installer (`install.sh`): hands PhotoCraft's menu entry to OpenFill, or
  adds a separate one, and can download PhotoCraft itself.

### Changed

- Every result is one step in PhotoCraft's history, an expand included (it
  crops the canvas and adds its layers, then the steps are made into one named
  Generative Expand). Keep after picking a variation leaves it one step, with
  the kept variation alone; Discard takes the result back completely.
- Create as a layer with a selection makes a layer the size of the selection,
  placed on it, as in OpenFill for GIMP.
- A removal's layer is called Object Removal, whichever model made it.

### Fixed

- The FLUX.1 Fill removal LoRA asked for the FLUX.1 Fill Q4 GGUF (another
  6.8 GB) even when the Q6 GGUF or the full FLUX.1 Fill was installed. It now
  runs on whichever FLUX.1 Fill is there.
- Picking the active fill model in the Remove menu ran LaMa instead.
- A busy ComfyUI that did not answer one status request for a moment failed
  the job with "The AI backend is not available".
- Starting a fill without a selection from the page named GIMP's tools.
- PhotoCraft is started without its own control and automation variables
  (`PHOTOCRAFT_CONTROL_TOKEN` and the others) in the environment. A token
  exported next to OpenFill's token file stopped PhotoCraft from starting.
- Image names that start with `-` reach PhotoCraft as files, not as options
  (PhotoCraft refuses an option it does not know).

### Known limits

- Fill with FLUX or Qwen, Expand with every model, and Create work at about
  1024 px on the long side and are scaled up on large images (see the README).
  A second Expand pass at full resolution for SD 1.5 and SDXL is in the code
  but off: it has not been tested on real photos yet.
