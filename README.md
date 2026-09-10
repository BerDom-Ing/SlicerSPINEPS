# SlicerSPINEPS

A [3D Slicer](https://www.slicer.org) extension for
[SPINEPS](https://github.com/Hendrik-code/spineps), a whole-spine segmentation pipeline
for MRI (T2w, T1w, Vibe) and CT.

From a single volume it produces:

- a **per-vertebra instance mask** with anatomical labels (C1-C7, T1-T13, L1-L6, sacrum),
  plus intervertebral discs and endplates;
- a **semantic subregion mask** (vertebral body, arcus, spinous process, spinal cord,
  spinal canal, discs).

Both are loaded into the scene as `vtkMRMLSegmentationNode`s, ready for surface
generation, measurement, or use by another module.

**Install once, from one button.** No other extension, no separate installer, no
administrator rights, and no Slicer restart. PyTorch included.

Developed and maintained by **Bernardo Dominguez**.

---

## Requirements

**3D Slicer 5.10 or newer (Python 3.12).** SPINEPS is developed on Python 3.11, and its
dependency stack (`antspyx`, `nnunetv2`, `TPTBox`) either has no wheels or needs
version-specific workarounds on the Python 3.9 shipped by Slicer 5.8 and earlier.
`InstallLogic.isPythonVersionSupported()` enforces this rather than letting the install
fail halfway.

A CUDA GPU is **optional**. `light-the-torch` inspects the NVIDIA driver and selects a
matching CUDA wheel, or a CPU wheel when there is no usable GPU. CPU inference works and
is roughly 10x slower.

Nothing else. In particular the PyTorch extension is *not* required - see below.

## Installation

Until this extension is available in the Extensions Manager, clone the repository and add
`SlicerSPINEPS/SlicerSPINEPS` (the inner folder, the one containing `SlicerSPINEPS.py`)
in **Settings > Modules > Additional module paths**, then restart Slicer.

Open the **SPINEPS** module (Segmentation category) and press
**Install SPINEPS dependencies**. That is the whole procedure.

Two cases, and the module picks between them automatically:

| Machine | What is downloaded | Roughly |
|---|---|---|
| Already has PyTorch (from the PyTorch extension, SlicerMOOSE, SlicerTotalSegmentator, ...) | SPINEPS and its remaining dependencies only | a few hundred MB, a few minutes |
| Fresh Slicer, nothing else installed | PyTorch as well | 2-3 GB, 10-30 minutes |

Reusing an existing PyTorch is not just an optimisation. Reinstalling torch is the single
most failure-prone step of the process: it is the only one large enough to hit the Windows
260-character path limit, and on Windows it cannot be repaired from inside a Slicer that
has already imported it.

**The one case that needs a reboot** is a fresh Windows machine with *no* PyTorch and long
paths disabled. The module detects this before downloading anything and prints the
one-line registry command to fix it, rather than failing halfway through 2 GB. It never
applies to a machine that already has PyTorch.

### Why there is no `EXTENSION_DEPENDS` on PyTorch

Declaring the PyTorch extension as a dependency would force an extension install and a
Slicer restart before the user could even press the install button. Instead this module
depends on the `light-the-torch` *package* and invokes it the same way `PyTorchUtils`
does - as `python -m light_the_torch install torch torchvision` in a child interpreter.

Running it out of process matters: `PyTorchUtils.installTorch()` ends with `import torch`,
and on Windows an imported torch locks its own DLLs, so any later repair or reinstall
fails halfway and leaves an unusable distribution behind. **This module never imports
torch into the Slicer process, at any point** - every status check reads package metadata
or runs in a child interpreter.

### How dependencies are resolved

`spineps` itself is installed with `--no-deps`, then all of its dependencies are installed
in **one** pip call so that pip's real resolver runs, together with a constraints file
pinning everything that must not move:

| Pinned to installed version | Why |
|---|---|
| `SimpleITK`, `numpy` | Slicer's own builds. `SimpleITK` in particular uses a custom IO class; replacing it breaks volume reading application-wide. |
| `torch`, `torchvision`, `torchaudio` | Must match each other and the machine's CUDA version. |
| `nnunetv2` | Shared with other extensions (SlicerMOOSE, SlicerTotalSegmentator). Moving it would break them. |

`requests` is additionally never installed - it is already in Slicer, and reinstalling it
only forces a restart.

Installing package-by-package with `--no-deps` never runs a resolver and silently produces
wrong versions; that is how a `TPTBox 0.3.0` install slipped through during development.
One call with constraints gets a real resolution that still cannot move anything another
extension depends on.

Two of SPINEPS' declared dependencies are corrected during install
(`DEPENDENCY_OVERRIDES` in `InstallLogic.py`), both found empirically:

- **`acvl-utils`** - SPINEPS pins `==0.2`, but `nnunetv2>=2.6` calls
  `insert_crop_into_image()`, added in `0.2.6`. The three functions SPINEPS itself imports
  are unchanged across that range.
- **`TPTBox`** - declared with no lower bound, so a resolver may pick a version too old to
  provide `np_filter_connected_components`.

`pytorch-lightning` and `torchmetrics` are also listed explicitly. They *are* declared by
SPINEPS, but an earlier version of this installer stripped them from the installed
`METADATA` while trying to exclude `torch` (both names contain that substring). Listing
them means a machine with damaged metadata still gets a complete install.

## Known-good dependency set

Verified on Slicer 5.10.0 (Python 3.12), Windows 11, RTX 3070:

| Package | Version | Note |
|---|---|---|
| spineps | 2.0.0 | |
| TPTBox | 0.8.2 | **not** the 0.3.0 a plain resolve may pick; older versions lack `np_filter_connected_components` |
| nnunetv2 | 2.8.0 | |
| acvl-utils | **0.2.6** | overrides SPINEPS' stale `==0.2` pin |
| torch | 2.14.0+cu126 | CUDA build; a `+cpu` build runs but is ~10x slower |
| torchmetrics | 1.9.0 | |
| monai | 1.6.0 | |
| antspyx | 0.4.2 | |
| SimpleITK | 2.5.5 | Slicer's own build - must never be replaced |
| numpy | 2.5.0 | Slicer's own - must never be replaced |

Two `pip check` warnings are expected and deliberate: `acvl-utils 0.2.6` against SPINEPS'
`==0.2` pin, and `rich 15.0.0` against its `<14.0.0` pin. Both were verified against the
symbols SPINEPS actually imports.

## Model weights

The weights are **not** part of the dependency install. SPINEPS downloads them from its
GitHub releases on first use, per model - a few hundred MB each, and only for the models
actually selected (`t2w`, `t1w`, `vibe`, `ct`, plus the matching instance and labeling
models). So a successful install can still be followed by a first run that stalls for
several minutes on a download, or fails outright with no network.

The module handles this explicitly:

- **Location is pinned.** `SPINEPS_SEGMENTOR_MODELS` is set on the inference process to
  `<Slicer user settings dir>/SPINEPS/models`, so weights never land in an unpredictable
  per-user cache. An existing `SPINEPS_SEGMENTOR_MODELS` in the environment wins, so a
  deployment can point it at a shared or read-only location.
- **State is visible before you start.** The module panel shows which models are on disk
  and where. If none are, it says so, gives the path, and says the first run needs
  internet - rather than letting the operator discover it mid-procedure.
- **The download is not silent.** SPINEPS' output is streamed into the log during the run,
  and the run is cancellable.

**For an air-gapped machine**: run each model once on a connected machine, then copy that
folder across. No re-download, no code change, no network at use time.

## Using it from another module

The library is the real interface; the widget is a convenience.

```python
from SlicerSPINEPSLib import Parameter, SegmentationLogic, defaultModelsFolder

logic = SegmentationLogic()
logic.progressInfo.connect(print)
logic.errorOccurred.connect(slicer.util.errorDisplay)
logic.inferenceFinished.connect(lambda: logic.loadVertebraeSegmentation())
logic.setParameter(Parameter(semanticModel="t2w", modelsFolder=defaultModelsFolder()))
logic.startSegmentation(volumeNode)
```

`loadVertebraeSegmentation()` returns the per-vertebra instance mask (`seg-vert`);
`loadSemanticSegmentation()` returns the subregion mask (`seg-spine`).

### Instance mask label values

Following the [TPTBox](https://github.com/Hendrik-code/TPTBox) convention
(`core/vert_constants.py`), for a vertebra labelled `n`:

| Label | Meaning |
|---|---|
| 1-7 | C1-C7 |
| 8-19 | T1-T12 (28 = T13) |
| 20-25 | L1-L6 |
| 26 | S1 (sacrum) |
| `n + 100` | the intervertebral disc **below** vertebra `n` (e.g. 120 = disc below L1) |
| `n + 200` | the endplate of vertebra `n` (e.g. 220 = L1 endplate) |

The distinction matters for surface generation: endplates are cortical bone, discs are
soft tissue. Merging discs into a bone surface bridges every level with a slab of
non-bone.

### Missing T12, and transitional vertebrae

If your labels jump from T11 (18) straight to L1 (20), SPINEPS' labeling search
deliberately allowed itself to skip the class after T11, which is how it models
thoracolumbar transitional anatomy (`allow_skip_at_class` in `phase_labeling.py`). On a
truncated field of view this is often wrong - counting up from the unambiguous sacrum
makes the vertebra above L1 a T12.

The **Assume 12 thoracic vertebrae** checkbox (`-no_tltv_labeling`) forbids the skip. It
is off by default on purpose: forcing 12 thoracic vertebrae onto a patient who genuinely
has a variant is the same error inverted.

### CT

Selecting the `ct` semantic model also switches the instance and labeling models to their
CT-specific variants, and the BIDS acquisition tag to `iso`. This is not cosmetic: the
`spineps sample` subcommand has no `"auto"` model resolution, so it will silently accept
an MRI instance model applied to a CT and return a poor segmentation rather than an error.
`Parameter` pairs them so that cannot happen.

## Design notes

- Inference runs **out of process** (`qt.QProcess` driving the `spineps` console script
  under `PythonSlicer`). This keeps torch out of Slicer's address space, lets the user
  cancel a long run, and lets stdout be streamed back as progress. CPU inference takes
  many minutes, so cancel and progress are not optional.
- The input volume is exported with a BIDS-style name
  (`sub-spineps_acq-sag_T2w.nii.gz`) so SPINEPS infers the modality and acquisition from
  the file name and needs no compatibility override flag.
- SPINEPS writes intermediate results into sibling folders of the final masks
  (`output_raw_<format>/`, `debug_<format>/`) with near-identical names. The raw vertebra
  mask carries top-to-bottom instance ids rather than anatomical labels, so it must never
  be picked up - those folders are excluded explicitly, and there is a regression test.
- `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` are set on the child process: SPINEPS'
  citation banner contains characters outside cp1252 and otherwise crashes the run at exit
  on a Windows console.

## Repairing a damaged PyTorch install

If the module reports **"PyTorch is damaged"**, a pip install or uninstall was attempted
while torch was loaded into a running Slicer. On Windows the DLLs stay locked, so pip
leaves a partially written distribution behind (typically a `dist-info` folder with no
`METADATA`, and a missing `torchgen`).

This cannot be fixed from inside Slicer - the process holding the files must exit first.
**Close Slicer completely**, then from a terminal:

```sh
SLICER="$LOCALAPPDATA/slicer.org/3D Slicer 5.10.0"
PY="$SLICER/bin/PythonSlicer.exe"
SP="$SLICER/lib/Python/Lib/site-packages"

# 1. Remove every trace of the broken install (pip cannot, the metadata is unreadable)
rm -rf "$SP"/torch "$SP"/torchgen "$SP"/torchvision "$SP"/functorch
rm -rf "$SP"/torch-*.dist-info "$SP"/torchvision-*.dist-info "$SP"/torchgen-*.dist-info

# 2. Reinstall a CUDA build (drop the --index-url for a CPU-only machine)
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Verify -- must print True on a CUDA machine
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then start Slicer and press **Reinstall SPINEPS** to restore the remaining packages.

## License and citation

This extension is released under the **Apache License 2.0** (see [LICENSE](LICENSE)).

It wraps but does not include SPINEPS, which is also Apache-2.0 and is installed from PyPI
at runtime. `InstallLogic` and `SegmentationLogic` are adapted from
[SlicerNNUnet](https://github.com/KitwareMedical/SlicerNNUnet) (Apache-2.0), which in turn
adapted the selective-install approach from SlicerTotalSegmentator.

If you use this module in research, please cite SPINEPS. The current citation is given in
the [SPINEPS repository](https://github.com/Hendrik-code/spineps#citation).

## Acknowledgements

SPINEPS was developed by Hendrik Moeller et al. at the Technical University of Munich.
This extension only makes it usable from inside 3D Slicer.
