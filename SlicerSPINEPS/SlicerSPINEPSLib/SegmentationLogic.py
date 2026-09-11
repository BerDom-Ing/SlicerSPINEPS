"""Run SPINEPS inference out of process and load the result into the MRML scene.

Adapted from SlicerNNUnet (SlicerNNUNetLib/SegmentationLogic.py).

Inference runs in a child process (``PythonSlicer``) rather than in the Slicer
process. That keeps torch out of the application's address space, lets the user
cancel a long run, and lets stdout be streamed back as progress -- which matters
because CPU inference takes many minutes.
"""

import os
import re
import sys
from pathlib import Path
from typing import Callable, List, Optional

import qt
import slicer

from .Parameter import DERIVATIVES_FOLDER_NAME, Parameter
from .Signal import Signal

# CSI sequences emitted by `rich`. Left in the stream they show up in the log widget
# as literal "<ESC>[0m" noise around every progress line.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

#: Suffix SPINEPS uses for the final vertebra instance mask.
SEG_VERT_PATTERN = "*seg-vert_msk.nii.gz"
#: Suffix SPINEPS uses for the final semantic (subregion) mask.
SEG_SPINE_PATTERN = "*seg-spine_msk.nii.gz"

#: SPINEPS writes intermediate results into sibling folders of the final masks
#: (``output_raw_<format>`` and ``debug_<format>``, see ``spineps/seg_run.py``). Their
#: contents are named almost identically to the final masks -- ``seg-vert-raw_msk``
#: alongside ``seg-vert_msk`` -- but the raw vertebra mask is pre-cleanup and carries
#: top-to-bottom instance ids rather than anatomical vertebra labels. It must never be
#: picked up: a plain sort puts ``output_raw_.../`` before the final mask, so the choice
#: has to be explicit.
INTERMEDIATE_DIR_PREFIXES = ("output_raw_", "debug_")


class SegmentationLogic:
    r"""Write a volume to disk, run ``spineps sample`` on it, load the masks back.

    Progress is reported through :attr:`progressInfo`, failures through
    :attr:`errorOccurred`, and completion through :attr:`inferenceFinished`.

    Usage example:

    >>> from SlicerSPINEPSLib import SegmentationLogic, Parameter
    >>> logic = SegmentationLogic()
    >>> logic.progressInfo.connect(print)
    >>> logic.errorOccurred.connect(slicer.util.errorDisplay)
    >>> logic.inferenceFinished.connect(lambda: print(logic.loadVertebraeSegmentation()))
    >>> logic.setParameter(Parameter(semanticModel="t2w"))
    >>> logic.startSegmentation(volumeNode)
    """

    def __init__(self, process: Optional["Process"] = None):
        self.inferenceFinished = Signal()
        self.errorOccurred = Signal("str")
        self.progressInfo = Signal("str")

        self.inferenceProcess = process or Process(qt.QProcess.MergedChannels)
        self.inferenceProcess.finished.connect(self.inferenceFinished)
        self.inferenceProcess.errorOccurred.connect(self.errorOccurred)
        self.inferenceProcess.readInfo.connect(self.progressInfo)

        self._parameter: Optional[Parameter] = None
        self._tmpDir = qt.QTemporaryDir()

    def __del__(self):
        self.stopSegmentation()

    def setParameter(self, parameter: Parameter) -> None:
        self._parameter = parameter

    def startSegmentation(self, volumeNode: "slicer.vtkMRMLScalarVolumeNode") -> None:
        """Export the volume and launch SPINEPS on it."""
        if self._parameter is None or not self._parameter.isValid():
            self.errorOccurred("SPINEPS parameters are missing or invalid.")
            return

        inputFile = self._prepareInferenceDir(volumeNode)
        if inputFile is None:
            self.errorOccurred(f"Failed to export the volume node to {self.inDir}")
            return

        self._startInferenceProcess(inputFile)

    def stopSegmentation(self) -> None:
        self.inferenceProcess.stop()

    def waitForSegmentationFinished(self) -> None:
        self.inferenceProcess.waitForFinished()

    def loadVertebraeSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        """Load the per-vertebra instance mask produced by SPINEPS."""
        return self._loadSegmentation(SEG_VERT_PATTERN, "vertebra instance")

    def loadSemanticSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        """Load the semantic (subregion) mask produced by SPINEPS."""
        return self._loadSegmentation(SEG_SPINE_PATTERN, "semantic")

    def vertebraeSegmentationFile(self) -> Path:
        return self._findOutputFile(SEG_VERT_PATTERN, "vertebra instance")

    def semanticSegmentationFile(self) -> Path:
        return self._findOutputFile(SEG_SPINE_PATTERN, "semantic")

    def _loadSegmentation(self, pattern: str, description: str):
        path = self._findOutputFile(pattern, description)
        return slicer.util.loadSegmentation(path.as_posix())

    @staticmethod
    def _isIntermediateResult(path: Path) -> bool:
        return any(
            parent.name.startswith(INTERMEDIATE_DIR_PREFIXES) for parent in path.parents
        )

    def _findOutputFile(self, pattern: str, description: str) -> Path:
        # SPINEPS places results in a "derivatives_seg" folder whose exact location
        # depends on how it resolves the BIDS path, so search the whole temp tree.
        matches = sorted(Path(self._tmpDir.path()).rglob(pattern))
        matches = [m for m in matches if not self._isIntermediateResult(m)]
        if matches:
            return matches[0]

        # A non-zero exit code means SPINEPS crashed rather than merely producing
        # nothing; say so, because "no mask" on its own sends people looking in the
        # wrong place.
        exitCode = getattr(self.inferenceProcess, "exitCode", None)
        if exitCode:
            raise RuntimeError(
                f"SPINEPS exited with code {exitCode} without producing a "
                f"{description} mask.\n"
                "It failed before or during processing - see the log above for the "
                "actual error."
            )
        raise RuntimeError(
            f"SPINEPS did not produce a {description} mask ({pattern}).\n"
            "Check the log for errors and confirm the semantic model matches the "
            "modality of the input image."
        )

    @property
    def inDir(self) -> Path:
        return Path(self._tmpDir.path()).joinpath("rawdata")

    @property
    def derivativesDir(self) -> Path:
        return Path(self._tmpDir.path()).joinpath(DERIVATIVES_FOLDER_NAME)

    def _prepareInferenceDir(self, volumeNode) -> Optional[Path]:
        self._tmpDir.remove()
        self.inDir.mkdir(parents=True, exist_ok=True)

        volumePath = self.inDir.joinpath(self._parameter.inputFileName())
        self.progressInfo(f"Transferring volume to SPINEPS in {self._tmpDir.path()}\n")
        slicer.util.exportNode(volumeNode, volumePath.as_posix())
        return volumePath if volumePath.exists() else None

    @staticmethod
    def _slicerPythonDir() -> Path:
        return Path(sys.executable).parent.joinpath("..", "lib", "Python")

    @classmethod
    def _findSpinepsScriptPath(cls) -> Optional[Path]:
        """Locate the ``spineps`` console script installed by pip.

        Windows installs console scripts into ``Scripts``, Linux and macOS into ``bin``.
        """
        for path in ["Scripts", "bin"]:
            candidates = sorted(cls._slicerPythonDir().joinpath(path).glob("spineps*"))
            candidates = [c for c in candidates if c.suffix.lower() in ("", ".exe")]
            if candidates:
                return candidates[0].resolve()
        return None

    @staticmethod
    def _pythonSlicerPath() -> Path:
        exe = "PythonSlicer.exe" if os.name == "nt" else "PythonSlicer"
        return Path(sys.executable).parent.joinpath(exe)

    def _buildCommand(self, inputFile: Path):
        """Return (program, args) for the SPINEPS run.

        Prefers the installed console script. Falls back to bootstrapping the entry
        point through PythonSlicer when the script cannot be found (which happens with
        some editable installs).
        """
        args = self._parameter.asArgList(inputFile)

        scriptPath = self._findSpinepsScriptPath()
        if scriptPath is not None:
            return scriptPath.as_posix(), args

        bootstrap = (
            "import sys; from spineps.entrypoint import entry_point; "
            "sys.argv = ['spineps'] + sys.argv[1:]; entry_point()"
        )
        return self._pythonSlicerPath().as_posix(), ["-c", bootstrap] + args

    def _buildEnvironment(self) -> dict:
        """Return the environment overrides for the SPINEPS child process.

        PYTHONIOENCODING / PYTHONUTF8 are essential, not cosmetic. SPINEPS prints a
        banner with box-drawing characters through `rich`. When its stdout is a pipe,
        Python defaults to the Windows ANSI codepage (cp1252), which cannot encode
        them, and the process dies with UnicodeEncodeError before doing any work.
        """
        overrides = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            # SPINEPS prints through `rich`, which emits ANSI colour sequences even
            # into a pipe. They are not renderable by the log widget, so ask for
            # plain output at the source; _report() strips any that still arrive.
            "NO_COLOR": "1",
        }
        # Pin the weights folder so downloads land somewhere predictable and can be
        # pre-seeded on machines without internet access.
        if self._parameter.modelsFolder is not None:
            modelsFolder = Path(self._parameter.modelsFolder)
            modelsFolder.mkdir(parents=True, exist_ok=True)
            overrides["SPINEPS_SEGMENTOR_MODELS"] = modelsFolder.as_posix()
        return overrides

    def _startInferenceProcess(self, inputFile: Path) -> None:
        program, args = self._buildCommand(inputFile)

        self.progressInfo(
            "Starting SPINEPS with the following parameters:\n"
            f"\n{program} {' '.join(str(a) for a in args)}\n\n"
            f"{self._parameter.debugString()}\n"
        )
        self.progressInfo("SPINEPS preprocessing...\n")
        self.inferenceProcess.setEnvironment(self._buildEnvironment())
        self.inferenceProcess.start(
            program, args, qt.QProcess.Unbuffered | qt.QProcess.ReadOnly
        )


class Process:
    """Convenience wrapper around a QProcess run.

    Forwards read and error events, and kills the process on stop if it is running.
    Copied from SlicerNNUnet.
    """

    def __init__(self, channelMode: "qt.QProcess.ProcessChannelMode"):
        self.errorOccurred = Signal("str")
        self.finished = Signal()
        self.readInfo = Signal("str")

        self.exitCode: Optional[int] = None

        self.process = qt.QProcess()
        self.process.setProcessChannelMode(channelMode)
        self.process.finished.connect(self._onFinished)
        self.process.errorOccurred.connect(self._onErrorOccurred)
        self.process.readyRead.connect(self._onReadyRead)

    def stop(self) -> None:
        if self.process.state() == self.process.Running:
            self.readInfo("Killing process.")
            self.process.kill()

    def setEnvironment(self, overrides: dict) -> None:
        """Apply environment overrides on top of the inherited environment.

        Falls back to setting them on this process, which the child inherits. These
        variables are not optional -- without PYTHONIOENCODING the child dies on its
        first non-ASCII output -- so a missing PythonQt binding must not skip them.
        """
        try:
            environment = qt.QProcessEnvironment.systemEnvironment()
            for name, value in overrides.items():
                environment.insert(name, str(value))
            self.process.setProcessEnvironment(environment)
        except Exception:  # noqa: BLE001 - binding may be unavailable in PythonQt
            for name, value in overrides.items():
                os.environ[name] = str(value)

    def start(self, program, args: List[str], openMode: "qt.QIODevice.OpenMode") -> None:
        self.stop()
        self.exitCode = None
        self.process.start(program, [str(a) for a in args], openMode)

    def _onFinished(self, exitCode=0, *_) -> None:
        self.exitCode = int(exitCode)
        self.finished()

    def waitForFinished(self, timeOut_ms: Optional[int] = None) -> None:
        self.process.waitForFinished(timeOut_ms if timeOut_ms is not None else -1)

    def _onReadyRead(self) -> None:
        self._report(self.process.readAll(), self.readInfo)

    def _onErrorOccurred(self, *_) -> None:
        self._report(self.process.readAllStandardError(), self.errorOccurred)

    @staticmethod
    def _report(stream: "qt.QByteArray", outSignal: Callable[[str], None]) -> None:
        # Decode as UTF-8 unconditionally: the child is forced to UTF-8 in
        # _buildEnvironment(), and codecForUtfText() only recognises it from a BOM,
        # which SPINEPS does not emit. Without a BOM it falls back to the locale
        # codec -- cp1252 on Windows -- and renders the citation banner as mojibake.
        info = qt.QTextCodec.codecForName("UTF-8").toUnicode(stream)
        info = _ANSI_ESCAPE_RE.sub("", info)
        if info:
            outSignal(info)
