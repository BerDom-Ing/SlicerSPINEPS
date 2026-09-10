"""Install SPINEPS into the 3D Slicer Python environment without breaking Slicer.

Design goals, in priority order:

1. **Self-sufficient.** A fresh 3D Slicer with only this extension added must reach a
   working SPINEPS from one button press, with no other extension, no installer, and no
   restart. PyTorch is installed here when it is absent, using the same light-the-torch
   selection the PyTorch extension uses, but without requiring that extension.
2. **Never break what already works.** Slicer is a shared Python environment. Other
   extensions -- SlicerMOOSE and SlicerTotalSegmentator in particular -- share torch
   and nnunetv2 with SPINEPS. Installing SPINEPS must not move those underneath them.
3. **Never install PyTorch if a working one exists.** A machine with MOOSE already has
   torch. Reinstalling it is a ~2.5 GB download that, on Windows, can fail against the
   260 character path limit and leave a broken install behind. Reuse beats reinstall.
4. **Let pip resolve.** Dependencies are installed in one pip call with a constraints
   file pinning what must not move, so pip's real resolver runs. Installing package by
   package with --no-deps silently produces wrong versions.

torch is never imported into the Slicer process, at any point. On Windows that would
load and lock its DLLs, so a later repair or reinstall fails halfway. Every check here
reads package metadata or runs in a child interpreter instead.

Packages that are never installed by us:

- ``SimpleITK``: Slicer ships a patched build with a custom IO class. Replacing it
  breaks volume reading application wide.
- ``torch`` / ``torchvision`` / ``torchaudio``: must match each other and the machine's
  CUDA version. Reused when present (MOOSE, the PyTorch extension, an earlier run);
  otherwise installed once by :meth:`InstallLogic.ensureTorchAvailable`, never as an
  ordinary resolved dependency.
- ``requests``: already present in Slicer; reinstalling only forces a restart.

Targets Slicer 5.10 (Python 3.12).
"""

import importlib.metadata
import importlib.util
import logging
import os
import re
import subprocess  # nosec B404 - used only to run Slicer's own interpreter
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from subprocess import CalledProcessError
from typing import List, Optional, Union

import qt
import slicer
from packaging.requirements import Requirement
from packaging.version import Version, parse

from .Signal import Signal

#: Minimum interpreter this module supports. Slicer 5.10 ships Python 3.12.
MINIMUM_PYTHON_VERSION = (3, 12)

#: Requirements we never install. Compared against the *parsed, normalized* requirement
#: name, never as a string prefix: "torchmetrics" and "pytorch-lightning" both contain
#: "torch" but are ordinary dependencies that must be installed normally.
PACKAGES_TO_SKIP = [
    "SimpleITK",
    "torch",
    "torchvision",
    "torchaudio",
    "requests",
]

#: Pinned to their installed version while resolving, so that adding SPINEPS cannot
#: move a package that Slicer or MOOSE depends on. Only pinned when already present.
PACKAGES_TO_PIN = [
    "SimpleITK",
    "numpy",
    "torch",
    "torchvision",
    "torchaudio",
    "nnunetv2",
]

#: Corrections to dependencies SPINEPS declares incorrectly, keyed by normalized name.
#: Both were found by installing SPINEPS 2.0.0 alongside MOOSE on Slicer 5.10.
DEPENDENCY_OVERRIDES = {
    # SPINEPS pins acvl-utils==0.2, but nnunetv2>=2.6 (required by MOOSE) calls
    # insert_crop_into_image(), which was added in 0.2.6. The three functions SPINEPS
    # itself imports are unchanged across that range.
    "acvl-utils": "acvl-utils>=0.2.6,<0.3",
    # SPINEPS declares TPTBox with no lower bound, so a resolver may select a version
    # too old to provide np_filter_connected_components.
    "tptbox": "TPTBox>=0.8",
    # These two ARE declared by SPINEPS, but an earlier version of this installer
    # stripped them from the installed METADATA while trying to exclude "torch"
    # (both names contain the substring). Listing them here means a machine with
    # damaged metadata still gets a correct, complete install.
    "pytorch-lightning": "pytorch-lightning>=2.0.8,<3.0.0",
    "torchmetrics": "torchmetrics>=1.1.2,<2.0.0",
}

#: Installed together when no PyTorch is present. torchvision must match torch, so they
#: are always selected in the same resolution step.
TORCH_PACKAGES = ["torch", "torchvision"]

#: Chooses the CUDA or CPU wheel that matches the installed NVIDIA driver. This is the
#: same mechanism the PyTorch extension uses; depending on the package directly rather
#: than on the extension is what makes this module installable on its own.
LIGHT_THE_TORCH_REQUIREMENT = "light-the-torch>=0.8"


def normalizePackageName(name: str) -> str:
    """Normalize a distribution name for comparison (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirementName(requirement: str) -> Optional[str]:
    """Return the normalized distribution name of a requirement string, or None.

    None means the string could not be parsed; callers must then err on the side of
    installing the requirement rather than silently dropping it.
    """
    try:
        return normalizePackageName(Requirement(requirement).name)
    except Exception:  # noqa: BLE001 - malformed metadata must not abort the install
        return None


def isPackageToSkip(requirement: str, packagesToSkip) -> bool:
    """Return True when a requirement names one of the packages we must not install.

    Matches on the parsed, normalized distribution name. A substring or prefix test
    would wrongly capture "torchmetrics" and "pytorch-lightning" for the "torch" entry.
    """
    name = requirementName(requirement)
    if name is None:
        return False
    return name in {normalizePackageName(skip) for skip in packagesToSkip}


class InstallLogic:
    r"""Install SPINEPS and its dependencies into Slicer's Python environment.

    Usage example:

    >>> logic = InstallLogic()
    >>> logic.progressInfo.connect(print)
    >>> logic.setupPythonRequirements()
    True
    """

    #: Shown when long paths are disabled *and* PyTorch actually has to be installed.
    LONG_PATHS_MESSAGE = (
        "PyTorch is not installed, and Windows long path support is disabled on this "
        "machine.\n\n"
        "Installing PyTorch would fail partway through a large download and leave a "
        "broken installation behind.\n\n"
        "Run this in PowerShell as Administrator, then reboot and press Install again:\n\n"
        "    New-ItemProperty -Path "
        '"HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
        "-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force\n\n"
        "This is only needed once, and only on a machine that has no PyTorch yet. If "
        "another extension that provides PyTorch (SlicerMOOSE, PyTorch) is already "
        "installed, run its dependency install instead and no reboot is required."
    )

    def __init__(self, doAskConfirmation: bool = True):
        self.progressInfo = Signal("str")
        self.doAskConfirmation = doAskConfirmation

    def _log(self, text: str) -> None:
        logging.info(text)
        self.progressInfo(text)

    # ------------------------------------------------------------------ environment

    @staticmethod
    def isPythonVersionSupported() -> bool:
        return sys.version_info >= MINIMUM_PYTHON_VERSION

    @staticmethod
    def areWindowsLongPathsEnabled() -> bool:
        """Return True unless Windows is known to enforce the 260 character path limit.

        Non-Windows platforms and unreadable registries return True, so this never
        blocks an install it cannot prove will fail.
        """
        if sys.platform != "win32":
            return True
        try:
            import winreg  # noqa: PLC0415

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
            ) as key:
                value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
                return int(value) == 1
        except FileNotFoundError:
            return False  # key absent means the default, which is disabled
        except Exception:  # noqa: BLE001 - never block on an unreadable registry
            return True

    @staticmethod
    def pythonSlicerPath() -> str:
        """Return the path to Slicer's standalone Python interpreter."""
        executable = "PythonSlicer.exe" if os.name == "nt" else "PythonSlicer"
        return os.path.join(os.path.dirname(sys.executable), executable)

    @classmethod
    def runInPythonSlicer(cls, code: str, timeoutSeconds: int = 300):
        """Run a snippet in a child interpreter and return (succeeded, output).

        Used for health checks. Running them out of process means a broken dependency
        cannot leave a half-initialised module in Slicer's own interpreter, and means
        torch is never imported into the Slicer process (which on Windows would lock
        its DLLs and prevent any later repair).
        """
        try:
            environment = dict(os.environ, PYTHONIOENCODING="utf-8")
            completed = subprocess.run(  # nosec B603 - fixed interpreter, no shell
                [cls.pythonSlicerPath(), "-c", code],
                capture_output=True,
                text=True,
                timeout=timeoutSeconds,
                env=environment,
                check=False,
            )
            return completed.returncode == 0, (completed.stdout + completed.stderr)
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    # ------------------------------------------------------------------- inspection

    @classmethod
    def getInstalledPackageVersion(cls, req: Union[str, Requirement]) -> Optional[Version]:
        """Return the installed version, or None if absent or unreadable.

        A half-written distribution (dist-info present but METADATA missing) reads as
        None rather than raising, so callers treat it as "needs installing".
        """
        req = cls.asRequirement(req)
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            return None
        except Exception:  # noqa: BLE001 - damaged metadata must not abort the install
            return None
        if not installed:
            return None
        try:
            return parse(installed)
        except Exception:  # noqa: BLE001
            return None

    def getInstalledSpinepsVersion(self) -> Optional[Version]:
        return self.getInstalledPackageVersion("spineps")

    @classmethod
    def isPackageInstalled(cls, req: Union[str, Requirement]) -> bool:
        return cls.getInstalledPackageVersion(req) is not None

    @classmethod
    def isInstalledPackageCompatible(cls, req: Union[str, Requirement]) -> bool:
        req = cls.asRequirement(req)
        installedVersion = cls.getInstalledPackageVersion(req)
        return installedVersion in req.specifier if installedVersion is not None else True

    @classmethod
    def isPackageInstalledAndCompatible(cls, req: Union[str, Requirement]) -> bool:
        return cls.isPackageInstalled(req) and cls.isInstalledPackageCompatible(req)

    @classmethod
    def asRequirement(cls, req: Union[str, Requirement]) -> Requirement:
        return req if isinstance(req, Requirement) else Requirement(req)

    @classmethod
    def isSpinepsImportable(cls) -> bool:
        """Return True when SPINEPS can actually be imported.

        Stronger than a version check: SPINEPS can be installed while a transitive
        dependency is missing or too old, which only shows up on import. This is what
        distinguishes "installed" from "installed and working".
        """
        ok, _ = cls.runInPythonSlicer("import spineps")
        return ok

    @staticmethod
    def isTorchInstallDamaged() -> bool:
        """Return True when torch is on disk but its metadata is unreadable.

        The signature of a pip install that was interrupted - most often by the Windows
        260 character path limit while unpacking the CUDA wheel's license tree.
        """
        try:
            torchSpec = importlib.util.find_spec("torch")
        except Exception:  # noqa: BLE001 - a broken install can raise here too
            return True
        if torchSpec is None:
            return False  # simply not installed, which is a different situation
        try:
            return not version("torch")
        except PackageNotFoundError:
            return True
        except Exception:  # noqa: BLE001
            return True

    @classmethod
    def torchStatus(cls) -> dict:
        """Describe the installed PyTorch without importing it into this process."""
        status = {
            "installed": False,
            "damaged": cls.isTorchInstallDamaged(),
            "version": None,
            "cpuOnly": False,
            "cudaGpuPresent": cls.isCudaGpuAvailable(),
        }
        torchVersion = cls.getInstalledPackageVersion("torch")
        if torchVersion is not None:
            status["installed"] = True
            status["version"] = str(torchVersion)
            local = torchVersion.local
            status["cpuOnly"] = local is not None and "cpu" in local
        return status

    @staticmethod
    def isCudaGpuAvailable() -> bool:
        """Return True when an NVIDIA driver is present, without needing torch.

        Tries light-the-torch first, then falls back to nvidia-smi. The fallback matters:
        this is called to describe the machine *before* anything is installed, when
        light-the-torch may not be present yet on a fresh Slicer.
        """
        try:
            import light_the_torch._cb as computationBackend  # noqa: PLC0415

            return computationBackend._detect_nvidia_driver_version() is not None
        except Exception:  # noqa: BLE001 - not installed yet, or a private API moved
            pass

        try:
            completed = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return completed.returncode == 0 and bool(completed.stdout.strip())
        except Exception:  # noqa: BLE001 - absent driver, absent binary, or timeout
            return False

    @staticmethod
    def isCpuOnlyTorchInstalled() -> bool:
        """Return True when the installed torch is a CPU-only build."""
        try:
            localVersion = parse(version("torch")).local
        except Exception:  # noqa: BLE001
            return False
        return localVersion is not None and "cpu" in localVersion

    # ------------------------------------------------------------------- installing

    def setupPythonRequirements(
        self, spinepsRequirement: str = "spineps", force: bool = False
    ) -> bool:
        """Install SPINEPS, reusing whatever PyTorch is already present.

        :param force: reinstall even when SPINEPS is already present and importable.
        """
        try:
            if not self.isPythonVersionSupported():
                raise RuntimeError(
                    "SPINEPS requires Python "
                    f"{'.'.join(str(v) for v in MINIMUM_PYTHON_VERSION)} or newer, but this "
                    f"Slicer runs Python {sys.version.split()[0]}. "
                    "Please use 3D Slicer 5.10 or newer."
                )

            if not force and self.isPackageInstalled(spinepsRequirement):
                if self.isSpinepsImportable():
                    self._log(
                        f"SPINEPS {self.getInstalledSpinepsVersion()} is already installed "
                        "and imports correctly. Nothing to do."
                    )
                    return True
                self._log(
                    "SPINEPS is installed but does not import; repairing its dependencies."
                )

            # Ask before anything is downloaded, and say how big it will be: the answer
            # differs by two orders of magnitude depending on whether torch is present.
            if self.doAskConfirmation:
                self._requestPermissionToInstallOrRaise()

            self.ensureTorchAvailable()
            self._installSpineps(spinepsRequirement)

            if not self.isSpinepsImportable():
                ok, output = self.runInPythonSlicer("import spineps")
                raise RuntimeError(
                    "SPINEPS was installed but still cannot be imported.\n"
                    f"{output.strip()[-800:]}"
                )

            self._log("SPINEPS installation completed successfully.")
            return True
        except Exception as e:  # noqa: BLE001 - surfaced to the user via the widget
            self._log(f"Error occurred during install: {e}")
            return False

    def ensureTorchAvailable(self) -> None:
        """Make sure a usable PyTorch exists, without replacing a working one.

        On a Slicer that already has another torch-based extension installed this
        normally does nothing at all. That is deliberate: reinstalling torch is the
        single most failure-prone step of the whole process.
        """
        status = self.torchStatus()

        if status["damaged"]:
            raise RuntimeError(
                "The installed PyTorch is damaged (a previous install did not finish).\n"
                "Close Slicer completely and follow the repair steps in the "
                "SlicerSPINEPS README, then try again."
            )

        if status["installed"]:
            note = "CPU-only build" if status["cpuOnly"] else "GPU-capable build"
            self._log(f"Reusing the PyTorch already installed ({status['version']}, {note}).")
            if status["cpuOnly"] and status["cudaGpuPresent"]:
                # Deliberately not repaired automatically: replacing torch risks
                # breaking MOOSE and can fail against the Windows path limit. Tell the
                # user and let them decide.
                self._log(
                    "NOTE: this machine has an NVIDIA GPU but PyTorch is a CPU-only build. "
                    "Segmentation will work but will be much slower. See the "
                    "SlicerSPINEPS README to switch to a CUDA build."
                )
            return

        # No torch at all. This is the only path that downloads gigabytes, and the only
        # one that can hit the Windows path limit.
        if not self.areWindowsLongPathsEnabled():
            raise RuntimeError(self.LONG_PATHS_MESSAGE)

        self._log("PyTorch is not installed. Installing it now...")
        self._installTorch()

        status = self.torchStatus()
        if not status["installed"] or status["damaged"]:
            raise RuntimeError(
                "PyTorch could not be installed. See the log above for the reason.\n"
                "The SlicerSPINEPS README documents how to install it manually."
            )
        self._log(f"PyTorch {status['version']} installed.")

    def _installTorch(self) -> None:
        """Install torch and torchvision matched to this machine's NVIDIA driver.

        Uses light-the-torch, the same selector the PyTorch extension uses, but invoked
        directly so that this module works on a Slicer where that extension is absent.
        Installing the extension instead would force a restart before the install could
        even begin.

        light-the-torch runs as ``python -m`` in a child interpreter, so torch is
        installed without ever being imported into the running Slicer.
        """
        if not self.isPackageInstalled("light-the-torch"):
            self._log("Installing light-the-torch (selects the right PyTorch build)...")
            self.pip_install(LIGHT_THE_TORCH_REQUIREMENT)

        backend = "CUDA" if self.isCudaGpuAvailable() else "CPU"
        self._log(
            f"Downloading PyTorch ({backend} build, this is a large download and may "
            "take several minutes)..."
        )
        slicer.util._executePythonModule("light_the_torch", ["install", *TORCH_PACKAGES])
        self._refreshPackageMetadataCache()

    def _spinepsDependencies(self) -> List[str]:
        """Return SPINEPS' declared dependencies, corrected and filtered.

        Read from the installed metadata rather than hard-coded, so a future SPINEPS
        release is picked up automatically; :data:`DEPENDENCY_OVERRIDES` only corrects
        the entries known to be wrong.
        """
        requirements = importlib.metadata.requires("spineps") or []
        resolved = []
        seen = set()

        for requirement in requirements:
            name = requirementName(requirement)
            if name is None:
                continue
            if isPackageToSkip(requirement, PACKAGES_TO_SKIP):
                continue
            marker = self.asRequirement(requirement).marker
            if marker is not None and not marker.evaluate():
                continue  # e.g. the python_version == "3.9" only entries
            resolved.append(
                DEPENDENCY_OVERRIDES.get(name, self.cleanPyPiRequirement(requirement))
            )
            seen.add(name)

        # Apply overrides for packages SPINEPS does not declare at all.
        for name, override in DEPENDENCY_OVERRIDES.items():
            if name not in seen:
                resolved.append(override)

        return resolved

    def _writeConstraintsFile(self) -> str:
        """Pin every already-installed protected package to its current version.

        This is what stops adding SPINEPS from moving torch or nnunetv2 underneath
        MOOSE, while still letting pip's real resolver do its job on everything else.
        """
        lines = []
        for name in PACKAGES_TO_PIN:
            installed = self.getInstalledPackageVersion(name)
            if installed is not None:
                lines.append(f"{name}=={installed}")

        handle, path = tempfile.mkstemp(prefix="spineps-constraints-", suffix=".txt")
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self._log("Pinning already-installed packages so they are not changed:")
        for line in lines:
            self._log(f"  {line}")
        return path

    def _installSpineps(self, spinepsRequirement: str) -> None:
        """Install SPINEPS itself, then its dependencies in a single resolved step."""
        self._log(f"Installing {spinepsRequirement} (without dependencies)...")
        self.pip_install(f"{self.cleanPyPiRequirement(spinepsRequirement)} --no-deps")

        # Read the dependency list BEFORE editing the metadata below, so the list is
        # never derived from a file this method has already modified.
        dependencies = self._spinepsDependencies()

        # Keep a later "pip install --upgrade spineps" from dragging in a torch that
        # does not match this machine.
        try:
            self._removeSkippedPackagesFromMetaDataFile("spineps", PACKAGES_TO_SKIP)
        except Exception as e:  # noqa: BLE001 - cosmetic hardening only
            logging.info(f"Could not adjust SPINEPS metadata: {e}")

        constraintsPath = self._writeConstraintsFile()
        try:
            self._log("Installing dependencies (pip resolves these together)...")
            self.pip_install(
                " ".join(dependencies) + f' --constraint "{constraintsPath}"'
            )
        finally:
            try:
                os.remove(constraintsPath)
            except OSError:
                pass

    def _requestPermissionToInstallOrRaise(self) -> None:
        if self.torchStatus()["installed"]:
            scale = (
                "PyTorch is already installed and will be reused, so this is a few "
                "hundred megabytes and usually a few minutes."
            )
        else:
            scale = (
                "PyTorch is not installed yet and will be downloaded as well, so expect "
                "roughly 2-3 GB and 10-30 minutes depending on your connection."
            )

        ret = qt.QMessageBox.question(
            None,
            "SPINEPS about to be installed",
            "The SPINEPS AI packages will be downloaded and installed into 3D Slicer.\n\n"
            f"{scale}\n\n"
            "Slicer does not need to be restarted afterwards.\n\n"
            "Would you like to proceed?",
        )
        if ret == qt.QMessageBox.No:
            raise RuntimeError("Install process was manually canceled by the user.")

    # ----------------------------------------------------------------------- helpers

    @classmethod
    def _removeSkippedPackagesFromMetaDataFile(cls, packageToInstall, packagesToSkip):
        def doSkipLine(metaLine):
            if not metaLine.startswith("Requires-Dist: "):
                return False
            # Parse the requirement rather than substring-matching the whole line:
            # "Requires-Dist: pytorch-lightning (>=2.0.8)" contains "torch" but must
            # be preserved, otherwise a later pip upgrade drops a needed dependency.
            return isPackageToSkip(metaLine[len("Requires-Dist: "):].strip(), packagesToSkip)

        # Latin-1: the file may contain non-ASCII characters and is not necessarily UTF-8.
        with open(cls.packageMetaFilePath(packageToInstall), "r+", encoding="latin1") as file:
            filteredLines = "".join([line for line in file if not doSkipLine(line)])
            file.seek(0)
            file.write(filteredLines)
            file.truncate()

    @staticmethod
    def packageMetaFilePath(packageToInstall):
        return [
            p for p in importlib.metadata.files(packageToInstall) if "METADATA" in str(p)
        ][0].locate()

    @staticmethod
    def cleanPyPiRequirement(requirement) -> str:
        """Return a requirement string compatible with slicer.util.pip_install."""
        req = Requirement(requirement)

        extras = [extra for extra in req.extras]
        extras = str(extras) if extras else ""

        # Handle the case where the extra ends up in the marker instead of the extras
        # spec, e.g. "ruff ; extra == 'dev'" -> "ruff[dev]".
        extra_pattern = "extra == "
        req_marker = str(req.marker)
        if not extras and req_marker.startswith(extra_pattern):
            req_marker = re.sub(r"\W+", "", req_marker.replace(extra_pattern, ""))
            extras = f"[{req_marker}]"

        return f"{req.name}{extras}{req.specifier}"

    @staticmethod
    def _refreshPackageMetadataCache() -> None:
        """Make packages installed by a child pip visible to this process.

        The install runs as ``python -m pip`` in a subprocess, so nothing in the running
        interpreter knows about it. Without this, an immediately following version check
        can still report the package as absent and the install looks like it failed.
        """
        importlib.invalidate_caches()

    def pip_install(self, package) -> None:
        self._log(f"- Installing {package}...")
        try:
            slicer.util.pip_install(package)
        except CalledProcessError as e:
            self._log(f"Install returned non-zero exit status: {e}. Attempting to continue...")
        finally:
            self._refreshPackageMetadataCache()

    def pip_uninstall(self, package) -> None:
        self._log(f"- Uninstalling {package}...")
        try:
            slicer.util.pip_uninstall(package)
        except CalledProcessError as e:
            self._log(f"Uninstall returned non-zero exit status: {e}. Attempting to continue...")
