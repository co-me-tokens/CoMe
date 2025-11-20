"""
MIT License

Copyright (c) 2025 Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Code adapted from https://github.com/MAC-VO/MAC-VO
"""

import atexit
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


class SandboxFile:
    def __init__(self, root_path: Path, name: str, mode: str):
        self.file = Path(root_path, name)
        self.mode = mode
        self.fp = None

    def __enter__(self):
        self.fp = open(self.file, self.mode)
        return self.fp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp is None:
            return

        self.fp.close()
        return True


class Sandbox:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        if not self.folder.exists():
            self.folder.mkdir(parents=True)

    @classmethod
    def create(cls, project_root: Path, project_name: str, timestamp_dir: bool = True):
        timestr = cls.__get_curr_time()
        try:
            gitver = cls.__get_git_version()
        except:
            gitver = "NOT_AVAILABLE"
        cmd = cls.__get_sys_command()
        gpu_ver = cls.__get_sys_gpu_model()

        if timestamp_dir:
            box = cls(Path(project_root, project_name, timestr))
        else:
            box = cls(Path(project_root, project_name))

        with box.open("metadata.yaml", "w") as f:
            yaml.dump({"time": timestr, "git_version": gitver, "command": cmd, "gpus": gpu_ver}, f)
        return box

    @classmethod
    def load(cls, root: Path | str):
        if isinstance(root, str):
            root_path = Path(root)
        else:
            root_path = root
        if not root_path.exists():
            raise FileNotFoundError(f"Unable to load sandbox from {root_path}")
        return Sandbox(root_path)

    def path(self, name: str | Path) -> Path:
        target_path = Path(self.folder, name).parent
        if not target_path.exists():
            target_path.mkdir(parents=True)
        return Path(self.folder, name)

    def open(self, name: str, mode: str) -> SandboxFile:
        target_path = self.path(name)
        if not target_path.parent.exists():
            target_path.parent.mkdir(parents=True)
        return SandboxFile(self.folder, name, mode)

    def path_folder(self, name: str) -> Path:
        target_path = self.path(Path(self.folder, name))
        if not target_path.exists():
            target_path.mkdir(parents=True)
        return target_path

    def new_child(self, name: str) -> "Sandbox":
        subbox = Sandbox.create(self.folder, name, timestamp_dir=False)
        with self.open("children.txt", "a") as f:
            f.write(str(subbox.folder.relative_to(self.folder).as_posix()) + "\n")
        return subbox

    def get_children(self) -> list["Sandbox"]:
        if not Path(self.folder, "children.txt").exists():
            return []

        with self.open("children.txt", "r") as f:
            lines = f.read().strip().split("\n")
        return [Sandbox.load(Path(self.folder, Path(subbox_path))) for subbox_path in lines]

    def get_leaves(self) -> list["Sandbox"]:
        children = self.get_children()
        if len(children) == 0:
            return [self]
        result = []
        for child in children:
            result.extend(child.get_leaves())
        return result

    def set_autoremove(self):
        print(f"Sandbox at '{str(self.folder)}' is set to be auto-removed.")

        def autophagy():
            try:
                shutil.rmtree(str(self.folder))
            except Exception as _:
                print(f"Failed to auto-remove sandbox at {str(self.folder)}")

        atexit.register(autophagy)

    @staticmethod
    def __get_sys_command():
        return " ".join(sys.orig_argv)

    @staticmethod
    def __get_curr_time():
        time_str = datetime.now().strftime("%m_%d_%H%M%S")
        return time_str

    @staticmethod
    def __get_git_version():
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )

    @staticmethod
    def __get_sys_gpu_model():
        import torch
        if torch.cuda.is_available():
            return [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
        else:
            return ["No GPU"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sandbox")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--proj", type=str, required=True)
    args = parser.parse_args()

    box = Sandbox.create(args.path, args.proj)
    print(str(box.folder.absolute()))
