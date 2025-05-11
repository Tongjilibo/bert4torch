# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from .pipelines.chat import run_llm_serve
from enum import Enum, unique


VERSION = "v0.5.7"


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   bert4torch serve: launch an llm server                           |\n"
    + "|   bert4torch version: show version info                            |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to bert4torch, version {VERSION}"
    + " " * (24 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/Tongjilibo/bert4torch |\n"
    + "-" * 58
)


@unique
class Command(str, Enum):
    SERVE = "serve"
    VER = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.SERVE:
        run_llm_serve()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    main()
