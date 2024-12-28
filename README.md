# Project-8
Neural Networks for Partial Differential Equations

## Installing & Running
To install the project8 package, along with development dependencies, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), [AMGX](https://github.com/NVIDIA/AMGX), and [pyamgx](https://github.com/shwina/pyamgx).

Then, run
```sh
pip install '.[dev]'
```

To run the project as an executable, run
```sh
python -m project8
```

## DelftBlue
Add the following lines to `~/.bashrc`, and reopen your ssh session
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
NVARCH=`uname -s`_`uname -m`; export NVARCH
NVCOMPILERS=$HOME/.local/nvidia/hpc_sdk; export NVCOMPILERS
MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/24.7/compilers/man; export MANPATH
PATH=$NVCOMPILERS/$NVARCH/24.7/compilers/bin:$PATH; export PATH
```

Then, run the following commands to install the dependencies. This should take around 22 minutes.
```sh
cd delftblue
sbatch run_install.sh
```

To run the project as an executable, place the program's command-line arguments in `delftblue/args.txt` and edit the values in `delftblue/run_project8.sh`, then run
```sh
cd delftblue
sbatch run_project8.sh
```

## Using Docker
On Windows and MacOS, it's likely best to do everything in the docker container. Either download the docker image from the github repository's container repository, and tag it as `project8`
```sh
docker pull ghcr.io/project-08/project-8/project8:latest
```

Or build it as follows
```sh
cd docker
docker image build -t project8 .
```

Then to run an arbitrary command in docker, run
```sh
./docker_run.sh <YOUR_COMMAND>
```

To open an interactive shell in docker, simply run
```sh
./docker_run.sh bash
```

## Way of Working
- Don't merge to main, instead create a pull request and get it reviewed by at least one, but ideally two teammates.
- Create an issue with the task you're working on, so everyone can monitor progress. Close the issue upon merging your relevant PR.
- If you copy code FROM ANYWHERE, license it properly. Usually, this means copying over the license file, or reproducing some form of copyright notice.
- Only merge pull requests after the CI pipeline passes. You can run all the tools in the CI pipeline locally. To see the exact commands used, see [here](./.github/workflows/ci.yml).
