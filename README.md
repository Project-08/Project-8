# Project-8
Neural Networks for Partial Differential Equations

## Installing & Running
To install the project8 package, along with development dependencies, run
```sh
pip install '.[dev]'
```

To run the project as an executable, run
```sh
python -m project8
```

## Way of Working
- Don't merge to main, instead create a pull request and get it reviewed by at least one, but ideally two teammates.
- Create an issue with the task you're working on, so everyone can monitor progress. Close the issue upon merging your relevant PR.
- If you copy code FROM ANYWHERE, license it properly. Usually, this means copying over the license file, or reproducing some form of copyright notice.
- Only merge pull requests after the CI pipeline passes. You can run all the tools in the CI pipeline locally. To see the exact commands used, see [here](./.github/workflows/ci.yml).
