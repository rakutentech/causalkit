Please follow the guideline if you would like to contribute a new algorithm or add extra functionalities.

- please add the new algorithm in the `devel` branch. It may be merged to `master` branch after fully testing.
- in `causalkit/base_model.py`, the `PyModel` class is a template model class. Please inherit from that class and implement required methods accordingly. Please also add your algorithm name as a field to the Enum `ModelType` class
- if you have a single model file, you may put directly in `causalkit` folder. If you have multiple model files, please create a new folder in `causalkit` folder and put them in that folder.
- please add a notebook in `example` folder to demonstrate how to use it. And for example, explain hyperparameters that can be tuned.
- please add a python script in `test` folder for automated testing. this is optional when adding it to `devel` branch. But it is needed when it is merged to `master` branch.