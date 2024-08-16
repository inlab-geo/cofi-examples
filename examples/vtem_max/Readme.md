# Employing CoFI to invert VTEM max data for a thin plate target

This is a series  of notebooks that illustrate how CoFI can be employed to 
invert VTEM max data for a thin plate layered earth hybrid model, that is a model 
where an electrically conductive traget is approximated by a thin plate in the 
halfspace of a layered earth,

- Background
	- [Thin plate inversion](./thin_plate_inversion.ipynb)
- Synthetic examples
    - Inverting a single survey line
        - [Inline and vertical component](./single_survey_line.ipynb).
        - [Vertical component only](./single_survey_line_vertical_only.ipynb)
    - Inverting three survey lines
        - [Optimisation](./three_survey_lines_parameter_estimation.ipynb)
        - Surrogate model
            - [Latin Hypercube Sampling](./three_survey_lines_latin_hypercube_sampling.ipynb)
            - [Surrogate model creation](./three_survey_lines_surrogate_model_creation.ipynb)
            - [Inference using the surrogated model](./three_survey_lines_ensemble_method.ipynb) 
- Caber deposit
    - [Preprocessing](./caber_preprocessing.ipynb)
    - [Inversion](./caber_inversion.ipynb)
