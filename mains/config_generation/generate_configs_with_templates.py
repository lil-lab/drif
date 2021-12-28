from env_config.generation.generate_random_config import generate_config_files
from env_config.generation.generate_template_curves import generate_template_curves
from env_config.generation.generate_template_annotations import make_annotations
from env_config.generation.generate_template_alignments import generate_thesaurus

#from env_config.generation.generate_random_config import START_I, END_I

import parameters.parameter_server as P

# SET THE CONFIGURATION OPTIONS ON TOP OF generate_random_config.py
START_I = 0
END_I = 100

if __name__ == "__main__":
    P.initialize_experiment()

    print("-" * 80)
    print("Generating random config files!")
    print("-" * 80)
    generate_config_files(START_I, END_I)

    print("-" * 80)
    print("Generating paths!")
    print("-" * 80)
    generate_template_curves(START_I, END_I)

    print("-" * 80)
    print("Generating annotations!")
    print("-" * 80)
    make_annotations(START_I, END_I)

    print("-" * 80)
    print("Generating thesaurus!")
    print("-" * 80)
    generate_thesaurus()
    # TODO: Add taking env pics