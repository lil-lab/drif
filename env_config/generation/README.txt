# To generate environment configs with template data, such as in RSS 2018 paper, do the following steps:

1. Set up a data-collection parameter set. You can refer to parameters/run_params/rss_datacollect. call it param_file
2. generate_curriculum_config.py param_file
3. generate_template_annotations.py param_file
3. generate_template_curves.py param_file