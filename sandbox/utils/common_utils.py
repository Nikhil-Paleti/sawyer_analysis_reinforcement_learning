import logging
import logging.config
import yaml
import os


def load_config(filepath):
	with open(filepath) as file:
		params = yaml.safe_load(file)

	return params 


def set_logging(filepath):
	if not os.path.exists("log"):
		os.mkdir("log")
	logging.config.fileConfig(filepath)
	log = logging.getLogger("default")

	return log