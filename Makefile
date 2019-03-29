NAME := core

.PHONY: install clean

install:
	@python3 -m pip install -r requirements.txt

clean:
	@python3 setup.py clean
	@rm -rf $(NAME)/__pycache__/	2> /dev/null || true
	@rm -rf $(NAME).egg-info/ 		2> /dev/null || true
