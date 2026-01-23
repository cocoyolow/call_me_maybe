install:
	uv sync

run:
	uv run python3 -m src

debug:
	@echo "   Starting debugger..."
	@echo "   Useful commands:"
	@echo "   n (next)       - Execute next line"
	@echo "   s (step)       - Step into function"
	@echo "   c (continue)   - Continue until next breakpoint"
	@echo "   p <var>        - Print variable"
	@echo "   l (list)       - Show source code"
	@echo "   q (quit)       - Quit debugger"
	@echo ""
	uv run python3 -m pdb -m src.main

lint:
	uv run flake8 --exclude=llm_sdk,.venv
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude llm_sdk 

clean:
	rm -rf .mypy_cache
	rm -rf .venv
	rm -rf src/__pycache__
	rm -rf llm_sdk/__pycache__
	rm -rf .vscode

fclean: clean
	rm -rf data/output

.PHONY: install run debug lint clean fclean