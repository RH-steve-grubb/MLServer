SHELL := /bin/bash
VERSION := $(shell sed 's/^__version__ = "\(.*\)"/\1/' ./mlserver/version.py)
IMAGE_NAME := seldonio/mlserver

.PHONY: install-dev install-dev-odh install-dev-odh-cuda \
	_generate generate run build \
	push-test push test test-cuda lint fmt version clean licenses

.PHONY: bootstrap-test
bootstrap-test:
	for _runtime in ./runtimes/*; \
	do \
		[ "$$(basename $$_runtime)" = "onnx" ] && continue; \
		echo "Copying Tox configuration to $$_runtime..."; \
		cp tox.runtime.ini $$_runtime/tox.ini; \
	done

install-dev:
	poetry sync --with all-runtimes --with all-runtimes-dev

install-dev-odh:
	poetry sync --with odh-runtimes --with dev

install-dev-odh-cuda:
	poetry sync --with odh-runtimes-cuda --with odh-runtimes-cuda-dev --with dev

lock:
	echo "Locking mlserver deps..."
	poetry lock
	for _runtime in ./runtimes/*; \
	do \
		echo "Locking $$_runtime deps..."; \
		poetry lock -C $$_runtime; \
	done

_generate_model_repository: # "private" target to call `fmt` after `generate`
	poetry run bash ./hack/generate-types.sh model_repository

_generate_dataplane: # "private" target to call `fmt` after `generate`
	poetry run bash ./hack/generate-types.sh dataplane

generate-model-repository: | _generate_model_repository fmt

generate-dataplane: | _generate_dataplane fmt

generate: | _generate_model_repository _generate_dataplane fmt

run:
	mlserver start \
		./tests/testdata

build: clean
	./hack/build-images.sh ${VERSION}
	./hack/build-wheels.sh ./dist

clean:
	rm -rf ./dist ./build *.egg-info .tox
	for _runtime in ./runtimes/*; \
	do \
		rm -rf \
			$$_runtime/dist \
			$$_runtime/build \
			$$_runtime/*.egg-info \
			$$_runtime/.tox; \
	done

push-test:
	poetry config repositories.pypi-test https://test.pypi.org/legacy/
	poetry publish -r pypi-test
	for _runtime in ./runtimes/*; \
	do \
		poetry publish -C $$_runtime -r pypi-test; \
	done

push:
	poetry publish
	docker push ${IMAGE_NAME}:${VERSION}
	docker push ${IMAGE_NAME}:${VERSION}-slim
	for _runtime in ./runtimes/*; \
	do \
	  _runtimeName=$$(basename $$_runtime); \
		poetry publish --skip-existing -C $$_runtime; \
		docker push ${IMAGE_NAME}:${VERSION}-$$_runtimeName; \
	done

test:
	tox
	for _runtime in ./runtimes/*; \
	do \
		tox -c $$_runtime; \
	done

test-cuda:
	tox -c ./runtimes/onnx -e cuda

lint:
	black --check .
	flake8 .
	mypy ./mlserver
	for _runtime in ./runtimes/*; \
	do \
		mypy $$_runtime || exit 1; \
	done
	mypy ./tests
	mypy ./hack
	mypy ./benchmarking
	mypy ./docs/examples

lint-no-changes:
	git \
		--no-pager diff \
		--exit-code \
		.

lint-generate: generate lint-no-changes

licenses:
	tox --recreate -e licenses,licenses-cuda
	cut -d, -f1,3 ./licenses/cpu/license_info.csv \
		> ./licenses/cpu/license_info.no_versions.csv
	cut -d, -f1,3 ./licenses/cuda/license_info.csv \
		> ./licenses/cuda/license_info.no_versions.csv

fmt:
	poetry run black .

version:
	@echo ${VERSION}
