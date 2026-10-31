UID := $(shell id -u)
GID := $(shell id -g)
PWD = $(shell pwd)

args=
# if dockerfile is not defined
ifndef dockerfile
	dockerfile=compose.yml
endif

docker-compose-files=-f ${dockerfile}

help:  ## Show help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

build:  ## Build docker image(s) [args="<name_of_image>"].
	@docker compose $(docker-compose-files) build ${args}

build-no-cache:  ## Build docker image(s) without cache [args="<name_of_image>"].
	@docker compose $(docker-compose-files) --compatibility build ${args} --no-cache

up:  ## Start docker container(s) [args="<name_of_service>"].
	@docker compose ${docker-compose-files} up ${args} -d

down:  ## Stop docker container(s) [args="<name_of_service>"].
	@docker compose ${docker-compose-files} down ${args}

stop:  ## Stop docker containers [args="<name_of_service>"].
	@docker compose ${docker-compose-files} stop ${args}

restart:  ## Restart service(s) [args="<name_of_service>"].
	@docker compose ${docker-compose-files} restart ${args}

test:  ## Run tests, with the coverage report (branches included).
	@docker exec grinning_cat_core sh -c 'uv run python -m coverage run -m pytest --color=yes -vvv -W ignore --disable-warnings ${args}; status=$$?; uv run python -m coverage report -m --skip-covered; exit $$status'

install: ## Update the local virtual environment with the latest requirements.
	@uv sync --link-mode=copy --frozen --no-install-project --no-upgrade --no-cache
	@python3 scripts/install_plugin_requirements.py
	@python3 scripts/hash_plugin_requirements.py
	@uv cache clean
	@pip cache purge

update: ## Update and compile requirements for the local virtual environment.
	@uv sync --upgrade --link-mode=copy --no-install-project --no-cache
	@uv cache clean
	@pip cache purge
	@rm -rf *.egg-info

check: ## Check requirements for the local virtual environment.
	@uv sync --check

migrate:  ## Apply database migrations
	@docker exec -it grinning_cat_core uv run python migrations/manage_migrations.py upgrade head

make-migration:  ## Create the migration file after changing the models. Argument `args` is mandatory as the comment of the migration.
	@if [ -z "${args}" ]; then \
		echo "Error: 'args' is required for 'run'. Example: make make-migration args=\"The comment to the migration\"" >&2; \
		exit 1; \
	fi
	@docker exec -it grinning_cat_core uv run python migrations/manage_migrations.py revision -m "${args}"

BUILD_PROXY ?= http://10.42.0.1:3142
BUILD_NO_PROXY ?= localhost,127.0.0.1,::1,.intranet

dhi:  ## Build the hardened image (Dockerfile:dhi), pulling the latest DHI base images.
	docker buildx build . -f Dockerfile:dhi -t grinning-cat-core:dhi-full --pull \
		--build-arg HTTP_PROXY=$(BUILD_PROXY) \
		--build-arg HTTPS_PROXY=$(BUILD_PROXY) \
		--build-arg NO_PROXY=$(BUILD_NO_PROXY)
dev:  ## Build the hardened image (Dockerfile:dhi) tagged for development, pulling the latest DHI base images.
	docker buildx build . -f Dockerfile:dhi -t grinning-cat-core:dev-full --pull \
		--build-arg HTTP_PROXY=$(BUILD_PROXY) \
		--build-arg HTTPS_PROXY=$(BUILD_PROXY) \
		--build-arg NO_PROXY=$(BUILD_NO_PROXY)

##@ Plugin-enabled images

# Host directory with the plugins bind-mounted at runtime. It is outside the
# repo build context, so it is passed to BuildKit as a named build context.
PLUGINS_DIR ?= /root/DEV/data/grinning/plugins
# Image tag produced by `build-with-plugins` (the tag the compose file deploys).
IMAGE ?= grinning-cat-core:dev-full
# Dockerfile used by `build-with-plugins` (override to build the hardened variant).
DOCKERFILE_WITH_PLUGINS ?= Dockerfile

# Both targets pass:
#   --build-context plugins=$(PLUGINS_DIR)  read-only access to the host plugins dir;
#   --build-arg PLUGINS_CTX=plugins         select it as the COPY source in the Dockerfile;
#   --build-arg INSTALL_PLUGIN_DEPS=1       enable the optional dependency install.
# Without the named context (plain `make build`), PLUGINS_CTX defaults to an empty
# stage, so nothing is copied and no plugin dependencies are installed.
#
# WARNING — the venv volume SHADOWS the baked venv:
#   compose mounts ./data/grinning/venv:/app/.venv over the image's /app/.venv, so
#   the dependencies installed by these targets are invisible at runtime until the
#   volume is re-seeded from the new image (or the mount is removed). From /root/DEV:
#     docker rm -f gc-venv-seed 2>/dev/null; \
#     docker create --name gc-venv-seed $(IMAGE) && \
#     docker cp gc-venv-seed:/app/.venv/. ./data/grinning/venv/ && \
#     docker rm gc-venv-seed
build-with-plugins:  ## Build an image from Dockerfile with host plugin deps (requires re-seeding ./data/grinning/venv:/app/.venv).
	docker buildx build --load \
		--build-context plugins=$(PLUGINS_DIR) \
		--build-arg PLUGINS_CTX=plugins \
		--build-arg INSTALL_PLUGIN_DEPS=1 \
		-f $(DOCKERFILE_WITH_PLUGINS) -t $(IMAGE) .

dev-with-plugins:  ## Build $(IMAGE) (Dockerfile:dhi) installing host plugin deps (matches the compose image).
	docker buildx build --load . -f Dockerfile:dhi -t $(IMAGE) --pull \
		--build-context plugins=$(PLUGINS_DIR) \
		--build-arg PLUGINS_CTX=plugins \
		--build-arg INSTALL_PLUGIN_DEPS=1 \
		--build-arg HTTP_PROXY=$(BUILD_PROXY) \
		--build-arg HTTPS_PROXY=$(BUILD_PROXY) \
		--build-arg NO_PROXY=$(BUILD_NO_PROXY)

