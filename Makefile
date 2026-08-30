.DEFAULT_GOAL := debug

ifeq ($(VULKAN_SDK),)
$(error VULKAN_SDK is not set. Install the Vulkan SDK and the set environment variable.)
endif

NAME := odin-pt
ODIN := odin

ifeq ($(OS),Windows_NT)
SHELL        := cmd.exe
.SHELLFLAGS  := /c
EXE          := .exe
LINK_FLAGS   := -extra-linker-flags:"/LIBPATH:$(VULKAN_SDK)\Lib"
ASSIMP_SRC   := lib/assimp/bin/assimp-vc143-mt.dll
ASSIMP_BIN   := assimp-vc143-mt.dll
COPY          = copy /Y $(subst /,\,$<) $(subst /,\,$@) >nul
MKDIR         = mkdir $(subst /,\,$@)
CLEAN         = if exist bin rmdir /s /q bin
RUN           = $(subst /,\,$<)
else
EXE          :=
LINK_FLAGS   := -extra-linker-flags:"-L$(VULKAN_SDK)/lib -Wl,-rpath,\$$ORIGIN -Wl,-rpath,$(VULKAN_SDK)/lib"
ASSIMP_SRC   := lib/assimp/bin/libassimp.so
ASSIMP_BIN   := libassimp.so.6
COPY          = cp $< $@
MKDIR         = mkdir -p $@
CLEAN         = rm -rf bin
RUN           = ./$<
endif

ODIN_SRC    := $(wildcard src/*.odin) $(wildcard src/gpu/*.odin)

DEBUG_EXE   := bin/debug/$(NAME)$(EXE)
RELEASE_EXE := bin/release/$(NAME)$(EXE)

.PHONY: debug release run run-release clean

debug:   $(DEBUG_EXE)
release: $(RELEASE_EXE)

run: $(DEBUG_EXE)
	$(RUN)

run-release: $(RELEASE_EXE)
	$(RUN)

$(DEBUG_EXE): $(ODIN_SRC) | bin/debug bin/debug/$(ASSIMP_BIN)
	$(ODIN) build src -collection:lib=lib -out:$@ -debug -vet $(LINK_FLAGS)

$(RELEASE_EXE): $(ODIN_SRC) | bin/release bin/release/$(ASSIMP_BIN)
	$(ODIN) build src -collection:lib=lib -out:$@ -o:speed -vet $(LINK_FLAGS)

bin/debug/$(ASSIMP_BIN): $(ASSIMP_SRC) | bin/debug
	$(COPY)

bin/release/$(ASSIMP_BIN): $(ASSIMP_SRC) | bin/release
	$(COPY)

bin/debug bin/release:
	$(MKDIR)

clean:
	-$(CLEAN)
