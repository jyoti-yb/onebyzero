CFG ?= configs/synthetic.yaml
all:      ; monsoonpp all -c $(CFG)
build:    ; monsoonpp build -c $(CFG)
train:    ; monsoonpp train -c $(CFG)
evaluate: ; monsoonpp evaluate -c $(CFG)
export-demo: ; monsoonpp export-demo -c $(CFG) --bundle dist/demo-bundle
serve:    ; monsoonpp serve --bundle dist/demo-bundle
test:     ; pytest -q -W ignore
.PHONY: all build train evaluate export-demo serve test
