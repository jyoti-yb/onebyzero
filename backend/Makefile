CFG ?= configs/synthetic.yaml
all:      ; monsoonpp all -c $(CFG)
build:    ; monsoonpp build -c $(CFG)
train:    ; monsoonpp train -c $(CFG)
evaluate: ; monsoonpp evaluate -c $(CFG)
serve:    ; monsoonpp serve -c $(CFG)
test:     ; pytest -q -W ignore
.PHONY: all build train evaluate serve test
