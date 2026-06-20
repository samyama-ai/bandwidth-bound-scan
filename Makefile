# bandwidth-bound-scan — roofline micro-benchmark harness
# Native build: -O3 -march=native so the compiler auto-vectorizes (the FastLanes premise).
CC      ?= cc
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra
LDFLAGS ?=

SRC := src/roofline.c src/codecs.c src/scanbench.c
HDR := src/common.h src/roofline.h src/codecs.h

.PHONY: all test clean beta sweep zone repro

all: scanbench test_codecs

scanbench: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

test_codecs: tests/test_codecs.c src/codecs.c src/codecs.h src/common.h
	$(CC) $(CFLAGS) tests/test_codecs.c src/codecs.c -o $@ $(LDFLAGS)

test: test_codecs
	./test_codecs

beta: scanbench
	./scanbench beta

# one-command repro: build, test, measure beta, run both sweeps into results/
repro: scanbench test_codecs
	./test_codecs
	mkdir -p results
	./scanbench beta | tee results/beta_$${BBS_MACHINE:-host}.txt
	./scanbench sweep  $${BBS_N:-67108864} $${BBS_REPEATS:-7} > results/sweep_$${BBS_MACHINE:-host}.csv
	./scanbench zone   $${BBS_N:-67108864} $${BBS_REPEATS:-7} > results/zone_$${BBS_MACHINE:-host}.csv
	./scanbench select $${BBS_N:-67108864} $${BBS_REPEATS:-7} > results/select_$${BBS_MACHINE:-host}.csv
	@echo "results written to results/ — now run: python analyze.py"

clean:
	rm -f scanbench test_codecs
