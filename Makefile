DUCKDB_DIR ?= build/deps/duckdb
NATIVE_BINARY := build/native/nba-retrieve

.PHONY: check demo app native-deps native check-native
check:
	ruff check .
	pytest -q
demo:
	nba-predictor --root data/synthetic --artifacts artifacts/synthetic --reports reports/synthetic demo
app:
	streamlit run app.py

native-deps:
	sh cpp/fetch_duckdb.sh "$(DUCKDB_DIR)"

native: $(NATIVE_BINARY)

$(NATIVE_BINARY): cpp/retrieve.cpp Makefile
	@mkdir -p build/native
	duckdb_dir="$$(cd "$(DUCKDB_DIR)" && pwd)" && \
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -O2 -Wall -Wextra -Wpedantic \
		-I"$$duckdb_dir" "$<" -L"$$duckdb_dir" -lduckdb \
		-Wl,-rpath,"$$duckdb_dir" $(LDFLAGS) -o "$@"

check-native: native
	PYTHONPATH=src NBA_RETRIEVAL_BINARY="$$PWD/$(NATIVE_BINARY)" pytest -q
