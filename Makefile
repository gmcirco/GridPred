# Makefile for running GridPred quickly

PYTHON := uv run python
SCRIPT := main.py

# Default arguments for quick testing
CRIME_CSV := input/hartford_robberies.csv
REGION_SHP := input/hartford.shp
FEATURES_CSV := input/hartford_pois.csv
TIMEVAR := year
FEATVAR := types
INPUT_CRS := 3508
PROJECTED_CRS := 3508
GRID_SIZE := 300

# Default run with ALL inputs present
run:
	$(PYTHON) $(SCRIPT) \
		$(CRIME_CSV) \
		--crime_time_variable $(TIMEVAR) \
		--input_region_path $(REGION_SHP) \
		--input_features_path $(FEATURES_CSV) \
		--features_names_variable $(FEATVAR) \
		--input_crs $(INPUT_CRS) \
		--projected_crs $(PROJECTED_CRS) \
		--do_projection \
		--grid_size $(GRID_SIZE)

# Run WITHOUT region
run-noregion:
	$(PYTHON) $(SCRIPT) \
		$(CRIME_CSV) \
		--crime_time_variable $(TIMEVAR) \
		--input_features_path $(FEATURES_CSV) \
		--features_names_variable $(FEATVAR) \
		--input_crs $(INPUT_CRS) \
		--projected_crs $(PROJECTED_CRS) \
		--do_projection \
		--grid_size $(GRID_SIZE)

# Run WITHOUT spatial predictors
run-nopreds:
	$(PYTHON) $(SCRIPT) \
		$(CRIME_CSV) \
		--crime_time_variable $(TIMEVAR) \
		--input_region_path $(REGION_SHP) \
		--input_crs $(INPUT_CRS) \
		--projected_crs $(PROJECTED_CRS) \
		--do_projection \
		--grid_size $(GRID_SIZE)

# Install dependencies with uv
install:
	uv sync

# Clean artifacts if needed later
clean:
	rm -f *.log
	rm -rf __pycache__

.PHONY: run run-noproj install clean
