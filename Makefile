# Makefile for running GridPred quickly

PYTHON := uv run python
SCRIPT := main.py

# Default arguments for quick testing
CRIME_CSV := /home/gmcirco/Documents/Projects/data/crime_2019-2023_v2.csv
REGION_SHP := /home/gmcirco/Documents/Projects/data/malmo_shapefiles/DeSo_Malmo.shp
FEATURES_CSV := /home/gmcirco/Documents/Projects/data/features_malmo.csv
TIMEVAR := yearvar
FEATVAR := type
INPUT_CRS := 4326
PROJECTED_CRS := 3006
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
