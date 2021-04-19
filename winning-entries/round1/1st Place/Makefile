.PRECIOUS: %.sensorsparams %.aircraftpos %.aircraftposts %.splineaircraftpos %.splineaircraftposts %.splineaircraftposwithfeature %.splineaircraftpostswithfeature %.filter


PBNAME = round1_competition
COVERAGE = 0.5

# SENSORSPARAMS PARAMETERS
LR_SENSORS = 5e2
BATCH_SIZE_SENSORS = 100000

# AIRCRAFTPOS PARAMETERS
LR_AIRCRAFTPOS = 0.1
CLOSE_SENSOR = 15000
CONTINUITY = 1

# SPLINEAIRCRAFTPOS PARAMETERS
THR_ERROR = 20
SMOOTH = 0.001

# FILTER PARAMETERS
NJOBS_TRAIN_FILTER = 70
N_ITER = 100
CLASS= --classif
LATLON= --latlon
DBARO= --dbaro

FILE = ./Data/$(PBNAME)/$(PBNAME)
FILE_SENSORSPARAMS = $(FILE)-$(LR_SENSORS)-$(BATCH_SIZE_SENSORS)
FILE_AIRCRAFTPOS = $(FILE_SENSORSPARAMS)?$(CONTINUITY)-$(CLOSE_SENSOR)-$(LR_AIRCRAFTPOS)
FILE_SPLINEAIRCRAFTPOS = $(FILE_AIRCRAFTPOS)?$(THR_ERROR)-$(SMOOTH)
FILE_FILTER = $(FILE_SPLINEAIRCRAFTPOS)?$(CLASS)-$(LATLON)-$(DBARO)
FILE_FILTEREDRESULT = $(FILE_FILTER)?$(COVERAGE)


all: result filteredresult

result: $(FILE_SPLINEAIRCRAFTPOS).result

filteredresult: $(FILE_FILTEREDRESULT).filteredresult



$(FILE_SENSORSPARAMS).sensorsparams:
	python	trainpytorchlightning.py --max_epochs 10 sensorsparams --lr $(LR_SENSORS) --pbname $(PBNAME) --batch_size $(BATCH_SIZE_SENSORS) --save_sensorsparams $@ > logsensorsparams$(notdir $(FILE_SENSORSPARAMS))


$(FILE_AIRCRAFTPOS).aircraftpos: $(FILE_SENSORSPARAMS).sensorsparams
	python trainpytorchlightning.py --max_epochs 30 aircraftpos --pbname $(PBNAME) --close_sensor $(CLOSE_SENSOR) --continuity $(CONTINUITY) --load_sensorsparams $^  --lr $(LR_AIRCRAFTPOS) --save_aircraftpos $@ > logaircraftpos$(notdir $(FILE_AIRCRAFTPOS))

$(FILE_AIRCRAFTPOS).aircraftposts: $(FILE_SENSORSPARAMS).sensorsparams
	python trainpytorchlightning.py --max_epochs 30 aircraftpos --pbname $(PBNAME) --close_sensor $(CLOSE_SENSOR) --continuity $(CONTINUITY) --load_sensorsparams $^  --lr $(LR_AIRCRAFTPOS) --save_aircraftpos $@ --ts > logaircraftposts$(notdir $(FILE_AIRCRAFTPOS))


$(FILE_SPLINEAIRCRAFTPOS).splineaircraftpos: $(FILE_AIRCRAFTPOS).aircraftpos
	python splineaircraftpos.py --inputfile $^ --thr_error $(THR_ERROR) --smooth $(SMOOTH) --pbname $(PBNAME) --outputfile $@ > logsplineaircraftpos$(notdir $(FILE_SPLINEAIRCRAFTPOS))

$(FILE_SPLINEAIRCRAFTPOS).splineaircraftposts: $(FILE_AIRCRAFTPOS).aircraftposts
	python splineaircraftpos.py --inputfile $^ --thr_error $(THR_ERROR) --smooth $(SMOOTH) --pbname $(PBNAME) --outputfile $@ --ts > logsplineaircraftposts$(notdir $(FILE_SPLINEAIRCRAFTPOS))


$(FILE_SPLINEAIRCRAFTPOS).splineaircraftposwithfeature: $(FILE_SPLINEAIRCRAFTPOS).splineaircraftpos
	python splineaircraftposwithfeature.py --inputfile $^ --pbname $(PBNAME) --outputfile $@ > logsplineaircraftposwithfeature$(notdir $(FILE_SPLINEAIRCRAFTPOS))

$(FILE_SPLINEAIRCRAFTPOS).splineaircraftpostswithfeature: $(FILE_SPLINEAIRCRAFTPOS).splineaircraftposts
	python splineaircraftposwithfeature.py --inputfile $^ --pbname $(PBNAME) --outputfile $@ > logsplineaircraftpostswithfeature$(notdir $(FILE_SPLINEAIRCRAFTPOS))


$(FILE_FILTER).filter: $(FILE_SPLINEAIRCRAFTPOS).splineaircraftpostswithfeature
	python learnfilter.py --inputfile $^ --n_jobs $(NJOBS_TRAIN_FILTER) --n_iter $(N_ITER) --outputfile $@ $(LATLON) $(DBARO) $(CLASS) > logfilter$(notdir $(FILE_FILTER))


$(FILE_SPLINEAIRCRAFTPOS).result: $(FILE_SPLINEAIRCRAFTPOS).splineaircraftpos
	python writeprediction.py --inputfile $^ --pbname $(PBNAME)  --outputfile $@ > logresult$(notdir $(FILE_SPLINEAIRCRAFTPOS))


$(FILE_FILTEREDRESULT).filteredresult: $(FILE_SPLINEAIRCRAFTPOS).splineaircraftposwithfeature $(FILE_FILTER).filter
	python writeprediction.py --inputfile $< --pbname $(PBNAME) --model $(word 2, $^) --coverage $(COVERAGE)  --outputfile $@ > logfilteredresult$(notdir $(FILE_FILTEREDRESULT))
