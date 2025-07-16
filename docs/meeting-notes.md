# 2025 EAS summer ML working group meeting notes

# 2025-07-16
- Spencer has updated the Github repo posting all materials requested to date, including these meeting notes
- Michelle shared her NN created w/ ChatGPT to identify the Hadley cell shared edge from the streamfunction
  - It's available at `projects/michelle`
  - It's a fully connected network w/ 2 hidden layers
  - It works!
  - She's in the process of going through line by line to ~fully understand the code that ChatGPT generated
  - She had to do some tweaking/iterating, but mostly ChatGPT was able to create the whole thing easily
  - (Spencer editorial: totally legit approach!  So long as you do the subsequent work, like Michelle is, of going through and understanding the code.)
  - Potential next steps (after Michelle completes that process): predict the descending edges too; play w/ architecture; explore techniques to avoid overfitting
- Misc. things Haochang brought up to be on the lookout for:
  - [tensorboard](https://www.tensorflow.org/tensorboard): tools for visualizing ML models
  - "gradient explosion" e.g. [here](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/)
  - overfitting
    - Plot training loss and validation loss vs. epoch
	- "weight decay" strategy: when defining your optimizer in pytorch via e.g. `optimizer = optim.Adam(model.parameters(), lr=1e-3)
`, you can add as "decayrate"; a good starting point value is ~1/10 of the learning rate.


# 2025-07-09
## Agenda
- very brief check in of where people are at
- live demo of functional ML model (handwritten digits reader)
- more work time on input data preprocessing


## Status updates
### Spencer
- model architecture/problem still not 100% locked in
-  thinks input data is properly split into test/validate/train chunks

### Nick
- lit review on appropriate models; foundational models for remote sensing; terramind IBM (written in geotorch, available on huggingface)
- so take this foundational model and fine-tune it for his tasks

### Shreya
- using pytorch docs to better understand some fundamentals
- using Haochang's provided tutorial as a template for her classification problem

### Michelle
- classic simple feedforward NN for her regression task
- data loaded and successful(?) train/validate/test split

## Haochang
- Applied existing resnet model to his AEW developer/non-developer problem, doesn't get better results than his own custom model
- resnet built into pytorch; can call it in one line; he'll demo that for us next time

## Angela
- Doing linear regression first before digging into building an ML model 
  - Spencer note: this is a great idea!  Make sure you have a legit baseline before trying to construct a complicated ML model


# 2025-07-02
## Goal going into this week
Have your conda environment with PyTorch setup properly: you can activate the environment, open a Jupyter session, and call `import pytorch` without error

## Agenda
1. (Spencer) primer on **classes** in Python and object-oriented programming (OOP), since the PyTorch workflow requires creating simple classes
2. (Haochang) **Data preprocessing**: overview of the various “massaging” steps required before feeding your input data into an ML model, including load balancing, the train/validate/test split, and other things.
3. Work time

## Goal going from this week to next week
Get your data preprocessing all handled, so that we can then turn to constructing the ML models architectures next time (or maybe the time after, if this data preprocessing step takes longer than anticipated)

# 2025-06-25
## Agenda
- Ultra-brief intros/1-sentence project summary from those who were there last week
- Somewhat longer description of project ideas from all others
- Python environment setup for this project via the env.yaml file we provide
- Free work time for each of us

## Project ideas
### Greg
- NYC heatwaves and associated synoptic patterns
  - Negative PNA phase preceding?
  - k-means clustering? (unsupervised problem)
- Software stack
  - SOM: mini-som package
  - scikit-learn good for clustering
  - scipy

### Haochang
- Distinguish Africna Easterly Waves (AEWs): those that develop into tropical cyclones (TCs), "developers", vs. those that do not, "non-developers"
- Already has a pretty serious model built up, is going to refine it even further, including using full timeseries of input fields

### Spencer
- Convective buoyancy vs. precip for NYC

### Grace

### Jimmy
- 3 potential ideas
  1. CS student building *diffusion* model of time-evolving ETC storms
     - working already on SLP, now trying to get it to work on winds
     - its output is all storm-centered (so distinct from Jimmy's past synth ETC statistical model stuff, could eventually combine them)
     - end-goal: connect winds to impacts
  2. Carolien: 
     - hourly precip NYC timeseries
     - hierarchical clustering to sort time evolution of extreme events
     - can we distinguish long continuous vs. intense bursty rain
     - It's already up and running (via scipy), working through the kinks now
  3. Ty: flash floods in NYC: has dates and associated upper-level wind and Z fields
     - self-organized maps (SOM)
     - NWS has already IDed the features they associate with flash floods (via expert assessment by forecasters), and see if the SOM-generated ones line up

### Nick Steiner
- background: imaging radar to look at open water and wetlands
  - NASA NISAR, NOAA operational imaging-based flood status: maps of best-estimate flooding
  - prev work: integrate NASA's dynamic SAR (Synthetic Aperture Radar) product into NOAA's workflow
    - existing product uses *fuzzy logic*
    - and can be unskillful and take a long time
- project idea
  - emulate the NASA-generated product as an ML model, to speed it up
  - fine tune the existing giant satellite data products to his applications
  - plus maybe: integrate the optical data
- software stack
  - pytorch
  - torch-geo: https://github.com/microsoft/torchgeo 

## environment 
- shreya: 
  - had to bump python down to 3.10.18
  - numpy: <v2
  - on 2018 macbook pro w/ intel CPU and GPU

- Michelle
  - on windows laptop
  - (the environment file that Haochang shared via email works for her)

- Nick: on a PC but using WSL




# 2025-06-10 meeting
## Project ideas
### Shreya Keshri
- Distinguish between MRGs that are associated with extratropical PV intrusions vs. not associated with PV intrusions
- Dataset: 40 years record, pre-identified ET vs. non-ET, focus on E. Pacific, ERA5
- Potential input layer features
  - meridional wind
  - geopotential height
  - (me: maybe PV field itself)
### Angela Padilla
- from tidal gauges, predict tidal level from the preceding timesteps, e.g. 2pm each day what might you expect
- NOAA publicly available tide gauge data at Bridgeport
- 3 yrs of data on hand so far
### Michelle Wagner
- identification of the Hadley cell ascending and descending edges
- input: ERA5 overturning streamfunction
  - Descending and ascending edges already identified for each calendar month, Jan 1979 through Dec 2024
- end goal: CNN but use conventional NN to start
### Edda Hobuss
- IGRA radiosonde data
- use e.g. zonal wind to identify the month of the year in a monsoonal location where there's a strong seasonal cycle
### Kyle MacDonald
- Two project ideas!
- **winner** Idea #1: wetlands and inundated forests
  - dataset
    - remote sensed surface inundation of forested landscapes in tropical wetlands
    - inundation maps: water, inundated forest, non-inundated, etc.
    - imaging radar: ~20-50 m resolution: high spatial but poorer temporal resolution
    - CYGNIS(sp?): constellation that does frequent overpasses but is spatially narrow
  - Model: can we predict the inundation maps at higher frequency
- Idea #2
  - suite of satellite images from various sensors
  - Peru, multi-band
  - land cover as relates to the underlying geology
  - end goal: be able to tell e.g. geologists and paleoecologists where there might be interesting stuff
