# DARE Biodiversity Monitoring Database

------------------------------------------------------------------------
# Introduction and aims

The data provided is from the DigiFarm project for which the key aims are listed for context below:

-   Develop a digitally enabled network which will monitor native flora
    and fauna to inform sustainable agricultural practices. A unique
    combination of methods will be used: We will test new methods in
    camera trapping and acoustic recorders (birds and bats) in
    quantifying on-farm biodiversity and develop spatial models to
    identify biodiversity hotspots.
-   The project is part of a larger initiative that aims to develop a
    digitally enabled network which will simultaneously monitor crop and
    animal production, and soil and ecosystem health.
-   Building on current investments in Narrabri, we shall build a
    physical and virtual DigiFarm hub and satellite farm network for
    north-west NSW providing digital dashboards of ‘health, production
    and social’ metrics.
-   We will create an education platform at Narrabri for farmers,
    agribusiness, schools, environmental stakeholders to experience the
    latest ag-innovation thinking.

Here we present a summary of the DigiFarm acoustic recorder ecoacoustic indices from NW NSW. You can see some initial plots that have been produced in the `Biodiversity Ecoacoustics Dataset - Readme Appendix.html' file.

------------------------------------------------------------------------
# Data collection

A total of 20 Plots are located across the property in a grid formation. Recorders are mounted on star pickets. Frontier Labs BAR recorders are set to record for 10 min on the hour for every hour at a sample rate 48kHz. The biodiversity project was set up in 14 and 15th Nov 2019. 

![Location of DigiFarm plots.](./imgs/Llara_biodiversity_equipment.png) 

Set up in 14/15th Nov 2019 and first test download a few days later. 

------------------------------------------------------------------------
# Raw data Proccessing

## Date/Time
Data are provided in UTC time. It can easily be converted back to instrument time (GMT+10) with:
```python
instData['timeStart'] = pd.to_datetime(instData['timeStart'], utc=True)
instData['timeStart'] = instData['timeStart'].dt.tz_convert('Etc/GMT-10') 
# or to local time with daylight savings with .tz_convert('Australia/Sydney')
```

## Indices  
  
From: Sueur J. (2018) *Sound Analysis and Synthesis with R*. Springer International Publishing, Cham.  

*and*  

Eldridge A., Guyot P., Moscoso P., Johnston A., Eyre-Walker Y. & Peck M. (2018) Sounding out ecoacoustic metrics: Avian species richness is predicted by acoustic indices in temperate but not tropical habitats. *Ecol. Indicators* 95, 939-52.  


**ADI** = acoustic_diversity(). uses, as Hf does, the Shannon entropy on the spectral content. The ADI index computes the Short-time Fourier transform (STDFT), cuts the STDFT into a determined number of bins, selects the relative amplitude of each bin that is above a dB threshold, and applies the Shannon entropy index on these selected values. By default the frequency bandwidth between 0 and 10 kHz is split into 10 bins, and the dB threshold is set to -50 dB.  Originally developed to assess habitats along a gradient of degradation under the assumption that ADI and AEI would be respectively positively and negatively associated with habitat status as the distribution of sounds became more even with increasing diversity
(Villanueva-Rivera et al., 2011). ADI was shown to increase from agricultural to forested sites; AEI was shown to decrease over the same gradient, as expected. Positive association between ADI and avian species richness has been reported in the savannas of central Brazil (Alquezar and Machado, 2015).

**BI** = bioacoustic_index() #assess relative avian abundance. The index consists in computing the dB mean spectrum and in calculating the area under the curve between two frequency limits.   

**NDSIs** = normalized difference soundscape index, NDSI, aims at estimating the level of anthropogenic disturbance on the soundscape by computing the ratio of human-generated (anthropophony) to biological (biophony) acoustic components.  

**AEI** = acoustic_evenness(). proceeds the same first step as ADI but computes the Gini coefficient which is a measure of distribution inequality. Originally developed to assess habitats along a gradient of degradation under the assumption that ADI and AEI would be respectively positively and negatively associated with habitat status as the distribution of sounds became more even with increasing diversity
(Villanueva-Rivera et al., 2011). Negative, if weak, associations between AEI and biocondition 

**H*f*** = spectral entropy index, Hf , follows the same principle as Ht but works in the frequency domain. Hf is actually the Shannon evenness of the frequency spectrum, usually the mean spectrum. The index is constrained between 0 and 1 by transforming the frequency spectrum into a probability mass function as done for the Hilbert amplitude envelope with Ht. **Increases with species diversity**.

**ACI** =  acoustic_complexity(). aims at measuring the complexity of STDFT matrix giving more importance to sounds that are modulated in amplitude and, hence, reducing the importance of sound with a rather constant amplitude as anthropogenic noise may have. The main principle of the ACI is to compute the average absolute amplitude difference between adjacent cells of the STDFT matrix in each frequency bin, that is, in each row of the STDFT matrix. ACI has been reported to correlate significantly with the number of avian vocalisations.

## Weather data
Some weather data is provided in `/data/weather_station/` but this does not cover the entire period with acoustic sensors (only 03-2021 to 03-2022).

You could also look to external sources for data. A good source of daily data is SILO:

<a href="https://www.longpaddock.qld.gov.au/silo/point-data/">https://www.longpaddock.qld.gov.au/silo/point-data/</a>

## GIS data
There are some shapefiles provided which have vegetation, farm layout and other information.
Some of this data is already visualised in figures in the `/imgs/` folder.

## Field notes data

This has been collected using EpiCollect5. This includes information on planting/harvesting. This data is available in `data/epicollect/equipment_logs.csv`. **However**, this is raw data - to be processed and used with caution. Feel free to discuss with Josh if you would like to use it.