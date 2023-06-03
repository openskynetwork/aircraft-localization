# aircraft-localization
Code and data from the OpenSky Network - Cyber-Defence Campus aircraft localization competition hosted by AICrowd. 
The competition and all details can be found at https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition

The first round ran from 2020-06-15 08:00 until 2020-07-31 23:59.

The (reloaded) second round ran from 2020-12-01 09:00 until 2021-01-31 18:00.


# Training & competition data
Round 1:
All 7 training data sets and the competition data set for round 1 are available at https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition/dataset_files


Round 2:
All training data sets are available at https://www.dropbox.com/s/7hu8y3kutpx4bgq/round2_training.zip?dl=0
The competition data set is available at: https://www.dropbox.com/s/585fhm9e7dqrygq/round2_competition_data.zip?dl=0

Please note that all code is provided as is. It has not been unified or is necessarily ready-to-run as provided.


# Ground Truth and Evaluation Code

The ground truth and the submissions of the winners have been made available. The evaluators used for round 1 and 2 are also published here.


# Code of the winners
The code of the winning teams is licensed under the GNU GPLv3 license as required by the rules of the competitions: https://www.aicrowd.com/challenges/cyd-campus-aircraft-localization-competition/challenge_rules

Round 1:
The winning teams were all very close, using traditional multilateration techniques as their basis.

Place	Team              	RMSE [m]

1st		richardalligier   	25.02

2nd		ZAViators         	25.817 	

3rd		ck.ua             	26.214 

4th		sergei_markochev		33.544 

5th		dataWizard        	59.467 	


Round 2: 
In the second round, we used a hidden score for ranking. As the task was more complex, the results were slightly worse than round 1, however, the participants were still very close with a lot of outstanding performances.

Place   Team              	RMSE [m]

1st     sergei_markochev   	81.890

2nd 		ck.ua         			98.370	

3rd     nwpu.i4Sky          154.574

4th     ZAViators		 				171.663

5th	    dataWizard        	2392.535


# Publications
Besides the documentations of the code, the 2nd place competitors of round 1 have provided a detailed writeup for the 8th OpenSky Symposium. 

Video: https://www.youtube.com/watch?v=msBtF0Swfn4 (Benoit Figuet – Combined multilateration with machine learning for enhanced aircraft localization)

Publication: Benoit Figuet, Michael Felux and Raphael Monstein. "Combined multilateration with machine learning for enhanced aircraft localization." In Proceedings of the 8th OpenSky Symposium 2020. November 2020. [Open Access link: https://www.mdpi.com/2504-3900/59/1/2]


Before the extended round 2, we have released a full discussion of the training and test datasets:

Preprint: Matthias Schäfer, Martin Strohmeier, Mauro Leonardi and Vincent Lenders. LocaRDS: A Localization Reference Data Set. arXiv preprint arXiv:2012.00116. December 2020. https://arxiv.org/abs/2012.00116 

Publication: https://www.mdpi.com/1424-8220/21/16/5516

Dataset: https://zenodo.org/record/4739276


The first place of the second round has also provided a publication and video in addition to the documentation in this repository for the 9th OpenSky Symposium:

Video: https://www.youtube.com/watch?v=hB4QrLhoWgE (Sergei Markochev - Aircraft localization with nanosecond precision from distributed receivers)

Publication:  Markochev S. Aircraft Localization Using ATC Data with Nanosecond Precision from Distributed Crowdsourced Receivers. Engineering Proceedings. 2021; 13(1):12. https://doi.org/10.3390/engproc2021013012 


Finally, we have a published a write-up, detailing the full history of this competition:

Preprint: Martin Strohmeier, Mauro Leonardi, Sergei Markochev, Fabio Ricciato, Matthias Schäfer, Vincent Lenders. Improving Aircraft Localization: Experiences and Lessons Learned from an Open Competition. https://arxiv.org/abs/2209.13669

Publication: Strohmeier, M., Leonardi, M., Markochev, S., Ricciato, F., Schäfer, M. and Lenders, V., 2023. In Pursuit of Aviation Cybersecurity: Experiences and Lessons From a Competitive Approach. IEEE Security & Privacy. https://ieeexplore.ieee.org/abstract/document/10109762


