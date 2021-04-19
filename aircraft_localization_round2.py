import pandas as pd
import numpy as np
import sklearn

class AIcrowdEvaluator:
    def __init__(self, **kwargs):
        """
        `round` : Holds the round for which the evaluation is being done. 
        can be 1, 2...upto the number of rounds the challenge has.
        Different rounds will mostly have different ground truth files.
        """
        self.answer_file_path = "./data/round2_ground_truth.csv"
        self.publicdata_file_path    = "./data/round2_nonhiddenevaluationids.csv"
        self.round = 2

        # Return haversine distance
    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6372800    # Earth radius in meters
        
        phi1, phi2 = np.radians(lat1), np.radians(lat2) 
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + \
                np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        
        return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Return RMSE
    def _rmse(self, haversine_distance):
        return np.sqrt((haversine_distance ** 2).mean())

    # Return error if less than 70% predicted
    def _checkcoverage(self, submission):
        return 1 - ((len(submission.latitude) - submission.latitude.count()) / len(submission.latitude)) 
        
                             
    def _evaluate(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
            - submission_file_path : local file path of the submitted file
            - aicrowd_submission_id : A unique id representing the submission
            - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]     
        aicrowd_submission_id = client_payload["aicrowd_submission_id"]
        aicrowd_participant_uid = client_payload["aicrowd_participant_id"]
        
        submission = pd.read_csv(submission_file_path)
        gt_data = pd.read_csv(self.answer_file_path)
        nonhidden_data = pd.read_csv(self.publicdata_file_path)
        submission_public = submission[submission.id.isin(nonhidden_data.id)]
        gt_data_public = gt_data[gt_data.id.isin(nonhidden_data.id)]

        
        # Or your preferred way to read your submission

        """
        Do something with your submitted file to come up
        with a score and a secondary score.

        If you want to report back an error to the user,
        then you can simply do :
            `raise Exception("YOUR-CUSTOM-ERROR")`

         You are encouraged to add as many validations as possible
         to provide meaningful feedback to your users
        """
        
        if not len(submission.index) == len(gt_data.index):
                raise Exception("Wrong format, number of rows needs to be: " + str(len(gt_data.index)))
                
        if not len(submission.columns) == len(gt_data.columns):
                raise Exception("Wrong format, number of columns needs to be: " + str(len(gt_data.columns)))

        coverage = self._checkcoverage(submission)
        
        if coverage < 0.7:
                raise Exception("Less than 70% coverage, submission not eligible. Coverage: " + str(100*coverage) + "%")
                
        # Get haversine distances        
        distances = self._haversine(gt_data.latitude,gt_data.longitude,submission.latitude,submission.longitude)
        distances_public    = self._haversine(gt_data_public.latitude,gt_data_public.longitude,submission_public.latitude,submission_public.longitude)
        
        # Sort, remove NaNs
        sorted_distances = distances.sort_values().dropna()
        sorted_distances_public = distances_public.sort_values().dropna() 
        
        # Truncate worst 10%, get RSME
        cutoff = int(np.round(sorted_distances.size*0.9))
        eval_distances = sorted_distances[0:cutoff]
        
        cutoff_public = int(np.round(sorted_distances_public.size*0.9))
        eval_distances_public = sorted_distances_public[0:cutoff_public]
        
        _result_object = {
            "score": self._rmse(eval_distances_public),
            "score_secondary": coverage,
            "meta": {
                "private_score": self._rmse(eval_distances),
                "private_score_secondary": coverage    
            }
        }
        

        return _result_object
