import pandas as pd
import numpy as np

class AircraftLocalization:
  def __init__(self, answer_file_path, round=1):

    self.answer_file_path = answer_file_path
    self.round = round

    # Return haversine distance
  def _haversine(self, lat1, lon1, lat2, lon2):
    R = 6372800  # Earth radius in meters
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2) 
    dphi       = np.radians(lat2 - lat1)
    dlambda    = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + \
        np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1 - a))

  # Return RMSE
  def _rmse(self, haversine_distance):
    return np.sqrt((haversine_distance ** 2).mean())

  # Return error if less than 50% predicted
  def _checkcoverage(self, submission):
    return 1 - ((len(submission.latitude) - submission.latitude.count()) / len(submission.latitude)) 
    
               
  def _evaluate(self, client_payload, _context={}):

    submission_file_path = client_payload["submission_file_path"]
    
    submission = pd.read_csv(submission_file_path)
    gt_data = pd.read_csv(self.answer_file_path)
    
    # Or your preferred way to read your submission

    if not len(submission.index) == len(gt_data.index):
        raise Exception("Wrong format, number of data rows needs to be: " + str(len(gt_data.index)))
        
    if not len(submission.columns) == len(gt_data.columns):
        raise Exception("Wrong format, number of columns needs to be: " + str(len(gt_data.columns)))

    coverage = self._checkcoverage(submission)
    
    if coverage < 0.5:
        raise Exception("Less than 50% coverage, submission not eligible. Coverage: " + str(100*coverage) + "%")
        
    # Get haversine distances    
    distances = self._haversine(gt_data.latitude,gt_data.longitude,submission.latitude,submission.longitude)
    
    # Sort, remove NaNs
    sorted_distances = distances.sort_values().dropna()  
    
    # Truncate worst 10%, get RSME
    cutoff = int(np.round(sorted_distances.size*0.9))
    eval_distances = sorted_distances[0:cutoff]

    
    _result_object = {
        "score": self._rmse(eval_distances),
        "score_secondary" : coverage
        
        
    }
    
    media_dir = '/tmp/'

    return _result_object


  
        
if __name__ == "__main__":
    answer_file_path = "data/competition/round1/round1_ground_truth.csv"
    _client_payload = {}
    _client_payload["submission_file_path"] = "data/sample-submissions/round1/winners/5th-place-submission75307.csv"

    
    # Instantiate a dummy context
    _context = {}
    # Instantiate an evaluator
    evaluator = AircraftLocalization(answer_file_path)
    # Evaluate
    result = evaluator._evaluate(_client_payload, _context)
    print(result)
