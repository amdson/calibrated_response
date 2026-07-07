"""
Forecast entries are of the format: 
{'id': '0x641f7bc48e2a81d1483fc40d05e9be1abc44d60a83b283142ab6a7d101e73232',
  'source': 'polymarket',
  'question': 'Will the San Antonio Spurs have the worst record in the NBA?',
  'resolution_criteria': 'Resolves to the outcome of the question found at https://polymarket.com/market/will-the-san-antonio-spurs-have-the-worst-record-in-the-nba.',
  'background': 'This market will resolve according to the team that finishes with the worst regular season record in the NBA for the 2025-2026 Season.\n\nIf multiple teams finish with identical records, the league’s tiebreaker rules will be used to determine the worst record.\n\nIf it becomes impossible for the listed team to finish with the worst record based on the rules of the NBA, the market for that team may resolve to “No”.\n\nThe primary resolution source for this market is official information from the NBA',
  'market_info_open_datetime': '2025-10-22',
  'market_info_close_datetime': '2026-04-12T00:00:00+00:00',
  'market_info_resolution_criteria': 'N/A',
  'url': 'https://polymarket.com/market/will-the-san-antonio-spurs-have-the-worst-record-in-the-nba',
  'freeze_datetime': '2026-01-22T00:00:00+00:00',
  'freeze_datetime_value': '0.023',
  'freeze_datetime_value_explanation': 'The market price.',
  'source_intro': "We would like you to predict the outcome of a prediction market. A prediction market, in this context, is the aggregate of predictions submitted by users on the website Polymarket. You're going to predict the probability that the market will resolve as 'Yes'.",
  'resolution_dates': 'N/A',
  'resolution_date': '2026-02-20',
  'resolved_to': 0.0},
  
  1. Based on question, background, and resolution_date entries
"""

prediction_date = "2025-09-09T00:00:00Z"
resolution_date = "2026-02-08T00:00:00Z"

def iter_dataset(dataset, resolution_date: str = resolution_date, prediction_date: str=prediction_date):
    for i, entry in enumerate(dataset):
      question_details = {
          "title": entry["question"],
          "resolution_criteria": entry["resolution_criteria"],
          "description": entry["background"],
          "fine_print": entry.get("source_intro", ""),
      }

      question = entry["question"]
      question = question.replace("{resolution_date}", resolution_date)
      question = question.replace("{forecast_due_date}", prediction_date)
      question_details["title"] = question
      yield question_details